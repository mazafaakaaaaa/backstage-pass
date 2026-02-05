package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// MCPTool represents an available MCP tool
type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// MCPClientWrapper wraps an MCP client instance for a specific user
type MCPClientWrapper struct {
	Token      string
	HTTPClient *http.Client
	mu         sync.RWMutex

	// MCP protocol communication
	requestID   int64
	initialized bool
	mcpURL      string
	sessionID   string // Session ID from the MCP server

	// Available tools from MCP server
	availableTools map[string]*MCPTool
	toolsMu        sync.RWMutex
}

// MCPRequest represents an MCP JSON-RPC request
type MCPRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int64       `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// MCPResponse represents an MCP JSON-RPC response
type MCPResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *MCPError       `json:"error,omitempty"`
}

// MCPError represents an MCP error
type MCPError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// NewMCPClientWrapper creates a new MCP client wrapper for a user
// It connects to the remote GitHub MCP server via HTTP API
// The MCP server URL and configuration are loaded from mcp.json (required)
func NewMCPClientWrapper(ctx context.Context, githubToken string) (*MCPClientWrapper, error) {
	// Load MCP configuration from mcp.json
	mcpConfig, err := LoadMCPConfig("mcp.json")
	if err != nil {
		return nil, fmt.Errorf("failed to load mcp.json: %w", err)
	}

	// Get GitHub MCP server config
	githubServer, err := mcpConfig.GetGitHubMCPServer()
	if err != nil {
		return nil, fmt.Errorf("failed to get GitHub server from mcp.json: %w", err)
	}

	// Inject the user's GitHub token into the config
	githubServer.InjectToken(githubToken)

	// Validate that we have a URL for HTTP transport
	if githubServer.URL == "" {
		return nil, fmt.Errorf("GitHub MCP server URL not found in mcp.json")
	}

	log.Printf("Loaded MCP config from mcp.json: URL=%s, TransportType=%s", githubServer.URL, githubServer.TransportType)

	return newMCPClientWrapperWithURL(ctx, githubToken, githubServer.URL)
}

// newMCPClientWrapperWithURL creates a new MCP client with a specific URL
func newMCPClientWrapperWithURL(ctx context.Context, githubToken, mcpURL string) (*MCPClientWrapper, error) {
	wrapper := &MCPClientWrapper{
		Token: githubToken,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		requestID:      1,
		availableTools: make(map[string]*MCPTool),
		mcpURL:         mcpURL,
		initialized:    false,
	}

	// Initialize MCP connection
	if err := wrapper.initializeMCP(ctx); err != nil {
		log.Printf("Failed to initialize MCP server, falling back to direct API calls: %v", err)
		// Fall back to direct GitHub API calls - still return the wrapper
		return wrapper, nil
	}

	log.Printf("MCP client wrapper created for user with URL: %s", mcpURL)
	return wrapper, nil
}

// initializeMCP performs the MCP handshake
func (m *MCPClientWrapper) initializeMCP(ctx context.Context) error {
	// Send initialize request
	initReq := MCPRequest{
		JSONRPC: "2.0",
		ID:      m.getNextRequestID(),
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "mcp-chatbot",
				"version": "1.0.0",
			},
		},
	}

	resp, err := m.sendRequest(ctx, initReq)
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}

	if resp.Error != nil {
		return fmt.Errorf("initialize error: %s", resp.Error.Message)
	}

	// Try to extract session ID from response if not already set
	// The session ID might be in the response body or we might have gotten it from the header
	m.mu.RLock()
	hasSessionID := m.sessionID != ""
	m.mu.RUnlock()

	if !hasSessionID && resp.Result != nil {
		// Try to parse session ID from result
		var result map[string]interface{}
		if err := json.Unmarshal(resp.Result, &result); err == nil {
			if sessionID, ok := result["sessionId"].(string); ok && sessionID != "" {
				m.mu.Lock()
				m.sessionID = sessionID
				m.mu.Unlock()
				log.Printf("Extracted session ID from initialize response: %s", sessionID)
			}
		}
	}

	// Send initialized notification (only if we have a session ID or if the server doesn't require it)
	// Some MCP servers might not require this notification
	initializedReq := MCPRequest{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}

	if err := m.sendNotification(initializedReq); err != nil {
		// Log the error but don't fail initialization - some servers might not require this
		log.Printf("Warning: initialized notification failed (may not be required): %v", err)
	}

	// Give the server a moment to process the initialized notification
	time.Sleep(200 * time.Millisecond)

	// List available tools from the MCP server
	log.Printf("Attempting to list available tools from MCP server...")
	if err := m.listAvailableTools(ctx); err != nil {
		log.Printf("Warning: Failed to list available tools during initialization: %v", err)
		// Try again after a short delay
		time.Sleep(500 * time.Millisecond)
		log.Printf("Retrying to list available tools...")
		if err := m.listAvailableTools(ctx); err != nil {
			log.Printf("Error: Failed to list available tools after retry: %v", err)
			// Don't fail initialization if tool listing fails, but log the error
		} else {
			log.Printf("Successfully listed tools on retry")
		}
	} else {
		log.Printf("Successfully listed tools during initialization")
	}

	// Mark as initialized after successful initialization
	m.mu.Lock()
	m.initialized = true
	m.mu.Unlock()
	log.Printf("MCP client initialized successfully")

	return nil
}

// listAvailableTools queries the MCP server for available tools
func (m *MCPClientWrapper) listAvailableTools(ctx context.Context) error {
	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      m.getNextRequestID(),
		Method:  "tools/list",
	}

	log.Printf("Sending tools/list request to MCP server via HTTP...")
	resp, err := m.sendRequest(ctx, req)
	if err != nil {
		log.Printf("Error sending tools/list request: %v", err)
		return fmt.Errorf("failed to list tools: %w", err)
	}

	if resp.Error != nil {
		log.Printf("MCP server returned error for tools/list: %s", resp.Error.Message)
		return fmt.Errorf("tools/list error: %s", resp.Error.Message)
	}

	if resp.Result == nil {
		log.Printf("MCP server returned nil result for tools/list")
		return fmt.Errorf("tools/list returned nil result")
	}

	var result struct {
		Tools []*MCPTool `json:"tools"`
	}

	if err := json.Unmarshal(resp.Result, &result); err != nil {
		log.Printf("Error parsing tools/list result: %v, raw result: %s", err, string(resp.Result))
		return fmt.Errorf("failed to parse tools list: %w", err)
	}

	// Store available tools
	m.toolsMu.Lock()
	previousCount := len(m.availableTools)
	for _, tool := range result.Tools {
		m.availableTools[tool.Name] = tool
		log.Printf("Available MCP tool: %s - %s", tool.Name, tool.Description)
	}
	m.toolsMu.Unlock()

	log.Printf("Found %d available MCP tools (was %d before)", len(result.Tools), previousCount)
	return nil
}

// GetAvailableTools returns the list of available tools
func (m *MCPClientWrapper) GetAvailableTools() []*MCPTool {
	m.toolsMu.RLock()
	defer m.toolsMu.RUnlock()

	tools := make([]*MCPTool, 0, len(m.availableTools))
	for _, tool := range m.availableTools {
		tools = append(tools, tool)
	}
	return tools
}

// ReloadTools attempts to reload tools from the MCP server
func (m *MCPClientWrapper) ReloadTools(ctx context.Context) error {
	log.Printf("ReloadTools called - attempting to list tools from MCP server")

	// If not initialized, try to initialize first
	m.mu.RLock()
	initialized := m.initialized
	m.mu.RUnlock()

	if !initialized {
		log.Printf("MCP client not initialized, attempting to initialize...")
		if err := m.initializeMCP(ctx); err != nil {
			log.Printf("Failed to initialize MCP: %v", err)
			return fmt.Errorf("MCP client not initialized: %w", err)
		}
		log.Printf("MCP client initialized successfully")
	}

	return m.listAvailableTools(ctx)
}

// IsInitialized returns whether the MCP client is initialized
func (m *MCPClientWrapper) IsInitialized() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.initialized
}

// HasStdin returns whether stdin pipe is available (always false for HTTP-based MCP)
func (m *MCPClientWrapper) HasStdin() bool {
	return false
}

// HasStdout returns whether stdout pipe is available (always false for HTTP-based MCP)
func (m *MCPClientWrapper) HasStdout() bool {
	return false
}

// CallTool calls a specific MCP tool by name with the given arguments
func (m *MCPClientWrapper) CallTool(ctx context.Context, toolName string, arguments map[string]interface{}) (map[string]interface{}, error) {
	m.toolsMu.RLock()
	_, exists := m.availableTools[toolName]
	m.toolsMu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("tool '%s' not found. Available tools: %v", toolName, m.getToolNames())
	}

	// Add timeout to context if not already present
	callCtx := ctx
	if _, hasTimeout := ctx.Deadline(); !hasTimeout {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		log.Printf("Added 30-second timeout to tool call context for %s", toolName)
	}

	log.Printf("About to create MCP request for tool %s", toolName)

	// Get request ID without holding any locks
	requestID := m.getNextRequestID()
	log.Printf("Got request ID %d for tool %s", requestID, toolName)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      requestID,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name":      toolName,
			"arguments": arguments,
		},
	}

	log.Printf("Calling tool %s with timeout context...", toolName)
	startTime := time.Now()
	resp, err := m.sendRequest(callCtx, req)
	elapsed := time.Since(startTime)
	log.Printf("Tool %s call completed in %v (err: %v)", toolName, elapsed, err)

	if err != nil {
		if callCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("tool %s timed out after 30 seconds", toolName)
		}
		return nil, fmt.Errorf("failed to call tool %s: %w", toolName, err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("tool %s error: %s", toolName, resp.Error.Message)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("failed to parse tool result: %w", err)
	}

	// Log the result structure for debugging
	log.Printf("Successfully called MCP tool: %s", toolName)
	log.Printf("Tool result keys: %v", getMapKeys(result))

	// Log result size for arrays
	for key, value := range result {
		if arr, ok := value.([]interface{}); ok {
			log.Printf("Tool result '%s' contains %d items", key, len(arr))
			if len(arr) == 0 {
				log.Printf("WARNING: Tool %s returned empty array for key '%s'", toolName, key)
			}
		}
	}

	// Log the full result structure for debugging (first 500 chars)
	resultJSON, _ := json.Marshal(result)
	if len(resultJSON) > 500 {
		log.Printf("Tool %s full result (first 500 chars): %s", toolName, string(resultJSON[:500]))
	} else {
		log.Printf("Tool %s full result: %s", toolName, string(resultJSON))
	}

	// Check if the result indicates an error
	if isError, ok := result["isError"].(bool); ok && isError {
		// Extract error message from content
		if content, ok := result["content"].([]interface{}); ok && len(content) > 0 {
			if item, ok := content[0].(map[string]interface{}); ok {
				if text, ok := item["text"].(string); ok {
					return nil, fmt.Errorf("%s", text)
				}
			}
		}
		return nil, fmt.Errorf("tool returned an error")
	}

	return result, nil
}

// getToolNames returns a list of available tool names
func (m *MCPClientWrapper) getToolNames() []string {
	m.toolsMu.RLock()
	defer m.toolsMu.RUnlock()

	names := make([]string, 0, len(m.availableTools))
	for name := range m.availableTools {
		names = append(names, name)
	}
	return names
}

// sendRequest sends an MCP request via HTTP and waits for a response
func (m *MCPClientWrapper) sendRequest(ctx context.Context, req MCPRequest) (*MCPResponse, error) {
	if req.ID == 0 {
		req.ID = m.getNextRequestID()
	}

	m.mu.RLock()
	mcpURL := m.mcpURL
	token := m.Token
	m.mu.RUnlock()

	// Marshal request
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	log.Printf("Sending MCP request via HTTP: method=%s, id=%d, url=%s", req.Method, req.ID, mcpURL)

	// Check if context has a deadline
	if deadline, ok := ctx.Deadline(); ok {
		timeUntilDeadline := time.Until(deadline)
		log.Printf("Context deadline: %v (in %v)", deadline, timeUntilDeadline)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", mcpURL, strings.NewReader(string(data)))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	// Include session ID if we have one
	m.mu.RLock()
	sessionID := m.sessionID
	m.mu.RUnlock()
	if sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", sessionID)
		log.Printf("Including session ID in request: %s", sessionID)
	}

	// Send request with logging
	log.Printf("Sending HTTP request to %s...", mcpURL)
	log.Printf("Request headers: Authorization=Bearer *****, Mcp-Session-Id=%s", sessionID)
	log.Printf("Request body: %s", string(data))
	startTime := time.Now()
	resp, err := m.HTTPClient.Do(httpReq)
	elapsed := time.Since(startTime)

	if err != nil {
		log.Printf("HTTP request error after %v: %v", elapsed, err)
		// Check if it's a timeout
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("HTTP request timed out after %v", elapsed)
		}
		return nil, fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("HTTP request completed in %v, status: %d", elapsed, resp.StatusCode)

	// Read response
	log.Printf("Reading response body...")
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading response body: %v", err)
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	log.Printf("Read %d bytes from response body", len(body))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("MCP server returned status %d: %s", resp.StatusCode, string(body))
	}

	// Check for session ID in response header
	if sessionIDHeader := resp.Header.Get("Mcp-Session-Id"); sessionIDHeader != "" {
		m.mu.Lock()
		m.sessionID = sessionIDHeader
		m.mu.Unlock()
		log.Printf("Received session ID from MCP server: %s", sessionIDHeader)
	}

	// Try to parse as plain JSON first (most common case)
	jsonData := body
	var mcpResp MCPResponse

	// First attempt: try parsing body directly as JSON
	if err := json.Unmarshal(jsonData, &mcpResp); err != nil {
		// If JSON parsing fails, check if it might be SSE format
		contentType := resp.Header.Get("Content-Type")
		if strings.Contains(contentType, "text/event-stream") {
			log.Printf("JSON parse failed, trying SSE format extraction...")
			bodyStr := string(body)
			lines := strings.Split(bodyStr, "\n")
			var jsonLines []string
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "data: ") {
					jsonStr := strings.TrimPrefix(line, "data: ")
					jsonStr = strings.TrimSpace(jsonStr)
					if jsonStr != "" && (strings.HasPrefix(jsonStr, "{") || strings.HasPrefix(jsonStr, "[")) {
						jsonLines = append(jsonLines, jsonStr)
					}
				}
			}
			if len(jsonLines) > 0 {
				jsonData = []byte(jsonLines[len(jsonLines)-1])
				log.Printf("Extracted JSON from SSE: %d bytes", len(jsonData))
				// Try parsing again with extracted JSON
				if err := json.Unmarshal(jsonData, &mcpResp); err != nil {
					previewLen := 500
					if len(body) < previewLen {
						previewLen = len(body)
					}
					log.Printf("Failed to parse JSON even after SSE extraction. Full body preview: %s", string(body[:previewLen]))
					return nil, fmt.Errorf("failed to parse MCP response: %w", err)
				}
			} else {
				// No SSE data found, return original error
				previewLen := 500
				if len(body) < previewLen {
					previewLen = len(body)
				}
				log.Printf("Failed to parse JSON and no SSE data found. Full body preview: %s", string(body[:previewLen]))
				return nil, fmt.Errorf("failed to parse MCP response: %w", err)
			}
		} else {
			// Not SSE format, return original error
			previewLen := 500
			if len(body) < previewLen {
				previewLen = len(body)
			}
			log.Printf("Failed to parse JSON. Full body preview: %s", string(body[:previewLen]))
			return nil, fmt.Errorf("failed to parse MCP response: %w", err)
		}
	}

	log.Printf("Received MCP response: id=%d, hasError=%v", mcpResp.ID, mcpResp.Error != nil)
	return &mcpResp, nil
}

// sendNotification sends an MCP notification via HTTP (no response expected)
func (m *MCPClientWrapper) sendNotification(req MCPRequest) error {
	m.mu.RLock()
	mcpURL := m.mcpURL
	token := m.Token
	m.mu.RUnlock()

	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal notification: %w", err)
	}

	ctx := context.Background()
	httpReq, err := http.NewRequestWithContext(ctx, "POST", mcpURL, strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	// Include session ID if we have one
	m.mu.RLock()
	sessionID := m.sessionID
	m.mu.RUnlock()
	if sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", sessionID)
	}

	resp, err := m.HTTPClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send notification: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("notification returned status %d: %s", resp.StatusCode, string(body))
	}

	// Check for session ID in response header
	if sessionIDHeader := resp.Header.Get("Mcp-Session-Id"); sessionIDHeader != "" {
		m.mu.Lock()
		m.sessionID = sessionIDHeader
		m.mu.Unlock()
		log.Printf("Received session ID from MCP server in notification response: %s", sessionIDHeader)
	}

	return nil
}

// getNextRequestID returns the next request ID
func (m *MCPClientWrapper) getNextRequestID() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := m.requestID
	m.requestID++
	return id
}

// GetContext uses the MCP client to get relevant context for a user query
func (m *MCPClientWrapper) GetContext(ctx context.Context, query string) (string, error) {
	queryLower := strings.ToLower(query)

	// Check if user is asking about available tools
	if strings.Contains(queryLower, "available tools") ||
		strings.Contains(queryLower, "what tools") ||
		strings.Contains(queryLower, "list tools") ||
		strings.Contains(queryLower, "show tools") ||
		strings.Contains(queryLower, "what can you do") ||
		strings.Contains(queryLower, "what commands") ||
		strings.Contains(queryLower, "help") {
		return m.listAvailableToolsForUser(), nil
	}

	// Check if user is asking about their account
	if strings.Contains(queryLower, "my github") ||
		strings.Contains(queryLower, "my account") ||
		strings.Contains(queryLower, "who am i") ||
		strings.Contains(queryLower, "what is my") {
		// Get authenticated user's account info
		userInfo, err := m.getAuthenticatedUserInfo(ctx)
		if err == nil && userInfo != "" {
			return userInfo, nil
		}
	}

	// Check if MCP server is initialized (read lock only for this check)
	m.mu.RLock()
	initialized := m.initialized
	m.mu.RUnlock()

	if initialized {
		log.Printf("MCP server is initialized, calling getMCPContext for query: %s", query)
		result, err := m.getMCPContext(ctx, query)
		if err != nil {
			log.Printf("getMCPContext returned error: %v", err)
			// If we're looking for issues and got an error, provide helpful message
			if strings.Contains(strings.ToLower(query), "issue") {
				// Check if error is about missing repo
				if strings.Contains(err.Error(), "missing required parameter: owner") ||
					strings.Contains(err.Error(), "No repository extracted") {
					return "", fmt.Errorf("Please specify which repository you want to get issues from. Try:\n- \"get issues for repo owner/repo-name\"\n- \"show issues for https://github.com/owner/repo-name\"")
				}
				return "", fmt.Errorf("failed to fetch issues: %w", err)
			}
		}
		return result, err
	}

	// Fallback to direct GitHub API calls (shouldn't reach here if initialization succeeded)
	log.Printf("MCP server still not initialized after retry, falling back to direct GitHub API")
	return m.getGitHubContext(ctx, query)
}

// ListAvailableToolsForUser returns a formatted list of available tools for the user (public method)
func (m *MCPClientWrapper) ListAvailableToolsForUser() string {
	return m.listAvailableToolsForUser()
}

// listAvailableToolsForUser returns a formatted list of available tools for the user
func (m *MCPClientWrapper) listAvailableToolsForUser() string {
	availableTools := m.GetAvailableTools()

	if len(availableTools) == 0 {
		// Try to reload tools if none are available
		log.Printf("No tools found, attempting to reload tools...")
		ctx := context.Background()
		if err := m.listAvailableTools(ctx); err != nil {
			log.Printf("Failed to reload tools: %v", err)
			return fmt.Sprintf("No MCP tools are currently available. The MCP server may not be initialized or no tools were found.\n\nError: %v\n\nPlease check server logs for more details.", err)
		}
		// Try again after reload
		availableTools = m.GetAvailableTools()
		if len(availableTools) == 0 {
			return "No MCP tools are currently available. The MCP server may not be initialized or no tools were found. Please check server logs for errors."
		}
		log.Printf("Successfully reloaded %d tools", len(availableTools))
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("Available GitHub MCP Tools (%d total):\n\n", len(availableTools)))

	for _, tool := range availableTools {
		result.WriteString(fmt.Sprintf("• %s\n", tool.Name))
	}

	return result.String()
}

// getAuthenticatedUserInfo gets the authenticated user's GitHub account information
func (m *MCPClientWrapper) getAuthenticatedUserInfo(ctx context.Context) (string, error) {
	// Try MCP tool first
	if m.initialized {
		// Try common tool names for getting user info
		toolNames := []string{"get_user", "get_authenticated_user", "github_get_user", "user_info"}
		for _, toolName := range toolNames {
			m.toolsMu.RLock()
			_, exists := m.availableTools[toolName]
			m.toolsMu.RUnlock()

			if exists {
				result, err := m.CallTool(ctx, toolName, map[string]interface{}{})
				if err == nil {
					return m.formatUserInfo(result), nil
				}
			}
		}
	}

	// Fall back to direct GitHub API
	req, err := http.NewRequestWithContext(ctx, "GET", "https://api.github.com/user", nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", fmt.Sprintf("token %s", m.Token))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GitHub API error: %s", string(body))
	}

	var user struct {
		Login     string `json:"login"`
		Name      string `json:"name"`
		Email     string `json:"email"`
		Bio       string `json:"bio"`
		Company   string `json:"company"`
		Location  string `json:"location"`
		Blog      string `json:"blog"`
		AvatarURL string `json:"avatar_url"`
		HTMLURL   string `json:"html_url"`
		Type      string `json:"type"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return "", err
	}

	info := "Authenticated GitHub User Information:\n"
	info += fmt.Sprintf("- Username: %s\n", user.Login)
	if user.Name != "" {
		info += fmt.Sprintf("- Name: %s\n", user.Name)
	}
	if user.Email != "" {
		info += fmt.Sprintf("- Email: %s\n", user.Email)
	}
	if user.Bio != "" {
		info += fmt.Sprintf("- Bio: %s\n", user.Bio)
	}
	if user.Company != "" {
		info += fmt.Sprintf("- Company: %s\n", user.Company)
	}
	if user.Location != "" {
		info += fmt.Sprintf("- Location: %s\n", user.Location)
	}
	if user.Blog != "" {
		info += fmt.Sprintf("- Website: %s\n", user.Blog)
	}
	info += fmt.Sprintf("- Profile URL: %s\n", user.HTMLURL)
	info += fmt.Sprintf("- Type: %s\n", user.Type)

	return info, nil
}

// formatUserInfo formats user info from MCP tool result
func (m *MCPClientWrapper) formatUserInfo(result map[string]interface{}) string {
	info := "Authenticated GitHub User Information:\n"

	if login, ok := result["login"].(string); ok {
		info += fmt.Sprintf("- Username: %s\n", login)
	}
	if name, ok := result["name"].(string); ok && name != "" {
		info += fmt.Sprintf("- Name: %s\n", name)
	}
	if email, ok := result["email"].(string); ok && email != "" {
		info += fmt.Sprintf("- Email: %s\n", email)
	}
	if bio, ok := result["bio"].(string); ok && bio != "" {
		info += fmt.Sprintf("- Bio: %s\n", bio)
	}
	if company, ok := result["company"].(string); ok && company != "" {
		info += fmt.Sprintf("- Company: %s\n", company)
	}
	if location, ok := result["location"].(string); ok && location != "" {
		info += fmt.Sprintf("- Location: %s\n", location)
	}
	if htmlURL, ok := result["html_url"].(string); ok {
		info += fmt.Sprintf("- Profile URL: %s\n", htmlURL)
	}

	return info
}

// getMCPContext uses the MCP server to get context based on user query
func (m *MCPClientWrapper) getMCPContext(ctx context.Context, query string) (string, error) {
	// Get all available tools
	availableTools := m.GetAvailableTools()
	queryLower := strings.ToLower(query)
	isIssueQuery := strings.Contains(queryLower, "issue")

	if len(availableTools) == 0 {
		log.Println("No MCP tools available")
		// Don't fall back to repository API if looking for issues
		if isIssueQuery {
			return "", fmt.Errorf("no MCP tools available and cannot fetch issues without tools")
		}
		log.Println("Falling back to direct API")
		return m.getGitHubContext(ctx, query)
	}

	// Log available issue tools for debugging
	if isIssueQuery {
		log.Printf("Query contains 'issue', looking for issue tools...")
		issueTools := []string{}
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") && !strings.Contains(toolNameLower, "sub_issue") {
				issueTools = append(issueTools, tool.Name)
			}
		}
		log.Printf("Found %d issue-related tools: %v", len(issueTools), issueTools)
	}

	// Try to intelligently match the query to available tools
	toolName, args := m.matchQueryToTool(query)

	if toolName != "" {
		log.Printf("Matched query to tool: %s with args: %v", toolName, args)
		result, err := m.CallTool(ctx, toolName, args)
		if err == nil {
			formatted := m.formatToolResult(toolName, result)
			log.Printf("Successfully got result from tool %s: %d chars", toolName, len(formatted))
			return formatted, nil
		}
		log.Printf("Failed to call tool %s: %v, trying other tools", toolName, err)
	}

	// If no match found, try all tools that might be relevant
	// Score tools based on how well they match the query
	// But filter out repository tools if we're looking for issues
	var toolsToScore []*MCPTool
	if isIssueQuery {
		// Only score issue-related tools
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") && !strings.Contains(toolNameLower, "sub_issue") {
				toolsToScore = append(toolsToScore, tool)
			}
		}
		log.Printf("Filtering to %d issue-related tools (query contains 'issue')", len(toolsToScore))
	} else {
		toolsToScore = availableTools
	}

	if len(toolsToScore) > 0 {
		bestTool, bestScore := m.findBestMatchingTool(query, toolsToScore)
		if bestTool != nil && bestScore > 0 {
			log.Printf("Found best matching tool: %s (score: %.2f)", bestTool.Name, bestScore)
			args := m.extractToolArguments(query, bestTool)
			result, err := m.CallTool(ctx, bestTool.Name, args)
			if err == nil {
				formatted := m.formatToolResult(bestTool.Name, result)
				log.Printf("Successfully got result from best matching tool %s: %d chars", bestTool.Name, len(formatted))
				return formatted, nil
			}
			log.Printf("Failed to call best matching tool %s: %v", bestTool.Name, err)
		}
	}

	// Final fallback: try common tool patterns
	result, err := m.tryCommonToolPatterns(ctx, query, availableTools)
	if err == nil && result != "" {
		return result, nil
	}

	// If we're looking for issues and all tools failed, don't fall back to repository API
	if isIssueQuery {
		log.Printf("All issue tools failed, not falling back to repository API")
		return "", fmt.Errorf("failed to fetch issues: no issue tools available or all issue tool calls failed")
	}

	// For non-issue queries, fall back to repository API
	return m.getGitHubContext(ctx, query)
}

// findBestMatchingTool scores all available tools and returns the best match
func (m *MCPClientWrapper) findBestMatchingTool(query string, tools []*MCPTool) (*MCPTool, float64) {
	queryLower := strings.ToLower(query)
	queryWords := strings.Fields(queryLower)

	var bestTool *MCPTool
	bestScore := 0.0

	for _, tool := range tools {
		score := m.scoreToolMatch(queryLower, queryWords, tool)
		if score > bestScore {
			bestScore = score
			bestTool = tool
		}
	}

	return bestTool, bestScore
}

// scoreToolMatch scores how well a tool matches a query
func (m *MCPClientWrapper) scoreToolMatch(queryLower string, queryWords []string, tool *MCPTool) float64 {
	score := 0.0
	toolNameLower := strings.ToLower(tool.Name)
	descLower := strings.ToLower(tool.Description)

	// Exact tool name match (highest score)
	if strings.Contains(queryLower, toolNameLower) {
		score += 10.0
	}

	// Tool name word matches
	for _, word := range queryWords {
		if strings.Contains(toolNameLower, word) {
			score += 5.0
		}
	}

	// Description matches
	for _, word := range queryWords {
		if strings.Contains(descLower, word) {
			score += 2.0
		}
	}

	// Keyword-based matching
	keywordMatches := m.getKeywordMatches(queryLower, toolNameLower, descLower)
	score += float64(keywordMatches) * 3.0

	return score
}

// getKeywordMatches counts keyword matches between query and tool
func (m *MCPClientWrapper) getKeywordMatches(query, toolName, description string) int {
	matches := 0

	// Common keyword mappings
	keywordMap := map[string][]string{
		"issue":      {"issue", "issues", "bug", "bugs", "ticket"},
		"repository": {"repo", "repository", "repositories", "project", "projects"},
		"pull":       {"pull", "pr", "pull request", "merge request", "merge"},
		"file":       {"file", "files", "content", "read", "get file"},
		"search":     {"search", "find", "lookup", "query"},
		"user":       {"user", "users", "account", "profile"},
		"commit":     {"commit", "commits", "history"},
		"branch":     {"branch", "branches"},
		"star":       {"star", "stars", "favorite"},
		"fork":       {"fork", "forks"},
	}

	for keyword, synonyms := range keywordMap {
		queryHasKeyword := false
		for _, synonym := range synonyms {
			if strings.Contains(query, synonym) {
				queryHasKeyword = true
				break
			}
		}

		if queryHasKeyword {
			if strings.Contains(toolName, keyword) || strings.Contains(description, keyword) {
				matches++
			}
		}
	}

	return matches
}

// tryCommonToolPatterns tries common tool patterns as a last resort
func (m *MCPClientWrapper) tryCommonToolPatterns(ctx context.Context, query string, tools []*MCPTool) (string, error) {
	queryLower := strings.ToLower(query)

	// Prioritize based on intent - check for specific actions first
	// Issues (highest priority when "issue" is in query)
	if strings.Contains(queryLower, "issue") {
		for _, tool := range tools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") && !strings.Contains(toolNameLower, "sub_issue") {
				log.Printf("Trying issue tool: %s for query: %s", tool.Name, query)
				result, err := m.callToolWithQuery(ctx, tool, query)
				if err == nil && result != "" {
					return result, nil
				}
				log.Printf("Issue tool %s failed: %v", tool.Name, err)
			}
		}
	}

	// Pull requests
	if strings.Contains(queryLower, "pull") || strings.Contains(queryLower, "pr") {
		for _, tool := range tools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "pull") || strings.Contains(toolNameLower, "pr") {
				log.Printf("Trying pull request tool: %s for query: %s", tool.Name, query)
				result, err := m.callToolWithQuery(ctx, tool, query)
				if err == nil && result != "" {
					return result, nil
				}
				log.Printf("Pull request tool %s failed: %v", tool.Name, err)
			}
		}
	}

	// Repositories (only if no other intent found and NOT looking for issues)
	// Don't try repository tools if query contains "issue" - that's a different intent
	if (strings.Contains(queryLower, "repo") || strings.Contains(queryLower, "repository")) &&
		!strings.Contains(queryLower, "issue") && !strings.Contains(queryLower, "pull") && !strings.Contains(queryLower, "pr") {
		for _, tool := range tools {
			toolNameLower := strings.ToLower(tool.Name)
			if (strings.Contains(toolNameLower, "repo") || strings.Contains(toolNameLower, "repository")) &&
				!strings.Contains(toolNameLower, "search") {
				log.Printf("Trying repository tool: %s for query: %s", tool.Name, query)
				result, err := m.callToolWithQuery(ctx, tool, query)
				if err == nil && result != "" {
					return result, nil
				}
				log.Printf("Repository tool %s failed: %v", tool.Name, err)
			}
		}
	}

	// Final fallback to direct API (only if not looking for issues)
	if strings.Contains(strings.ToLower(query), "issue") {
		log.Printf("All issue tools failed in tryCommonToolPatterns, not falling back to repository API")
		return "", fmt.Errorf("failed to fetch issues: all issue tool attempts failed")
	}
	return m.getGitHubContext(ctx, query)
}

// callToolWithQuery calls a tool with arguments extracted from the query
func (m *MCPClientWrapper) callToolWithQuery(ctx context.Context, tool *MCPTool, query string) (string, error) {
	args := m.extractToolArguments(query, tool)

	// Check if tool requires owner/repo but we don't have it
	if tool.InputSchema != nil {
		if properties, ok := tool.InputSchema["properties"].(map[string]interface{}); ok {
			if _, hasOwner := properties["owner"]; hasOwner {
				if required, ok := tool.InputSchema["required"].([]interface{}); ok {
					for _, req := range required {
						if req == "owner" {
							// Check if owner is in args
							if owner, ok := args["owner"].(string); !ok || owner == "" {
								log.Printf("Tool %s requires 'owner' parameter but no repository was extracted from query: %s", tool.Name, query)
								return "", fmt.Errorf("Please specify which repository you want to get issues from. Try:\n- \"get issues for repo owner/repo-name\"\n- \"show issues for https://github.com/owner/repo-name\"")
							}
						}
					}
				}
			}
		}
	}

	log.Printf("Calling tool %s with args: %v", tool.Name, args)
	result, err := m.CallTool(ctx, tool.Name, args)
	if err != nil {
		log.Printf("Tool %s call failed: %v", tool.Name, err)
		// If error is about missing owner parameter, provide helpful message
		if strings.Contains(err.Error(), "missing required parameter: owner") {
			return "", fmt.Errorf("Please specify which repository you want to get issues from. Try:\n- \"get issues for repo owner/repo-name\"\n- \"show issues for https://github.com/owner/repo-name\"")
		}
		return "", err
	}
	log.Printf("Tool %s call succeeded, result keys: %v", tool.Name, getMapKeys(result))
	formatted := m.formatToolResult(tool.Name, result)
	log.Printf("Formatted result from %s: %d characters", tool.Name, len(formatted))
	return formatted, nil
}

// matchQueryToTool tries to match user query to available MCP tools
func (m *MCPClientWrapper) matchQueryToTool(query string) (string, map[string]interface{}) {
	queryLower := strings.ToLower(query)
	availableTools := m.GetAvailableTools()

	if len(availableTools) == 0 {
		return "", nil
	}

	// Prioritize specific intents - if query contains "issue", prioritize issue tools
	if strings.Contains(queryLower, "issue") {
		log.Printf("Query contains 'issue', searching for issue tools in %d available tools", len(availableTools))
		issueTools := []string{}
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") && !strings.Contains(toolNameLower, "sub_issue") {
				issueTools = append(issueTools, tool.Name)
			}
		}
		log.Printf("Found %d issue tools: %v", len(issueTools), issueTools)

		// Check if user wants to read/analyze a specific issue (e.g., "read issue #13")
		issueNumberPattern := regexp.MustCompile(`#(\d+)`)
		issueNumberMatches := issueNumberPattern.FindStringSubmatch(query)

		if len(issueNumberMatches) > 1 {
			// User is asking about a specific issue - use issue_read to get details + comments
			log.Printf("Detected specific issue number: #%s", issueNumberMatches[1])
			for _, tool := range availableTools {
				if strings.ToLower(tool.Name) == "issue_read" {
					log.Printf("Using issue_read tool for detailed issue analysis")
					args := m.extractToolArguments(query, tool)
					// Add issue number and method
					issueNum, _ := strconv.Atoi(issueNumberMatches[1])
					args["issue_number"] = issueNum
					args["method"] = "get" // Get issue details first
					log.Printf("Extracted arguments for issue_read: %v", args)
					return tool.Name, args
				}
			}
		}

		// Prioritize list_issues and search_issues over list_issue_types
		// First, try to find list_issues or search_issues
		preferredTools := []string{"list_issues", "search_issues", "issue_read"}
		for _, preferredName := range preferredTools {
			for _, tool := range availableTools {
				if strings.ToLower(tool.Name) == preferredName {
					log.Printf("Matched query to preferred issue tool: %s (description: %s)", tool.Name, tool.Description)
					args := m.extractToolArguments(query, tool)
					log.Printf("Extracted arguments for %s: %v", tool.Name, args)
					return tool.Name, args
				}
			}
		}

		// If no preferred tool found, try other issue tools (but exclude list_issue_types)
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") &&
				!strings.Contains(toolNameLower, "sub_issue") &&
				!strings.Contains(toolNameLower, "issue_type") {
				log.Printf("Matched query to issue tool: %s (description: %s)", tool.Name, tool.Description)
				args := m.extractToolArguments(query, tool)
				log.Printf("Extracted arguments for %s: %v", tool.Name, args)
				return tool.Name, args
			}
		}
		log.Printf("WARNING: Query contains 'issue' but no issue tools were matched!")
	}

	// Prioritize pull requests
	if strings.Contains(queryLower, "pull") || strings.Contains(queryLower, "pr") {
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "pull") || strings.Contains(toolNameLower, "pr") {
				log.Printf("Matched query to pull request tool: %s", tool.Name)
				args := m.extractToolArguments(query, tool)
				return tool.Name, args
			}
		}
	}

	// Enhanced matching: score all tools and pick the best
	// But filter out repository tools if we're looking for issues
	var toolsToScore []*MCPTool
	if strings.Contains(queryLower, "issue") {
		// Only score issue-related tools
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			if strings.Contains(toolNameLower, "issue") && !strings.Contains(toolNameLower, "sub_issue") {
				toolsToScore = append(toolsToScore, tool)
			}
		}
	} else {
		toolsToScore = availableTools
	}

	if len(toolsToScore) > 0 {
		bestTool, bestScore := m.findBestMatchingTool(query, toolsToScore)
		if bestTool != nil && bestScore > 0 {
			args := m.extractToolArguments(query, bestTool)
			return bestTool.Name, args
		}
	}

	// Fallback: simple keyword matching (but skip if we already tried issue tools)
	if !strings.Contains(queryLower, "issue") {
		for _, tool := range availableTools {
			toolNameLower := strings.ToLower(tool.Name)
			descLower := strings.ToLower(tool.Description)

			// Check if query matches tool name or description
			if strings.Contains(queryLower, toolNameLower) ||
				strings.Contains(descLower, queryLower) ||
				m.queryMatchesTool(queryLower, toolNameLower, descLower) {

				args := m.extractToolArguments(query, tool)
				return tool.Name, args
			}
		}
	}

	return "", nil
}

// queryMatchesTool checks if query matches a tool based on keywords
func (m *MCPClientWrapper) queryMatchesTool(query, toolName, description string) bool {
	// Common mappings
	mappings := map[string][]string{
		"issue":      {"issue", "issues", "bug", "bugs"},
		"repository": {"repo", "repository", "repositories", "project", "projects"},
		"pull":       {"pull", "pr", "pull request", "merge"},
		"file":       {"file", "files", "content", "read"},
		"search":     {"search", "find", "lookup"},
	}

	// Check if query contains keywords that match tool purpose
	for keyword, synonyms := range mappings {
		if strings.Contains(toolName, keyword) || strings.Contains(description, keyword) {
			for _, synonym := range synonyms {
				if strings.Contains(query, synonym) {
					return true
				}
			}
		}
	}

	return false
}

// extractToolArguments extracts arguments for a tool from the query
func (m *MCPClientWrapper) extractToolArguments(query string, tool *MCPTool) map[string]interface{} {
	args := make(map[string]interface{})

	// Log the tool's input schema to see what parameters it expects
	if tool.InputSchema != nil {
		schemaJSON, _ := json.Marshal(tool.InputSchema)
		log.Printf("Tool %s input schema: %s", tool.Name, string(schemaJSON))
	}

	// Extract common arguments from query
	repoName := extractRepoFromQuery(query)
	log.Printf("Extracting arguments for tool %s from query: %s, extracted repo: '%s'", tool.Name, query, repoName)

	if repoName != "" && repoName != "repo" {
		// Split owner/repo if needed
		parts := strings.Split(repoName, "/")
		if len(parts) == 2 {
			args["repository"] = repoName // Full owner/repo format
			args["owner"] = parts[0]
			args["repo"] = parts[1]
			log.Printf("Split repo into owner=%s, repo=%s", parts[0], parts[1])
		} else {
			// If it's not in owner/repo format and it's not a valid repo name, don't add it
			if repoName != "repo" && repoName != "repository" {
				args["repository"] = repoName
				args["owner"] = "" // Will use authenticated user
				log.Printf("Repo not in owner/repo format, using as-is: %s", repoName)
			} else {
				log.Printf("WARNING: Extracted invalid repo name '%s', skipping repository arguments", repoName)
			}
		}
	} else {
		log.Printf("WARNING: No repository extracted from query: %s (extracted: '%s')", query, repoName)
		// For issue-related tools that require owner/repo, check if owner is required
		toolNameLower := strings.ToLower(tool.Name)
		if strings.Contains(toolNameLower, "issue") && tool.InputSchema != nil {
			if properties, ok := tool.InputSchema["properties"].(map[string]interface{}); ok {
				if _, hasOwner := properties["owner"]; hasOwner {
					// Check if owner is required
					if required, ok := tool.InputSchema["required"].([]interface{}); ok {
						for _, req := range required {
							if req == "owner" {
								log.Printf("Tool %s requires 'owner' parameter but no repository was extracted from query", tool.Name)
								// Don't set owner here - let the tool call fail with a clear error
								// The error handling will provide a better message to the user
							}
						}
					}
				}
			}
		}
	}

	// For issue-related tools, default to "open" if not specified
	toolNameLower := strings.ToLower(tool.Name)
	if strings.Contains(toolNameLower, "issue") {
		if strings.Contains(strings.ToLower(query), "closed") {
			args["state"] = "CLOSED" // GitHub GraphQL expects uppercase
			log.Printf("Setting state to 'CLOSED' for issue tool")
		} else {
			// Default to open issues if not specified
			args["state"] = "OPEN" // GitHub GraphQL expects uppercase
			log.Printf("Setting state to 'OPEN' for issue tool (default)")
		}
	} else if strings.Contains(toolNameLower, "pull") {
		// For PR tools, default to "open" if not specified
		if strings.Contains(strings.ToLower(query), "closed") {
			args["state"] = "CLOSED" // GitHub GraphQL expects uppercase
		} else {
			args["state"] = "OPEN" // GitHub GraphQL expects uppercase
		}
	}

	// Add pagination defaults only if the tool schema expects them
	// Check if the tool schema has these properties
	if tool.InputSchema != nil {
		if properties, ok := tool.InputSchema["properties"].(map[string]interface{}); ok {
			// Only add perPage if the schema expects it
			if _, hasPerPage := properties["perPage"]; hasPerPage {
				args["perPage"] = 100
			}
			// Only add page if the schema expects it
			if _, hasPage := properties["page"]; hasPage {
				args["page"] = 1
			}
		}
	} else {
		// If no schema, add defaults anyway
		args["perPage"] = 100
		args["page"] = 1
	}

	log.Printf("Final arguments for %s: %v", tool.Name, args)
	return args
}

// formatToolResult formats the tool result into a readable string
func (m *MCPClientWrapper) formatToolResult(toolName string, result map[string]interface{}) string {
	log.Printf("Formatting tool result from %s, result keys: %v", toolName, getMapKeys(result))

	// Log full result for debugging
	resultJSON, _ := json.Marshal(result)
	if len(resultJSON) > 1000 {
		log.Printf("Full result from %s (first 1000 chars): %s", toolName, string(resultJSON[:1000]))
	} else {
		log.Printf("Full result from %s: %s", toolName, string(resultJSON))
	}

	// Try to format based on common result structures
	var context strings.Builder

	// Handle different result formats
	if items, ok := result["items"].([]interface{}); ok {
		log.Printf("Found %d items in result", len(items))
		context.WriteString(fmt.Sprintf("GitHub Issues (%d found):\n\n", len(items)))
		for _, item := range items {
			if itemMap, ok := item.(map[string]interface{}); ok {
				context.WriteString(m.formatItem(itemMap))
			}
		}
	} else if issues, ok := result["issues"].([]interface{}); ok {
		log.Printf("Found %d issues in result", len(issues))
		context.WriteString(fmt.Sprintf("GitHub Issues (%d found):\n\n", len(issues)))
		for _, issue := range issues {
			if issueMap, ok := issue.(map[string]interface{}); ok {
				context.WriteString(m.formatItem(issueMap))
			}
		}
	} else if repos, ok := result["repositories"].([]interface{}); ok {
		log.Printf("Found %d repositories in result", len(repos))
		context.WriteString(fmt.Sprintf("GitHub Repositories (%d found):\n\n", len(repos)))
		for _, repo := range repos {
			if repoMap, ok := repo.(map[string]interface{}); ok {
				context.WriteString(m.formatItem(repoMap))
			}
		}
	} else {
		// Generic formatting - log what we got
		log.Printf("No recognized array format found, using generic formatting")
		context.WriteString(fmt.Sprintf("GitHub data from %s:\n\n", toolName))
		for key, value := range result {
			// Try to format arrays generically
			if arr, ok := value.([]interface{}); ok {
				context.WriteString(fmt.Sprintf("%s (%d items):\n", key, len(arr)))
				for i, item := range arr {
					if itemMap, ok := item.(map[string]interface{}); ok {
						context.WriteString(fmt.Sprintf("  [%d] ", i+1))
						context.WriteString(m.formatItem(itemMap))
					} else {
						context.WriteString(fmt.Sprintf("  [%d] %v\n", i+1, item))
					}
				}
			} else {
				context.WriteString(fmt.Sprintf("- %s: %v\n", key, value))
			}
		}
	}

	formatted := context.String()
	log.Printf("Formatted result length: %d characters", len(formatted))
	return formatted
}

// getMapKeys returns the keys of a map
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// formatItem formats a single item from tool results
func (m *MCPClientWrapper) formatItem(item map[string]interface{}) string {
	var parts []string

	// For issues/PRs: format as "#number: title (state)"
	if number, ok := item["number"].(float64); ok {
		parts = append(parts, fmt.Sprintf("#%.0f", number))
	}

	if title, ok := item["title"].(string); ok && title != "" {
		parts = append(parts, title)
	}

	if state, ok := item["state"].(string); ok && state != "" {
		parts = append(parts, fmt.Sprintf("(%s)", state))
	}

	// For repositories: format as "name (full_name)"
	if name, ok := item["name"].(string); ok && name != "" {
		if len(parts) == 0 {
			parts = append(parts, name)
		}
	}

	if fullName, ok := item["full_name"].(string); ok && fullName != "" {
		if !strings.Contains(strings.Join(parts, " "), fullName) {
			parts = append(parts, fmt.Sprintf("(%s)", fullName))
		}
	}

	// Add URL if available
	if htmlURL, ok := item["html_url"].(string); ok && htmlURL != "" {
		parts = append(parts, fmt.Sprintf("[%s](%s)", "View", htmlURL))
	}

	if len(parts) > 0 {
		return fmt.Sprintf("- %s\n", strings.Join(parts, " "))
	}

	// Fallback: show all fields
	var fallback []string
	for key, value := range item {
		fallback = append(fallback, fmt.Sprintf("%s: %v", key, value))
	}
	if len(fallback) > 0 {
		return fmt.Sprintf("- %s\n", strings.Join(fallback, ", "))
	}

	return ""
}

// listRepositoriesFromMCP lists repositories using MCP
func (m *MCPClientWrapper) listRepositoriesFromMCP(ctx context.Context) (string, error) {
	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      m.getNextRequestID(),
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "list_repositories",
			"arguments": map[string]interface{}{
				"limit": 10,
			},
		},
	}

	resp, err := m.sendRequest(ctx, req)
	if err != nil {
		return m.getGitHubContext(ctx, "")
	}

	if resp.Error != nil {
		return m.getGitHubContext(ctx, "")
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return m.getGitHubContext(ctx, "")
	}

	context := "User's GitHub repositories (via MCP):\n"
	if repos, ok := result["repositories"].([]interface{}); ok {
		for _, repo := range repos {
			if repoMap, ok := repo.(map[string]interface{}); ok {
				name := ""
				fullName := ""
				desc := ""
				if n, ok := repoMap["name"].(string); ok {
					name = n
				}
				if fn, ok := repoMap["full_name"].(string); ok {
					fullName = fn
				}
				if d, ok := repoMap["description"].(string); ok {
					desc = d
				}
				if fullName != "" {
					context += fmt.Sprintf("- %s: %s\n", fullName, desc)
				} else if name != "" {
					context += fmt.Sprintf("- %s: %s\n", name, desc)
				}
			}
		}
	}

	return context, nil
}

// getIssuesFromMCP gets issues for a repository using MCP
func (m *MCPClientWrapper) getIssuesFromMCP(ctx context.Context, query string) (string, error) {
	// Try to extract repository name from query
	repoName := extractRepoFromQuery(query)
	log.Printf("Extracted repository from query: '%s' (query: '%s')", repoName, query)

	// Split owner/repo if needed
	var owner, repo string
	if repoName != "" {
		parts := strings.Split(repoName, "/")
		if len(parts) == 2 {
			owner = parts[0]
			repo = parts[1]
			log.Printf("Split repository into owner='%s', repo='%s'", owner, repo)
		} else {
			// If not in owner/repo format, assume it's just the repo name
			repo = repoName
			log.Printf("Repository not in owner/repo format, using as repo name: '%s'", repo)
		}
	}

	// Try different MCP tool names for getting issues
	toolNames := []string{"get_issues", "list_issues", "github_get_issues", "github_list_issues"}

	for _, toolName := range toolNames {
		args := map[string]interface{}{}
		if repoName != "" {
			if owner != "" && repo != "" {
				// Try both formats - some tools want owner/repo, others want separate fields
				args["repository"] = repoName // Full owner/repo format
				args["owner"] = owner
				args["repo"] = repo
			} else {
				args["repository"] = repo
				args["owner"] = "" // Will use authenticated user
			}
		}
		args["state"] = "open" // Default to open issues
		args["limit"] = 10

		req := MCPRequest{
			JSONRPC: "2.0",
			ID:      m.getNextRequestID(),
			Method:  "tools/call",
			Params: map[string]interface{}{
				"name":      toolName,
				"arguments": args,
			},
		}

		resp, err := m.sendRequest(ctx, req)
		if err != nil {
			continue
		}

		if resp.Error != nil {
			continue
		}

		var result map[string]interface{}
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			continue
		}

		// Build context from issues
		context := "GitHub Issues (via MCP):\n"
		if issues, ok := result["issues"].([]interface{}); ok {
			for _, issue := range issues {
				if issueMap, ok := issue.(map[string]interface{}); ok {
					title := ""
					number := ""
					state := ""
					if t, ok := issueMap["title"].(string); ok {
						title = t
					}
					if n, ok := issueMap["number"].(float64); ok {
						number = fmt.Sprintf("#%.0f", n)
					}
					if s, ok := issueMap["state"].(string); ok {
						state = s
					}
					context += fmt.Sprintf("- %s %s (%s): %s\n", number, title, state, title)
				}
			}
		} else if issues, ok := result["items"].([]interface{}); ok {
			// Some APIs return items instead of issues
			for _, issue := range issues {
				if issueMap, ok := issue.(map[string]interface{}); ok {
					title := ""
					number := ""
					state := ""
					if t, ok := issueMap["title"].(string); ok {
						title = t
					}
					if n, ok := issueMap["number"].(float64); ok {
						number = fmt.Sprintf("#%.0f", n)
					}
					if s, ok := issueMap["state"].(string); ok {
						state = s
					}
					context += fmt.Sprintf("- %s %s (%s): %s\n", number, title, state, title)
				}
			}
		}

		if context != "GitHub Issues (via MCP):\n" {
			return context, nil
		}
	}

	// If MCP tools don't work, fall back to direct API
	log.Printf("MCP tools failed, falling back to direct GitHub API for repo: %s", repoName)
	return m.getIssuesFromAPI(ctx, repoName)
}

// getPullRequestsFromMCP gets pull requests using MCP
func (m *MCPClientWrapper) getPullRequestsFromMCP(ctx context.Context, query string) (string, error) {
	repoName := extractRepoFromQuery(query)

	// Try different MCP tool names
	toolNames := []string{"get_pull_requests", "list_pull_requests", "github_get_pull_requests"}

	for _, toolName := range toolNames {
		args := map[string]interface{}{}
		if repoName != "" {
			args["repository"] = repoName
		}
		args["state"] = "open"
		args["limit"] = 10

		req := MCPRequest{
			JSONRPC: "2.0",
			ID:      m.getNextRequestID(),
			Method:  "tools/call",
			Params: map[string]interface{}{
				"name":      toolName,
				"arguments": args,
			},
		}

		resp, err := m.sendRequest(ctx, req)
		if err != nil {
			continue
		}

		if resp.Error != nil {
			continue
		}

		var result map[string]interface{}
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			continue
		}

		context := "GitHub Pull Requests (via MCP):\n"
		if prs, ok := result["pull_requests"].([]interface{}); ok {
			for _, pr := range prs {
				if prMap, ok := pr.(map[string]interface{}); ok {
					title := ""
					number := ""
					if t, ok := prMap["title"].(string); ok {
						title = t
					}
					if n, ok := prMap["number"].(float64); ok {
						number = fmt.Sprintf("#%.0f", n)
					}
					context += fmt.Sprintf("- %s: %s\n", number, title)
				}
			}
		}

		if context != "GitHub Pull Requests (via MCP):\n" {
			return context, nil
		}
	}

	// Fall back to direct API
	return m.getPullRequestsFromAPI(ctx, repoName)
}

// extractRepoFromQuery tries to extract repository name from user query
func extractRepoFromQuery(query string) string {
	log.Printf("Extracting repo from query: %s", query)

	// Clean up query - remove leading/trailing slashes and whitespace
	query = strings.TrimSpace(query)
	query = strings.Trim(query, "/")

	// First, try to extract from GitHub URL (with or without https://)
	// Pattern: https://github.com/owner/repo or github.com/owner/repo
	// Make the pattern case-insensitive and match http/https or no protocol
	githubURLPattern := `(?i)(?:https?://)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)`
	re := regexp.MustCompile(githubURLPattern)
	matches := re.FindStringSubmatch(query)
	if len(matches) >= 3 {
		// Return owner/repo format
		repo := fmt.Sprintf("%s/%s", matches[1], matches[2])
		log.Printf("Extracted repo from GitHub URL: %s", repo)
		return repo
	}
	log.Printf("No match found with GitHub URL pattern (tried: %s)", githubURLPattern)

	// Also try to extract from URL without https://
	if strings.Contains(query, "github.com/") {
		parts := strings.Split(query, "github.com/")
		if len(parts) > 1 {
			repoPart := strings.Fields(parts[1])[0] // Get first part before space
			// Remove trailing slashes or fragments
			repoPart = strings.TrimSuffix(repoPart, "/")
			repoPart = strings.Split(repoPart, "#")[0]
			repoParts := strings.Split(repoPart, "/")
			if len(repoParts) >= 2 {
				repo := fmt.Sprintf("%s/%s", repoParts[0], repoParts[1])
				log.Printf("Extracted repo from github.com URL: %s", repo)
				return repo
			}
		}
	}
	log.Printf("No match found with github.com URL pattern")

	// Simple extraction - look for patterns like "repo name", "for repo", etc.
	queryLower := strings.ToLower(query)

	// Look for "for <repo>" or "in <repo>" - but only if we haven't found a URL yet
	// Also handle "for repo <owner/repo>" pattern
	patterns := []string{"for repo ", "for repository ", "in repo ", "in repository ", "for ", "in ", "repo ", "repository "}
	for _, pattern := range patterns {
		idx := strings.Index(queryLower, pattern)
		if idx != -1 {
			log.Printf("Found pattern '%s' at index %d in query: %s", pattern, idx, query)
			after := query[idx+len(pattern):]
			log.Printf("Text after pattern '%s': '%s'", pattern, after)
			// Skip URL if it's part of the pattern - try to extract from URL first
			if strings.Contains(after, "github.com") {
				// Try to extract from the URL in this section using the same pattern
				urlPattern := `(?i)(?:https?://)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)`
				urlRe := regexp.MustCompile(urlPattern)
				urlMatches := urlRe.FindStringSubmatch(after)
				if len(urlMatches) >= 3 {
					repo := fmt.Sprintf("%s/%s", urlMatches[1], urlMatches[2])
					log.Printf("Extracted repo from URL in pattern section: %s", repo)
					return repo
				}
				// If regex didn't match, try string split
				if strings.Contains(after, "github.com/") {
					urlParts := strings.Split(after, "github.com/")
					if len(urlParts) > 1 {
						repoPart := strings.Fields(urlParts[1])[0]
						repoPart = strings.TrimSuffix(repoPart, "/")
						repoPart = strings.Split(repoPart, "#")[0]
						repoParts := strings.Split(repoPart, "/")
						if len(repoParts) >= 2 {
							repo := fmt.Sprintf("%s/%s", repoParts[0], repoParts[1])
							log.Printf("Extracted repo from URL in pattern section (split): %s", repo)
							return repo
						}
					}
				}
				continue
			}
			// Try to extract owner/repo format first (e.g., "lozcrowther-eng/JavaCoffeeShop/")
			ownerRepoPattern := `([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)`
			ownerRepoRe := regexp.MustCompile(ownerRepoPattern)
			ownerRepoMatches := ownerRepoRe.FindStringSubmatch(after)
			log.Printf("Owner/repo regex matches: %v (len=%d)", ownerRepoMatches, len(ownerRepoMatches))
			if len(ownerRepoMatches) >= 3 {
				repo := fmt.Sprintf("%s/%s", ownerRepoMatches[1], ownerRepoMatches[2])
				log.Printf("Extracted owner/repo from pattern: %s", repo)
				return repo
			}

			parts := strings.Fields(after)
			if len(parts) > 0 {
				repo := parts[0]
				// Remove trailing slash if present
				repo = strings.TrimSuffix(repo, "/")
				// Skip if it's a URL
				if strings.HasPrefix(repo, "http") || strings.Contains(repo, "github.com") {
					log.Printf("Skipping URL in pattern extraction: %s (should have been extracted earlier)", repo)
					// Try one more time to extract from this URL
					urlPattern := `(?i)(?:https?://)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)`
					urlRe := regexp.MustCompile(urlPattern)
					urlMatches := urlRe.FindStringSubmatch(repo)
					if len(urlMatches) >= 3 {
						extractedRepo := fmt.Sprintf("%s/%s", urlMatches[1], urlMatches[2])
						log.Printf("Successfully extracted repo from URL in pattern section: %s", extractedRepo)
						return extractedRepo
					}
					continue
				}
				// Remove common words - also skip "repo" and "repository" as they're not actual repo names
				if repo != "the" && repo != "a" && repo != "my" && repo != "repo" && repo != "repository" {
					// Check if it's already in owner/repo format
					if strings.Contains(repo, "/") {
						log.Printf("Extracted repo from pattern: %s", repo)
						return repo
					}
					// Otherwise, might need owner - but we'll use it as-is
					log.Printf("Extracted repo from pattern (single word): %s", repo)
					return repo
				} else {
					log.Printf("Skipping common word or 'repo'/'repository': %s", repo)
					// If we skipped "repo", try the next part
					if len(parts) > 1 {
						nextRepo := parts[1]
						nextRepo = strings.TrimSuffix(nextRepo, "/")
						if strings.Contains(nextRepo, "/") {
							log.Printf("Extracted repo from next part after 'repo': %s", nextRepo)
							return nextRepo
						}
					}
				}
			}
		}
	}

	log.Printf("No repository extracted from query: %s", query)
	return ""
}

// getIssuesFromAPI gets issues using direct GitHub API (fallback)
func (m *MCPClientWrapper) getIssuesFromAPI(ctx context.Context, repoName string) (string, error) {
	var url string
	if repoName != "" {
		// Get issues for specific repo
		url = fmt.Sprintf("https://api.github.com/repos/%s/issues?state=open&per_page=10", repoName)
	} else {
		// Get issues for all user repos
		url = "https://api.github.com/issues?state=open&per_page=10"
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", fmt.Sprintf("token %s", m.Token))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GitHub API error: %s", string(body))
	}

	var issues []struct {
		Number int    `json:"number"`
		Title  string `json:"title"`
		State  string `json:"state"`
		Repo   struct {
			FullName string `json:"full_name"`
		} `json:"repository"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&issues); err != nil {
		return "", err
	}

	context := "GitHub Issues:\n"
	for _, issue := range issues {
		repoName := issue.Repo.FullName
		if repoName == "" {
			repoName = "unknown"
		}
		context += fmt.Sprintf("- %s #%d (%s): %s\n", repoName, issue.Number, issue.State, issue.Title)
	}

	return context, nil
}

// getPullRequestsFromAPI gets pull requests using direct GitHub API (fallback)
func (m *MCPClientWrapper) getPullRequestsFromAPI(ctx context.Context, repoName string) (string, error) {
	var url string
	if repoName != "" {
		url = fmt.Sprintf("https://api.github.com/repos/%s/pulls?state=open&per_page=10", repoName)
	} else {
		url = "https://api.github.com/pulls?state=open&per_page=10"
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", fmt.Sprintf("token %s", m.Token))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GitHub API error: %s", string(body))
	}

	var prs []struct {
		Number int    `json:"number"`
		Title  string `json:"title"`
		State  string `json:"state"`
		Repo   struct {
			FullName string `json:"full_name"`
		} `json:"repository"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&prs); err != nil {
		return "", err
	}

	context := "GitHub Pull Requests:\n"
	for _, pr := range prs {
		repoName := pr.Repo.FullName
		context += fmt.Sprintf("- %s #%d (%s): %s\n", repoName, pr.Number, pr.State, pr.Title)
	}

	return context, nil
}

// getGitHubContext makes GitHub API calls to get relevant context
func (m *MCPClientWrapper) getGitHubContext(ctx context.Context, query string) (string, error) {
	// Simple implementation: get user's repositories
	req, err := http.NewRequestWithContext(ctx, "GET", "https://api.github.com/user/repos?per_page=5&sort=updated", nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", fmt.Sprintf("token %s", m.Token))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := m.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GitHub API error: %s", string(body))
	}

	var repos []struct {
		Name        string `json:"name"`
		FullName    string `json:"full_name"`
		Description string `json:"description"`
		UpdatedAt   string `json:"updated_at"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&repos); err != nil {
		return "", err
	}

	// Build context string
	context := "User's recent GitHub repositories:\n"
	for _, repo := range repos {
		context += fmt.Sprintf("- %s: %s (updated: %s)\n", repo.FullName, repo.Description, repo.UpdatedAt)
	}

	return context, nil
}

// Close cleans up the MCP client
func (m *MCPClientWrapper) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Mark as not initialized
	m.initialized = false
	return nil
}
