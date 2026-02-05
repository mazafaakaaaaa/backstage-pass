package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

var (
	githubClientID     string
	githubClientSecret string
	oauthStates        = make(map[string]time.Time)
	oauthStatesMu      sync.RWMutex
)

func initGitHubOAuth(clientID, clientSecret string) {
	githubClientID = clientID
	githubClientSecret = clientSecret

	// Clean up expired OAuth states periodically
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			oauthStatesMu.Lock()
			now := time.Now()
			for state, expiry := range oauthStates {
				if now.After(expiry) {
					delete(oauthStates, state)
				}
			}
			oauthStatesMu.Unlock()
		}
	}()
}

func handleGitHubAuth(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session_id")
	if sessionID == "" {
		http.Error(w, "Session ID required", http.StatusBadRequest)
		return
	}

	// Generate OAuth state
	state := uuid.New().String()
	oauthStatesMu.Lock()
	oauthStates[state] = time.Now().Add(10 * time.Minute) // 10 minute expiry
	oauthStatesMu.Unlock()

	// Store session ID in state (we'll encode it)
	stateWithSession := fmt.Sprintf("%s:%s", state, sessionID)

	// Get callback URL
	callbackURL := getCallbackURL(r)
	
	// Log the callback URL for debugging
	log.Printf("OAuth callback URL: %s", callbackURL)
	log.Printf("OAuth request from host: %s", r.Host)

	// GitHub OAuth URL
	authURL := fmt.Sprintf(
		"https://github.com/login/oauth/authorize?client_id=%s&scope=repo,read:user&state=%s&redirect_uri=%s",
		githubClientID,
		url.QueryEscape(stateWithSession),
		url.QueryEscape(callbackURL),
	)

	http.Redirect(w, r, authURL, http.StatusTemporaryRedirect)
}

func handleGitHubCallback(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")
	stateWithSession := r.URL.Query().Get("state")
	errorParam := r.URL.Query().Get("error")

	if errorParam != "" {
		http.Error(w, fmt.Sprintf("GitHub OAuth error: %s", errorParam), http.StatusBadRequest)
		return
	}

	if code == "" || stateWithSession == "" {
		http.Error(w, "Missing code or state", http.StatusBadRequest)
		return
	}

	// Parse state to get session ID
	parts := strings.Split(stateWithSession, ":")
	if len(parts) != 2 {
		http.Error(w, "Invalid state", http.StatusBadRequest)
		return
	}
	state, sessionID := parts[0], parts[1]

	// Verify state
	oauthStatesMu.Lock()
	_, exists := oauthStates[state]
	if exists {
		delete(oauthStates, state)
	}
	oauthStatesMu.Unlock()

	if !exists {
		http.Error(w, "Invalid or expired state", http.StatusBadRequest)
		return
	}

	// Exchange code for token
	token, err := exchangeGitHubCode(code)
	if err != nil {
		log.Printf("Failed to exchange code: %v", err)
		http.Error(w, "Failed to exchange code for token", http.StatusInternalServerError)
		return
	}

	// Get session and store token
	sessionsMu.Lock()
	session, exists := sessions[sessionID]
	if !exists {
		session = &UserSession{
			ID:        sessionID,
			CreatedAt: time.Now(),
		}
		sessions[sessionID] = session
	}
	sessionsMu.Unlock()

	// Update session fields (thread-safe)
	session.mu.Lock()
	session.GitHubToken = token
	session.LastActivity = time.Now()

	// Initialize MCP client for this user
	mcpClient, err := NewMCPClientWrapper(context.Background(), token)
	if err != nil {
		log.Printf("Failed to initialize MCP client: %v", err)
		// Don't fail the auth, but log the error
	} else {
		session.MCPClient = mcpClient
	}
	session.mu.Unlock()

	// Redirect back to chat interface
	// Frontend will read session_id from URL, store in localStorage, and remove from URL immediately
	http.Redirect(w, r, fmt.Sprintf("/?session_id=%s", sessionID), http.StatusTemporaryRedirect)
}

func exchangeGitHubCode(code string) (string, error) {
	data := url.Values{}
	data.Set("client_id", githubClientID)
	data.Set("client_secret", githubClientSecret)
	data.Set("code", code)

	req, err := http.NewRequest("POST", "https://github.com/login/oauth/access_token", strings.NewReader(data.Encode()))
	if err != nil {
		return "", err
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var tokenResp struct {
		AccessToken string `json:"access_token"`
		TokenType   string `json:"token_type"`
		Scope       string `json:"scope"`
		Error       string `json:"error"`
		ErrorDesc   string `json:"error_description"`
	}

	if err := json.Unmarshal(body, &tokenResp); err != nil {
		return "", err
	}

	if tokenResp.Error != "" {
		return "", fmt.Errorf("%s: %s", tokenResp.Error, tokenResp.ErrorDesc)
	}

	return tokenResp.AccessToken, nil
}

func getCallbackURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	host := r.Host
	return fmt.Sprintf("%s://%s/auth/github/callback", scheme, host)
}

