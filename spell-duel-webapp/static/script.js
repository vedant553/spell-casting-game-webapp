// WebSocket connection
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// DOM Elements
const videoFeed = document.getElementById('video-feed');
const playerHpElement = document.getElementById('player-hp');
const playerMaxHpElement = document.getElementById('player-max-hp');
const playerHpBar = document.getElementById('player-hp-bar');
const playerShieldIndicator = document.getElementById('player-shield-indicator');
const opponentHpElement = document.getElementById('opponent-hp');
const opponentMaxHpElement = document.getElementById('opponent-max-hp');
const opponentHpBar = document.getElementById('opponent-hp-bar');
const lastSpellElement = document.getElementById('last-spell');
const aiActionElement = document.getElementById('ai-action');
const gameOverMessage = document.getElementById('game-over-message');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    updateConnectionStatus('connecting', 'Connecting...');
    
    ws = new WebSocket(wsUrl);
    
    // Connection opened
    ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus('connected', 'Connected');
    };
    
    // Listen for messages
    ws.onmessage = async (event) => {
        // Check if message is binary (video frame) or text (game state)
        if (event.data instanceof Blob) {
            // Video frame - display it
            const imageUrl = URL.createObjectURL(event.data);
            videoFeed.src = imageUrl;
            
            // Revoke old object URL to prevent memory leaks
            if (videoFeed.dataset.oldUrl) {
                URL.revokeObjectURL(videoFeed.dataset.oldUrl);
            }
            videoFeed.dataset.oldUrl = imageUrl;
        } else {
            // Game state JSON - update UI
            try {
                const gameState = JSON.parse(event.data);
                updateGameState(gameState);
            } catch (e) {
                console.error('Error parsing game state:', e);
            }
        }
    };
    
    // Connection closed
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus('disconnected', 'Disconnected');
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Reconnecting... Attempt ${reconnectAttempts}`);
            setTimeout(connectWebSocket, 2000);
        } else {
            updateConnectionStatus('error', 'Connection failed');
        }
    };
    
    // Connection error
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('error', 'Connection error');
    };
}

// Update game state in the UI
function updateGameState(state) {
    // Update Player HP
    playerHpElement.textContent = state.player_hp;
    playerMaxHpElement.textContent = state.player_max_hp;
    const playerHpPercent = (state.player_hp / state.player_max_hp) * 100;
    playerHpBar.style.width = `${playerHpPercent}%`;
    
    // Update shield indicator
    if (state.player_shield) {
        playerShieldIndicator.classList.remove('hidden');
    } else {
        playerShieldIndicator.classList.add('hidden');
    }
    
    // Update Opponent HP
    opponentHpElement.textContent = state.opponent_hp;
    opponentMaxHpElement.textContent = state.opponent_max_hp;
    const opponentHpPercent = (state.opponent_hp / state.opponent_max_hp) * 100;
    opponentHpBar.style.width = `${opponentHpPercent}%`;
    
    // Update spell feedback
    if (state.last_spell) {
        lastSpellElement.textContent = state.last_spell;
        lastSpellElement.style.display = 'block';
    } else {
        lastSpellElement.textContent = '';
        lastSpellElement.style.display = 'none';
    }
    
    // Update AI action
    if (state.ai_action) {
        aiActionElement.textContent = state.ai_action;
        aiActionElement.style.display = 'block';
    } else {
        aiActionElement.textContent = '';
        aiActionElement.style.display = 'none';
    }
    
    // Update game over status
    if (state.game_over) {
        if (state.winner === 'Player') {
            gameOverMessage.textContent = '🎉 YOU WIN! 🎉';
            gameOverMessage.className = 'game-over-msg win';
            gameOverMessage.classList.remove('hidden');
        } else {
            gameOverMessage.textContent = '💀 GAME OVER 💀';
            gameOverMessage.className = 'game-over-msg lose';
            gameOverMessage.classList.remove('hidden');
        }
    } else {
        gameOverMessage.classList.add('hidden');
    }
}

// Update connection status indicator
function updateConnectionStatus(status, text) {
    statusText.textContent = text;
    
    if (status === 'connected') {
        statusIndicator.classList.add('connected');
    } else {
        statusIndicator.classList.remove('connected');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Spell Duel Web App...');
    connectWebSocket();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (ws) {
        ws.close();
    }
});
