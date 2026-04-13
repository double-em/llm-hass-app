/**
 * LLM AI Dashboard - Frontend JavaScript
 * API helpers, animations, and interactivity
 */

// ============================================================================
// API Helper
// ============================================================================

async function api(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }

        return { success: response.ok, status: response.status };
    } catch (error) {
        console.error('API Error:', error);
        return { error: error.message, success: false };
    }
}

// ============================================================================
// Health Check
// ============================================================================

let healthInterval;

async function checkHealth() {
    try {
        const health = await api('/api/health');
        const indicator = document.getElementById('health-indicator');
        const text = document.getElementById('health-text');

        if (indicator && text) {
            if (health.omnivoice_loaded) {
                indicator.className = 'indicator online';
                text.textContent = 'System Ready';
            } else {
                indicator.className = 'indicator offline';
                text.textContent = 'Model Loading';
            }
        }

        // Clear interval once system is ready
        if (health.omnivoice_loaded && healthInterval) {
            clearInterval(healthInterval);
            healthInterval = null;
        }

        return health;
    } catch (e) {
        console.error('Health check failed:', e);
        return null;
    }
}

// Start health checks
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    healthInterval = setInterval(checkHealth, 10000);
});

// ============================================================================
// Tab System
// ============================================================================

function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(el => {
        el.style.display = 'none';
    });

    // Deactivate all tab buttons
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabId).style.display = 'block';

    // Activate button
    document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
}

// ============================================================================
// TTS Generation
// ============================================================================

async function generateTTS() {
    const text = document.getElementById('tts-text')?.value;
    const voice = document.getElementById('tts-voice')?.value;
    const provider = document.getElementById('tts-provider')?.value;
    const speed = parseFloat(document.getElementById('tts-speed')?.value) || 1.0;

    if (!text) {
        showAlert('Please enter text to synthesize', 'error');
        return;
    }

    const btn = event?.target;
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Generating...';
    }

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice, provider, speed })
        });

        if (response.ok) {
            const blob = await response.blob();
            const audio = document.getElementById('tts-audio');
            if (audio) {
                audio.src = URL.createObjectURL(blob);
                audio.style.display = 'block';
                audio.play();
                showAlert('Audio generated successfully', 'success');
            }
        } else {
            const err = await response.json();
            showAlert('Error: ' + (err.error || 'Generation failed'), 'error');
        }
    } catch (e) {
        showAlert('Error: ' + e.message, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Generate';
        }
    }
}

async function generateQuickTTS() {
    const text = document.getElementById('quick-text')?.value;
    const voice = document.getElementById('quick-voice')?.value;
    const provider = document.getElementById('quick-provider')?.value || 'omnivoice';

    if (!text) {
        showAlert('Please enter text', 'error');
        return;
    }

    const btn = event?.target;
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Generating...';
    }

    try {
        const body = { text, provider };
        if (voice) body.voice = voice;

        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (response.ok) {
            const blob = await response.blob();
            const audio = document.getElementById('quick-audio');
            if (audio) {
                audio.src = URL.createObjectURL(blob);
                audio.style.display = 'block';
                audio.play();
            }
        } else {
            const err = await response.json();
            showAlert('Error: ' + (err.error || 'Generation failed'), 'error');
        }
    } catch (e) {
        showAlert('Error: ' + e.message, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Generate';
        }
    }
}

// ============================================================================
// Conversation
// ============================================================================

let conversationHistory = [];

async function sendMessage() {
    const message = document.getElementById('conv-message')?.value;
    const provider = document.getElementById('conv-provider')?.value;

    if (!message) return;

    // Add user message
    addConvMessage('user', message);
    document.getElementById('conv-message').value = '';

    try {
        const response = await fetch('/api/conversation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, provider })
        });

        if (response.ok) {
            const data = await response.json();
            addConvMessage('assistant', data.response);
        } else {
            const err = await response.json();
            addConvMessage('assistant', 'Error: ' + (err.error || 'Request failed'));
        }
    } catch (e) {
        addConvMessage('assistant', 'Error: ' + e.message);
    }
}

function addConvMessage(role, content) {
    const container = document.getElementById('conv-messages');
    if (!container) return;

    const div = document.createElement('div');
    div.className = `conv-message ${role}`;
    div.textContent = content;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ============================================================================
// Alert System
// ============================================================================

function showAlert(message, type = 'error') {
    // Remove existing alerts
    document.querySelectorAll('.alert').forEach(el => el.remove());

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const container = document.querySelector('.content');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.style.opacity = '0';
            setTimeout(() => alertDiv.remove(), 300);
        }, 5000);
    }
}

// ============================================================================
// Dashboard Loading
// ============================================================================

async function loadDashboard() {
    const health = await checkHealth();

    const providers = await api('/api/providers');
    const providersCount = document.getElementById('providers-count');
    if (providersCount) {
        providersCount.textContent = Object.keys(providers.providers || {}).length + ' configured';
    }

    const voices = await api('/api/voices');
    const voicesCount = document.getElementById('voices-count');
    if (voicesCount) {
        voicesCount.textContent = (voices.voices || []).length + ' voices';
    }

    // Populate quick TTS voice dropdown
    const voiceSelect = document.getElementById('quick-voice');
    if (voiceSelect && voices.voices) {
        voices.voices.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.name;
            opt.textContent = v.name;
            voiceSelect.appendChild(opt);
        });
    }

    // Populate conversation provider dropdown
    const convProvider = document.getElementById('conv-provider');
    if (convProvider && providers.providers) {
        Object.keys(providers.providers).forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            convProvider.appendChild(opt);
        });
    }
}

// ============================================================================
// Form Helpers
// ============================================================================

function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const required = form.querySelectorAll('[required]');
    for (const input of required) {
        if (!input.value.trim()) {
            input.focus();
            showAlert('Please fill in all required fields', 'error');
            return false;
        }
    }
    return true;
}

// ============================================================================
// Date Formatting
// ============================================================================

function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// ============================================================================
// Keyboard shortcuts
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Enter to send message in conversation
    if (e.key === 'Enter' && e.target.id === 'conv-message' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
