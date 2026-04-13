// API Helper
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

// Health indicator check
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

        return health;
    } catch (e) {
        console.error('Health check failed:', e);
        return null;
    }
}

// Tab switching for TTS page
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).style.display = 'block';
    document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
}

// TTS Generation
async function generateTTS() {
    const text = document.getElementById('tts-text').value;
    const voice = document.getElementById('tts-voice').value;
    const provider = document.getElementById('tts-provider').value;
    const speed = parseFloat(document.getElementById('tts-speed').value) || 1.0;

    if (!text) {
        alert('Please enter text to synthesize');
        return;
    }

    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice, provider, speed })
        });

        if (response.ok) {
            const blob = await response.blob();
            const audio = document.getElementById('tts-audio');
            audio.src = URL.createObjectURL(blob);
            audio.style.display = 'block';
            audio.play();
        } else {
            const err = await response.json();
            alert('Error: ' + (err.error || 'Generation failed'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

// Form validation helper
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const required = form.querySelectorAll('[required]');
    for (const input of required) {
        if (!input.value.trim()) {
            input.focus();
            return false;
        }
    }
    return true;
}

// Show alert message
function showAlert(message, type = 'error') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const container = document.querySelector('.content');
    container.insertBefore(alertDiv, container.firstChild);

    setTimeout(() => alertDiv.remove(), 5000);
}

// Format date/time
function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check health on every page
    checkHealth();

    // Auto-refresh health indicator every 30 seconds
    setInterval(checkHealth, 30000);
});
