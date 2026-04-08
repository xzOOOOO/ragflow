const API_BASE = 'http://localhost:8000';

class ChatApp {
    constructor() {
        this.chatHistory = [];
        this.isLoading = false;

        this.initElements();
        this.bindEvents();
        this.loadHistory();
    }

    initElements() {
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messages = document.getElementById('messages');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.chatContainer = document.getElementById('chatContainer');
        this.uploadModal = document.getElementById('uploadModal');
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.chatHistoryEl = document.getElementById('chatHistory');
    }

    bindEvents() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.uploadBtn.addEventListener('click', () => {
            this.uploadModal.classList.add('active');
        });
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.userInput.addEventListener('input', () => {
            this.autoResize(this.userInput);
        });

        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '#4f46e5';
            this.uploadArea.style.background = '#f5f3ff';
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.style.borderColor = '';
            this.uploadArea.style.background = '';
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '';
            this.uploadArea.style.background = '';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.uploadFile(files[0]);
            }
        });

        document.getElementById('modalClose').addEventListener('click', () => {
            this.uploadModal.classList.remove('active');
        });

        this.uploadModal.addEventListener('click', (e) => {
            if (e.target === this.uploadModal) {
                this.uploadModal.classList.remove('active');
            }
        });

        this.newChatBtn.addEventListener('click', () => this.newChat());

        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.userInput.value = btn.dataset.question;
                this.sendMessage();
            });
        });
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message || this.isLoading) return;

        this.hideWelcome();
        this.addMessage(message, 'user');
        this.chatHistory.push({ role: 'user', content: message });

        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        this.showTypingIndicator();
        this.isLoading = true;

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: message })
            });

            if (!response.ok) throw new Error('请求失败');

            const data = await response.json();
            this.removeTypingIndicator();
            this.addMessage(data.answer, 'ai', data.sources, data.retrieved_count);
            this.chatHistory.push({ role: 'ai', content: data.answer });

            this.isLoading = false;
        } catch (error) {
            this.removeTypingIndicator();
            this.addMessage('抱歉，发生了错误。请确保后端服务正在运行。', 'ai');
            this.isLoading = false;
        }
    }

    addMessage(content, role, sources = null, retrievedCount = 0) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatarSvg = role === 'user'
            ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>'
            : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarSvg}</div>
            <div class="message-content">
                <div class="message-bubble">${this.formatContent(content)}</div>
            </div>
        `;

        if (sources && sources.length > 0) {
            const sourcesSection = document.createElement('div');
            sourcesSection.className = 'sources-section';
            sourcesSection.innerHTML = `
                <div class="sources-title">参考文档 (${sources.length})</div>
                ${sources.map(s => `<div class="source-item">${this.escapeHtml(s.content)}</div>`).join('')}
            `;
            messageDiv.querySelector('.message-content').appendChild(sourcesSection);
        }

        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatContent(content) {
        let html = this.escapeHtml(content);

        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[1-3]>)/g, '$1');
        html = html.replace(/(<\/h[1-3]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ol>)/g, '$1');
        html = html.replace(/(<\/ol>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message ai';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
                    <line x1="9" y1="9" x2="9.01" y2="9"></line>
                    <line x1="15" y1="9" x2="15.01" y2="9"></line>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        `;
        this.messages.appendChild(indicator);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    hideWelcome() {
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
    }

    showWelcome() {
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'flex';
        }
    }

    newChat() {
        this.messages.innerHTML = '';
        this.chatHistory = [];
        this.showWelcome();
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        const progressEl = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadArea = document.getElementById('uploadArea');

        uploadArea.style.display = 'none';
        progressEl.style.display = 'block';
        progressText.textContent = '上传中...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                progressText.textContent = '上传成功！';
                setTimeout(() => {
                    this.uploadModal.classList.remove('active');
                    uploadArea.style.display = 'block';
                    progressEl.style.display = 'none';
                    this.fileInput.value = '';
                }, 1500);
            } else {
                const error = await response.json().catch(() => ({ detail: '上传失败' }));
                throw new Error(error.detail || '上传失败');
            }
        } catch (error) {
            progressText.textContent = '上传失败: ' + error.message;
            setTimeout(() => {
                this.uploadModal.classList.remove('active');
                uploadArea.style.display = 'block';
                progressEl.style.display = 'none';
                this.fileInput.value = '';
            }, 3000);
        }
    }

    loadHistory() {
        const saved = localStorage.getItem('chatHistory');
        if (saved) {
            try {
                const history = JSON.parse(saved);
                this.renderHistory(history);
            } catch (e) {
                console.error('Failed to load history:', e);
            }
        }
    }

    saveHistory() {
        try {
            localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory));
        } catch (e) {
            console.error('Failed to save history:', e);
        }
    }

    renderHistory(history) {
        this.chatHistoryEl.innerHTML = '';

        const currentChat = document.createElement('div');
        currentChat.className = 'history-item active';
        currentChat.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <span>当前对话</span>
        `;
        this.chatHistoryEl.appendChild(currentChat);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
