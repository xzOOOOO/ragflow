const API_BASE = 'http://localhost:8000';

class RAGFlowApp {
    constructor() {
        this.chats = [];
        this.currentChatId = null;
        this.isLoading = false;
        this.documents = [];
        this.confirmCallback = null;

        this.init();
    }

    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadTheme();
        this.loadChats();
        this.loadDocuments();
        this.updateSendButtonState();
    }

    cacheElements() {
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            sidebarOverlay: document.getElementById('sidebarOverlay'),
            menuToggle: document.getElementById('menuToggle'),
            newChatBtn: document.getElementById('newChatBtn'),
            chatList: document.getElementById('chatList'),
            clearHistoryBtn: document.getElementById('clearHistoryBtn'),
            docManagerBtn: document.getElementById('docManagerBtn'),
            docCount: document.getElementById('docCount'),
            themeToggle: document.getElementById('themeToggle'),
            headerTitle: document.getElementById('headerTitle'),
            statusIndicator: document.getElementById('statusIndicator'),
            chatContainer: document.getElementById('chatContainer'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            messagesContainer: document.getElementById('messagesContainer'),
            messages: document.getElementById('messages'),
            userInput: document.getElementById('userInput'),
            sendBtn: document.getElementById('sendBtn'),
            attachBtn: document.getElementById('attachBtn'),
            docModalOverlay: document.getElementById('docModalOverlay'),
            documentModal: document.getElementById('documentModal'),
            closeDocModal: document.getElementById('closeDocModal'),
            uploadZone: document.getElementById('uploadZone'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressFilename: document.getElementById('progressFilename'),
            progressPercent: document.getElementById('progressPercent'),
            progressStatus: document.getElementById('progressStatus'),
            documentsList: document.getElementById('documentsList'),
            emptyDocsState: document.getElementById('emptyDocsState'),
            totalDocs: document.getElementById('totalDocs'),
            refreshDocsBtn: document.getElementById('refreshDocsBtn'),
            confirmModalOverlay: document.getElementById('confirmModalOverlay'),
            confirmModal: document.getElementById('confirmModal'),
            confirmIcon: document.getElementById('confirmIcon'),
            confirmTitle: document.getElementById('confirmTitle'),
            confirmMessage: document.getElementById('confirmMessage'),
            confirmCancel: document.getElementById('confirmCancel'),
            confirmOk: document.getElementById('confirmOk'),
            toastContainer: document.getElementById('toastContainer')
        };
    }

    bindEvents() {
        this.elements.menuToggle?.addEventListener('click', () => this.toggleSidebar());
        this.elements.sidebarOverlay?.addEventListener('click', () => this.closeSidebar());

        this.elements.newChatBtn?.addEventListener('click', () => this.createNewChat());
        this.elements.clearHistoryBtn?.addEventListener('click', () => this.clearAllHistory());
        this.elements.docManagerBtn?.addEventListener('click', () => this.openDocumentModal());
        this.elements.closeDocModal?.addEventListener('click', () => this.closeDocumentModal());
        this.elements.docModalOverlay?.addEventListener('click', (e) => {
            if (e.target === this.elements.docModalOverlay) this.closeDocumentModal();
        });

        this.elements.themeToggle?.addEventListener('click', () => this.toggleTheme());

        this.elements.userInput?.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.updateSendButtonState();
        });

        this.elements.userInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.elements.sendBtn?.addEventListener('click', () => this.sendMessage());
        this.elements.attachBtn?.addEventListener('click', () => {
            this.openDocumentModal();
            setTimeout(() => this.elements.fileInput?.click(), 100);
        });

        this.elements.uploadZone?.addEventListener('click', () => this.elements.fileInput?.click());
        this.elements.fileInput?.addEventListener('change', (e) => this.handleFileSelect(e));

        this.elements.uploadZone?.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadZone.classList.add('dragover');
        });

        this.elements.uploadZone?.addEventListener('dragleave', () => {
            this.elements.uploadZone.classList.remove('dragover');
        });

        this.elements.uploadZone?.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) this.uploadFile(files[0]);
        });

        this.elements.refreshDocsBtn?.addEventListener('click', () => this.loadDocuments());

        this.elements.confirmCancel?.addEventListener('click', () => this.closeConfirmModal());
        this.elements.confirmOk?.addEventListener('click', () => {
            if (this.confirmCallback) this.confirmCallback();
            this.closeConfirmModal();
        });

        this.elements.confirmModalOverlay?.addEventListener('click', (e) => {
            if (e.target === this.elements.confirmModalOverlay) this.closeConfirmModal();
        });

        document.querySelectorAll('.suggestion-chip').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                if (question) {
                    this.elements.userInput.value = question;
                    this.sendMessage();
                }
            });
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) this.closeSidebar();
        });
    }

    toggleSidebar() {
        this.elements.sidebar?.classList.toggle('open');
    }

    closeSidebar() {
        this.elements.sidebar?.classList.remove('open');
    }

    createNewChat() {
        const chatId = Date.now().toString();
        const chat = {
            id: chatId,
            title: '新对话',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        this.chats.unshift(chat);
        this.currentChatId = chatId;
        this.renderChatList();
        this.clearMessages();
        this.showWelcome();
        this.saveChats();
        this.closeSidebar();

        this.showToast('已创建新对话', 'success');
    }

    switchChat(chatId) {
        this.currentChatId = chatId;
        const chat = this.chats.find(c => c.id === chatId);

        if (chat) {
            this.clearMessages();
            this.renderMessages(chat.messages);
            this.elements.headerTitle.textContent = chat.title;

            if (chat.messages.length === 0) {
                this.showWelcome();
            } else {
                this.hideWelcome();
            }

            this.renderChatList();
            this.saveChats();
            this.closeSidebar();
        }
    }

    deleteChat(chatId, event) {
        event?.stopPropagation();

        this.showConfirm(
            '🗑️',
            '删除对话',
            '确定要删除这个对话吗？此操作无法撤销。',
            () => {
                this.chats = this.chats.filter(c => c.id !== chatId);

                if (this.currentChatId === chatId) {
                    if (this.chats.length > 0) {
                        this.switchChat(this.chats[0].id);
                    } else {
                        this.createNewChat();
                    }
                }

                this.renderChatList();
                this.saveChats();
                this.showToast('对话已删除', 'success');
            }
        );
    }

    clearAllHistory() {
        if (this.chats.length === 0) {
            this.showToast('暂无对话记录', 'info');
            return;
        }

        this.showConfirm(
            '⚠️',
            '清空所有历史',
            '确定要清空所有对话历史吗？此操作无法撤销。',
            () => {
                this.chats = [];
                this.createNewChat();
                this.showToast('已清空所有历史', 'success');
            }
        );
    }

    renderChatList() {
        if (!this.elements.chatList) return;

        this.elements.chatList.innerHTML = this.chats.map(chat => `
            <div class="chat-item ${chat.id === this.currentChatId ? 'active' : ''}" data-chat-id="${chat.id}">
                <div class="chat-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                </div>
                <div class="chat-info">
                    <div class="chat-title">${this.escapeHtml(chat.title)}</div>
                    <div class="chat-time">${this.formatTime(chat.updatedAt)}</div>
                </div>
                <div class="chat-actions">
                    <button class="chat-action-btn delete" onclick="window.app.deleteChat('${chat.id}', event)" title="删除">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');

        this.elements.chatList.querySelectorAll('.chat-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.closest('.chat-actions')) {
                    this.switchChat(item.dataset.chatId);
                }
            });
        });
    }

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        if (!textarea) return;

        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    updateSendButtonState() {
        const hasContent = this.elements.userInput?.value.trim().length > 0;
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = !hasContent || this.isLoading;
        }
    }

    async sendMessage() {
        const message = this.elements.userInput?.value.trim();
        if (!message || this.isLoading) return;

        this.hideWelcome();

        const currentChat = this.chats.find(c => c.id === this.currentChatId);
        if (currentChat) {
            currentChat.messages.push({ role: 'user', content: message, timestamp: new Date().toISOString() });

            if (currentChat.title === '新对话') {
                currentChat.title = message.slice(0, 30) + (message.length > 30 ? '...' : '');
                this.elements.headerTitle.textContent = currentChat.title;
            }

            currentChat.updatedAt = new Date().toISOString();
        }

        this.addMessageToUI(message, 'user');

        this.elements.userInput.value = '';
        this.autoResizeTextarea();
        this.updateSendButtonState();

        this.showTypingIndicator();
        this.isLoading = true;
        this.updateConnectionStatus('loading');

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: message,
                    top_k: 5,
                    strategy: 'step_back'
                })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();

            this.removeTypingIndicator();

            if (currentChat) {
                currentChat.messages.push({
                    role: 'ai',
                    content: data.answer,
                    sources: data.sources || [],
                    retrievedCount: data.retrieved_count || 0,
                    timestamp: new Date().toISOString()
                });
            }

            this.addMessageToUI(data.answer, 'ai', data.sources || [], data.retrieved_count || 0);
            this.renderChatList();
            this.saveChats();

            this.updateConnectionStatus('online');
        } catch (error) {
            console.error('发送消息失败:', error);
            this.removeTypingIndicator();

            const errorMessage = this.getErrorMessage(error);
            this.addMessageToUI(errorMessage, 'ai');

            if (currentChat) {
                currentChat.messages.push({
                    role: 'ai',
                    content: errorMessage,
                    timestamp: new Date().toISOString()
                });
            }

            this.updateConnectionStatus('error');
            this.showToast('请求失败，请检查网络连接', 'error');
        } finally {
            this.isLoading = false;
            this.updateSendButtonState();
        }
    }

    addMessageToUI(content, role, sources = [], retrievedCount = 0) {
        if (!this.elements.messages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatarContent = role === 'user'
            ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
            : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`;

        let sourcesHTML = '';
        if (sources && sources.length > 0) {
            sourcesHTML = `
                <div class="sources-card">
                    <div class="sources-header" onclick="this.parentElement.classList.toggle('collapsed')">
                        <span class="sources-title">
                            📎 参考来源
                            <span class="sources-count">${sources.length}</span>
                        </span>
                        <svg class="sources-toggle" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </div>
                    <div class="sources-content">
                        ${sources.map((source, index) => `
                            <div class="source-item">
                                <strong>来源 ${index + 1}:</strong> ${this.escapeHtml(source.content?.slice(0, 200) || '')}${source.content?.length > 200 ? '...' : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarContent}</div>
            <div class="message-content">
                <div class="message-bubble">${this.formatMarkdown(content)}</div>
                ${sourcesHTML}
            </div>
        `;

        this.elements.messages.appendChild(messageDiv);

        messageDiv.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        this.scrollToBottom();
    }

    showTypingIndicator() {
        if (!this.elements.messages) return;

        const indicator = document.createElement('div');
        indicator.className = 'message ai';
        indicator.id = 'typingIndicator';

        indicator.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            </div>
        `;

        this.elements.messages.appendChild(indicator);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    formatMarkdown(content) {
        let html = this.escapeHtml(content);

        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="${lang || ''}">${code.trim()}</code></pre>`;
        });

        html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

        html = html.replace(/(<li>.*<\/li>)(\s*<li>.*<\/li>)*/g, (match) => {
            return `<ol>${match}</ol>`;
        });

        html = html.replace(/\n\n+/g, '</p><p>');
        html = '<p>' + html + '</p>';

        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p>(<h[1-3]>)/g, '$1');
        html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>(<(?:ol|ul)>)/g, '$1');
        html = html.replace(/(<\/(?:ol|ul)>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>(<li>)/g, '$1');
        html = html.replace(/(<li>[^<]*)<\/p>/g, '$1');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        if (this.elements.chatContainer) {
            this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
        }
    }

    clearMessages() {
        if (this.elements.messages) {
            this.elements.messages.innerHTML = '';
        }
    }

    renderMessages(messages) {
        if (!messages || messages.length === 0) return;

        messages.forEach(msg => {
            if (msg.role === 'user') {
                this.addMessageToUI(msg.content, 'user');
            } else {
                this.addMessageToUI(msg.content, 'ai', msg.sources || [], msg.retrievedCount || 0);
            }
        });
    }

    showWelcome() {
        if (this.elements.welcomeScreen) {
            this.elements.welcomeScreen.style.display = 'flex';
        }
    }

    hideWelcome() {
        if (this.elements.welcomeScreen) {
            this.elements.welcomeScreen.style.display = 'none';
        }
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        this.showToast(`已切换到${newTheme === 'dark' ? '深色' : '浅色'}模式`, 'success');
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    openDocumentModal() {
        this.elements.docModalOverlay?.classList.add('active');
        this.loadDocuments();
    }

    closeDocumentModal() {
        this.elements.docModalOverlay?.classList.remove('active');
        this.resetUploadProgress();
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast('文件大小超过 50MB 限制', 'error');
            return;
        }

        const allowedTypes = ['.pdf', '.txt', '.docx'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(fileExt)) {
            this.showToast('不支持的文件格式', 'error');
            return;
        }

        this.showUploadProgress(file.name);

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.updateUploadStatus('正在上传...', 0);

            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    this.updateUploadStatus('正在上传...', percent);
                }
            });

            const response = await new Promise((resolve, reject) => {
                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        reject(new Error(xhr.statusText));
                    }
                };
                xhr.onerror = () => reject(new Error('网络错误'));
                xhr.open('POST', `${API_BASE}/upload`);
                xhr.send(formData);
            });

            this.updateUploadStatus('✅ 上传成功！', 100);
            this.showToast(`"${file.name}" 上传成功`, 'success');

            setTimeout(() => {
                this.resetUploadProgress();
                this.loadDocuments();
            }, 1500);

        } catch (error) {
            console.error('上传失败:', error);
            this.updateUploadStatus('❌ 上传失败', 0);
            this.showToast('上传失败: ' + error.message, 'error');

            setTimeout(() => {
                this.resetUploadProgress();
            }, 3000);
        }
    }

    showUploadProgress(filename) {
        if (this.elements.uploadZone) this.elements.uploadZone.style.display = 'none';
        if (this.elements.uploadProgress) this.elements.uploadProgress.style.display = 'block';
        if (this.elements.progressFilename) this.elements.progressFilename.textContent = filename;
        if (this.elements.progressFill) this.elements.progressFill.style.width = '0%';
        if (this.elements.progressPercent) this.elements.progressPercent.textContent = '0%';
    }

    updateUploadStatus(status, percent) {
        if (this.elements.progressStatus) this.elements.progressStatus.textContent = status;
        if (this.elements.progressPercent) this.elements.progressPercent.textContent = `${percent}%`;
        if (this.elements.progressFill) this.elements.progressFill.style.width = `${percent}%`;
    }

    resetUploadProgress() {
        if (this.elements.uploadZone) this.elements.uploadZone.style.display = 'block';
        if (this.elements.uploadProgress) this.elements.uploadProgress.style.display = 'none';
        if (this.elements.fileInput) this.elements.fileInput.value = '';
    }

    async loadDocuments() {
        try {
            const response = await fetch(`${API_BASE}/documents`);

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            console.log('API 返回文档数据:', data);

            this.documents = Array.isArray(data.documents) ? data.documents : (Array.isArray(data) ? data : []);

            this.renderDocuments();
            this.updateDocCount();

        } catch (error) {
            console.error('加载文档失败:', error);
            this.showToast('加载文档列表失败: ' + error.message, 'warning');
        }
    }

    renderDocuments() {
        if (!this.elements.documentsList) return;

        if (!this.documents || this.documents.length === 0) {
            this.elements.documentsList.innerHTML = `
                <div class="empty-state" id="emptyDocsState">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    <p>暂无文档</p>
                    <span>上传您的第一个文档开始使用</span>
                </div>
            `;
            return;
        }

        try {
            this.elements.documentsList.innerHTML = this.documents.map(doc => {
                const docId = doc.doc_id || doc.id || 'unknown';
                const fileName = this.extractFileName(docId);
                const icon = this.getFileIcon(fileName);
                const uploadTime = this.formatDate(doc.created_at || doc.upload_time || new Date().toISOString());

                return `
                    <div class="document-item" data-doc-id="${this.escapeHtml(docId)}">
                        <div class="doc-icon">${icon}</div>
                        <div class="doc-info">
                            <div class="doc-name">${this.escapeHtml(fileName)}</div>
                            <div class="doc-meta">
                                <span>ID: ${this.escapeHtml(docId)}</span>
                                <span>${uploadTime}</span>
                                <span class="doc-status ready">✓ 已就绪</span>
                            </div>
                        </div>
                        <button class="doc-delete-btn" onclick="window.app.deleteDocument('${this.escapeHtml(docId)}', '${this.escapeHtml(fileName)}')" title="删除">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                `;
            }).join('');
        } catch (renderError) {
            console.error('渲染文档列表失败:', renderError);
            this.elements.documentsList.innerHTML = `
                <div class="empty-state">
                    <p>⚠️ 文档列表渲染出错</p>
                    <span>请刷新页面重试</span>
                </div>
            `;
        }
    }

    async deleteDocument(docId, docName) {
        this.showConfirm(
            '🗑️',
            '删除文档',
            `确定要删除 "${docName}" 吗？删除后相关的检索数据也会被清除。`,
            async () => {
                try {
                    const response = await fetch(`${API_BASE}/documents/${docId}`, {
                        method: 'DELETE'
                    });

                    if (!response.ok) throw new Error(`HTTP ${response.status}`);

                    await response.json();

                    this.documents = this.documents.filter(d =>
                        (d.doc_id || d.id) !== docId
                    );

                    this.renderDocuments();
                    this.updateDocCount();
                    this.showToast(`"${docName}" 已删除`, 'success');

                } catch (error) {
                    console.error('删除文档失败:', error);
                    this.showToast('删除失败: ' + error.message, 'error');
                }
            }
        );
    }

    updateDocCount() {
        if (this.elements.docCount) {
            this.elements.docCount.textContent = this.documents.length.toString();
        }
        if (this.elements.totalDocs) {
            this.elements.totalDocs.textContent = this.documents.length.toString();
        }
    }

    extractFileName(docId) {
        try {
            if (!docId) return '未知文档';

            const match = docId.match(/^doc_(.+)$/);
            if (match && match[1]) {
                let name = match[1];

                name = name.replace(/_/g, ' ');

                const commonExtensions = ['pdf', 'txt', 'docx', 'doc', 'md', 'xlsx', 'pptx'];

                for (const ext of commonExtensions) {
                    const lowerName = name.toLowerCase();
                    if (lowerName.endsWith(ext) && name.length > ext.length + 1) {
                        name = name.slice(0, -(ext.length + 1));
                        break;
                    }
                }

                return name.trim() || docId;
            }

            return docId;
        } catch (e) {
            console.warn('提取文件名失败:', e, docId);
            return String(docId || '未知文档');
        }
    }

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            pdf: '📕',
            txt: '📄',
            docx: '📘',
            md: '📝'
        };
        return icons[ext] || '📄';
    }

    formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            const now = new Date();
            const diff = now - date;

            if (diff < 60000) return '刚刚';
            if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
            if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
            if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`;

            return date.toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch {
            return '未知时间';
        }
    }

    formatTime(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleTimeString('zh-CN', {
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return '';
        }
    }

    showConfirm(icon, title, message, callback) {
        if (this.elements.confirmIcon) this.elements.confirmIcon.textContent = icon;
        if (this.elements.confirmTitle) this.elements.confirmTitle.textContent = title;
        if (this.elements.confirmMessage) this.elements.confirmMessage.textContent = message;

        this.confirmCallback = callback;
        this.elements.confirmModalOverlay?.classList.add('active');
    }

    closeConfirmModal() {
        this.elements.confirmModalOverlay?.classList.remove('active');
        this.confirmCallback = null;
    }

    showToast(message, type = 'info') {
        if (!this.elements.toastContainer) return;

        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
            <button class="toast-close" onclick="this.parentElement.classList.add('hiding'); setTimeout(() => this.parentElement.remove(), 300)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        `;

        this.elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    updateConnectionStatus(status) {
        if (!this.elements.statusIndicator) return;

        const dot = this.elements.statusIndicator.querySelector('.status-dot');
        const text = this.elements.statusIndicator.querySelector('span:last-child');

        if (dot && text) {
            dot.className = 'status-dot ' + status;

            switch (status) {
                case 'online':
                    text.textContent = '已连接';
                    text.style.color = 'var(--success-color)';
                    break;
                case 'loading':
                    text.textContent = '思考中...';
                    text.style.color = 'var(--warning-color)';
                    break;
                case 'error':
                    text.textContent = '连接异常';
                    text.style.color = 'var(--error-color)';
                    break;
                default:
                    text.textContent = '未知状态';
                    text.style.color = 'var(--text-muted)';
            }
        }
    }

    getErrorMessage(error) {
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            return '❌ 无法连接到服务器。请确保后端服务正在运行（端口 8000）。';
        }

        if (error.message.includes('404')) {
            return '❌ 请求的资源不存在。请检查 API 地址是否正确。';
        }

        if (error.message.includes('500')) {
            return '❌ 服务器内部错误。请稍后重试或联系管理员。';
        }

        return `❌ 发生错误：${error.message}。请重试。`;
    }

    saveChats() {
        try {
            localStorage.setItem('ragflow_chats', JSON.stringify(this.chats));
            localStorage.setItem('ragflow_current_chat', this.currentChatId || '');
        } catch (e) {
            console.error('保存聊天记录失败:', e);
        }
    }

    loadChats() {
        try {
            const savedChats = localStorage.getItem('ragflow_chats');
            const savedCurrentChat = localStorage.getItem('ragflow_current_chat');

            if (savedChats) {
                this.chats = JSON.parse(savedChats);
            }

            if (this.chats.length === 0) {
                this.createNewChat();
            } else {
                this.currentChatId = savedCurrentChat || this.chats[0].id;
                const currentChat = this.chats.find(c => c.id === this.currentChatId);

                if (currentChat) {
                    this.elements.headerTitle.textContent = currentChat.title;

                    if (currentChat.messages && currentChat.messages.length > 0) {
                        this.hideWelcome();
                        this.renderMessages(currentChat.messages);
                    } else {
                        this.showWelcome();
                    }
                }

                this.renderChatList();
            }
        } catch (e) {
            console.error('加载聊天记录失败:', e);
            this.createNewChat();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGFlowApp();
});
