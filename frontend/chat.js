class ChatApp {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.apiUrl = 'http://localhost:8000';
        
        this.init();
    }
    
    init() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage(data.response, 'assistant');
            } else {
                this.addMessage('Sorry, I encountered an error.', 'assistant');
            }
        } catch (error) {
            this.addMessage('Sorry, I cannot connect to the server.', 'assistant');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        if (sender === 'assistant') {
            messageDiv.innerHTML = `
                <img src="assets/half body shmulik.png" alt="Shmulik" class="message-avatar">
                <div class="message-content">${content}</div>
            `;
        } else {
            messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    setLoading(loading) {
        this.sendButton.disabled = loading;
        this.sendButton.textContent = loading ? 'Sending...' : 'Send';
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
