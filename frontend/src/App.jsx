import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      text: 'Hello! I am your Retrieval-Augmented Generation (RAG) assistant. Ask me questions about the uploaded documents, and I will answer strictly based on retrieved context chunks.',
      sender: 'bot',
      meta: 'RAG Assistant • Grounded Context Active'
    }
  ]);
  const [userInput, setUserInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load API Key and files on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('rag_api_key') || '';
    setApiKey(savedKey);
    fetchStatus();
  }, []);

  // Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.files) {
        setIndexedFiles(data.files);
      }
    } catch (err) {
      console.error('Error fetching backend status:', err);
    }
  };

  const handleApiKeyChange = (e) => {
    const val = e.target.value.trim();
    setApiKey(val);
    localStorage.setItem('rag_api_key', val);
  };

  const askPreset = (questionText) => {
    sendMessage(questionText);
  };

  const handleSend = () => {
    if (!userInput.trim()) return;
    sendMessage(userInput);
    setUserInput('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const sendMessage = async (textToSend) => {
    const userMsgId = 'msg-' + Date.now();
    setMessages(prev => [...prev, {
      id: userMsgId,
      text: textToSend,
      sender: 'user',
      meta: 'You'
    }]);

    setLoading(true);
    const botLoadingId = 'msg-loading-' + Date.now();
    setMessages(prev => [...prev, {
      id: botLoadingId,
      text: 'Searching vector store and generating response...',
      sender: 'bot',
      meta: 'RAG Engine searching...'
    }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: textToSend,
          api_key: apiKey
        })
      });

      const data = await response.json();
      
      // Remove loading message
      setMessages(prev => prev.filter(m => m.id !== botLoadingId));

      // Append real response
      setMessages(prev => [...prev, {
        id: 'msg-ans-' + Date.now(),
        text: data.answer || "I don't know.",
        sender: 'bot',
        meta: `RAG Assistant (${data.provider || 'Grounded Context'})`
      }]);

      if (data.retrieved_chunks) {
        setRetrievedChunks(data.retrieved_chunks);
      } else {
        setRetrievedChunks([]);
      }
    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== botLoadingId));
      setMessages(prev => [...prev, {
        id: 'msg-err-' + Date.now(),
        text: 'Error communicating with RAG server.',
        sender: 'bot',
        meta: 'RAG Assistant'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleReindex = async () => {
    try {
      const res = await fetch('/api/reindex', { method: 'POST' });
      const data = await res.json();
      alert(data.message || 'Re-indexing complete!');
      fetchStatus();
    } catch (err) {
      alert('Re-indexing failed.');
    }
  };

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    setUploading(true);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      alert(data.message || 'File uploaded successfully!');
      fetchStatus();
    } catch (err) {
      alert('File upload failed.');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const toggleLeft = () => {
    setLeftOpen(!leftOpen);
    setRightOpen(false);
  };

  const toggleRight = () => {
    setRightOpen(!rightOpen);
    setLeftOpen(false);
  };

  const closeAll = () => {
    setLeftOpen(false);
    setRightOpen(false);
  };

  return (
    <div className="App" style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header>
        <div className="header-title">
          <button className="panel-toggle" onClick={toggleLeft}>☰ Docs</button>
          <h1>RAG Chatbot Dashboard</h1>
        </div>
        
        <div className="header-right">
          <div className="api-input-box">
            <label>API Key (Gemini/OpenAI):</label>
            <input 
              type="password" 
              placeholder="Paste key..." 
              value={apiKey}
              onChange={handleApiKeyChange}
            />
          </div>
          <button className="panel-toggle" onClick={toggleRight}>📋 Chunks</button>
        </div>
      </header>

      {/* Overlay */}
      <div 
        className={`panel-overlay ${leftOpen || rightOpen ? 'open' : ''}`} 
        onClick={closeAll}
      />

      {/* Main content grid */}
      <div className="main-container">
        
        {/* Left panel */}
        <div className={`panel panel-left ${leftOpen ? 'open' : ''}`}>
          <button className="panel-close" onClick={closeAll}>✕ Close</button>
          <div className="section-heading">Knowledge Base</div>
          <div className="card">
            <div className="drop-zone" onClick={triggerUpload}>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {uploading ? 'Uploading...' : 'Drag & Drop PDFs or TXT files here'}
              </p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                or click to select file
              </p>
              <input 
                type="file" 
                ref={fileInputRef}
                style={{ display: 'none' }} 
                accept=".txt,.pdf"
                onChange={handleFileUpload}
              />
            </div>
          </div>

          <div className="section-heading">Indexed Documents</div>
          <div className="card">
            <ul className="file-list">
              {indexedFiles.map((file, i) => (
                <li className="file-item" key={i}>
                  <span>{file}</span> 
                  <span style={{ color: '#10b981' }}>Indexed</span>
                </li>
              ))}
              {indexedFiles.length === 0 && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  No documents indexed.
                </p>
              )}
            </ul>
            <button 
              className="btn" 
              style={{ marginTop: '10px', background: 'rgba(255,255,255,0.08)' }} 
              onClick={handleReindex}
            >
              Re-Index Documents
            </button>
          </div>

          <div className="section-heading">RAG Pipeline Info</div>
          <div className="card" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            <p><strong>Embeddings:</strong> SentenceTransformers (all-MiniLM-L6-v2)</p>
            <p><strong>Vector DB:</strong> FAISS (L2 Cosine Distance)</p>
            <p><strong>Chunking:</strong> Recursive Character Splitter (500 tokens)</p>
            <p><strong>Strict Grounding:</strong> Enabled (Page 15 Rule)</p>
          </div>
        </div>

        {/* Center Panel */}
        <div className="chat-container">
          {/* Chips */}
          <div className="chips-container">
            <div className="chip" onClick={() => askPreset('What is RAG?')}>📖 What is RAG?</div>
            <div className="chip" onClick={() => askPreset('What is the hostel fee?')}>💡 Hostel fee?</div>
            <div className="chip" onClick={() => askPreset('When does admission open?')}>📅 Admission open?</div>
            <div className="chip" onClick={() => askPreset('What is the admission fee?')}>💰 Admission fee?</div>
            <div className="chip" onClick={() => askPreset('Who won the FIFA World Cup in 2030?')}>❓ FIFA 2030</div>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.map((m) => (
              <div className={`message ${m.sender}`} key={m.id}>
                <div className="message-bubble">{m.text}</div>
                <div className="message-meta">{m.meta}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Bar */}
          <div className="chat-input-container">
            <input 
              type="text" 
              className="chat-input" 
              placeholder="Ask a question about the document..." 
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
            />
            <button 
              className="btn send-btn" 
              onClick={handleSend}
              disabled={loading || !userInput.trim()}
            >
              Send
            </button>
          </div>
        </div>

        {/* Right Panel */}
        <div className={`panel panel-right ${rightOpen ? 'open' : ''}`}>
          <button className="panel-close" onClick={closeAll}>✕ Close</button>
          <div className="section-heading">Retrieved Chunks (Top-K)</div>
          <div id="chunksInspector">
            {retrievedChunks.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Retrieved context chunks will appear here after you ask a question.
              </p>
            ) : (
              retrievedChunks.map((chunk, i) => (
                <div className="chunk-card" key={i}>
                  <div className="chunk-score">Chunk #{i+1} • Distance Score: {chunk.similarity_score !== undefined ? chunk.similarity_score.toFixed(4) : (chunk.score !== undefined ? chunk.score.toFixed(4) : 'N/A')}</div>
                  <div className="chunk-text">"{chunk.content}"</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px', wordBreak: 'break-all' }}>
                    Source: {chunk.source}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
