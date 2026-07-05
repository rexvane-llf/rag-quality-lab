import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat'); // chat, documents, history

  // 加载文档列表
  useEffect(() => {
    loadDocuments();
    loadConversations();
  }, []);

  const loadDocuments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/documents`);
      setDocuments(res.data);
    } catch (error) {
      console.error('加载文档失败:', error);
    }
  };

  const loadConversations = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/conversations`);
      setConversations(res.data);
    } catch (error) {
      console.error('加载对话历史失败:', error);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      await axios.post(`${API_BASE}/api/upload`, formData);
      alert('文档上传成功！');
      loadDocuments();
    } catch (error) {
      alert('上传失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!question.trim()) return;

    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/api/chat`, {
        question: question,
        top_k: 5
      });
      setAnswer(res.data);
      loadConversations();
      setQuestion('');
    } catch (error) {
      alert('查询失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (docId) => {
    if (!confirm('确定删除这个文档吗？')) return;

    try {
      await axios.delete(`${API_BASE}/api/documents/${docId}`);
      loadDocuments();
    } catch (error) {
      alert('删除失败: ' + error.message);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* 顶部导航 */}
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            📚 个人知识库 AI 助手
          </h1>
        </div>
      </nav>

      {/* 标签切换 */}
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex space-x-4 border-b">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-4 py-2 font-medium ${
              activeTab === 'chat'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
          >
            💬 聊天
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            className={`px-4 py-2 font-medium ${
              activeTab === 'documents'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
          >
            📄 文档管理
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 font-medium ${
              activeTab === 'history'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600'
            }`}
          >
            🕒 对话历史
          </button>
        </div>
      </div>

      {/* 内容区域 */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* 聊天界面 */}
        {activeTab === 'chat' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">和你的知识库聊天</h2>

            {/* 答案显示 */}
            {answer && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-gray-800 mb-4">{answer.answer}</p>

                {answer.sources && answer.sources.length > 0 && (
                  <div className="mt-4">
                    <p className="font-semibold text-sm text-gray-600 mb-2">来源：</p>
                    {answer.sources.map((source, idx) => (
                      <div key={idx} className="text-sm text-gray-600 mb-1">
                        [{source.index}] {source.document} - {source.heading}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* 输入框 */}
            <div className="flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="问点什么..."
                className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <button
                onClick={handleSearch}
                disabled={loading || !question.trim()}
                className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300"
              >
                {loading ? '查询中...' : '发送'}
              </button>
            </div>
          </div>
        )}

        {/* 文档管理 */}
        {activeTab === 'documents' && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">文档管理</h2>
              <label className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 cursor-pointer">
                上传文档
                <input
                  type="file"
                  onChange={handleUpload}
                  accept=".pdf,.txt,.md"
                  className="hidden"
                />
              </label>
            </div>

            {documents.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                还没有文档，上传一个开始吧！
              </p>
            ) : (
              <div className="space-y-2">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex justify-between items-center p-4 border rounded-lg hover:bg-gray-50"
                  >
                    <div>
                      <p className="font-medium">{doc.name}</p>
                      <p className="text-sm text-gray-500">
                        {doc.type.toUpperCase()} · {(doc.size / 1024).toFixed(2)} KB · {doc.chunk_count} 个切片
                      </p>
                    </div>
                    <button
                      onClick={() => handleDelete(doc.id)}
                      className="px-3 py-1 text-red-600 hover:bg-red-50 rounded"
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* 对话历史 */}
        {activeTab === 'history' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">对话历史</h2>

            {conversations.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                还没有对话历史
              </p>
            ) : (
              <div className="space-y-4">
                {conversations.map((conv) => (
                  <div key={conv.id} className="border-b pb-4">
                    <p className="font-medium text-gray-800 mb-2">
                      Q: {conv.question}
                    </p>
                    <p className="text-gray-600 text-sm">
                      A: {conv.answer.substring(0, 200)}...
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      {new Date(conv.created_at).toLocaleString('zh-CN')}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
