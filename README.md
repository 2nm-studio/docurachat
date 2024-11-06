# Advanced Document Chat Bot

An AI-powered document analysis and chat interface that enables contextual conversations with your documents using local LLMs through Ollama.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 📄 Support for multiple document formats (PDF, TXT, MD, JS, PHP)
- 💬 Context-aware conversations with your documents
- 🔄 Multiple chat sessions management
- 🧠 Local LLM support via Ollama
- 🎯 Advanced document chunking and embedding
- 💪 CUDA GPU acceleration support
- 🔍 Vector similarity search using FAISS
- 📊 Performance monitoring and logging

## 🔧 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- [Ollama](https://ollama.ai/) installed and running locally
- 16GB RAM minimum (32GB recommended)
- At least 2GB free disk space

## 🚀 Quick Start

1. Clone the repository:

```bash
git clone https://github.com/2nm-studio/docurachat
cd docurachat
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Ollama server locally and pull a model:

```bash
ollama pull mistral  # or any other supported model
```

5. Launch the application:

```bash
streamlit run main.py
```

## 📋 Environment Setup Checklist

- [ ] Python 3.8+ installed
- [ ] CUDA toolkit installed (if using GPU)
- [ ] Ollama installed and running
- [ ] At least one Ollama model pulled
- [ ] Required Python packages installed
- [ ] Storage directories configured
- [ ] CUDA memory fraction set (if using GPU)

## 🔩 Configuration

Key configuration options in `config.py`:

- `CHUNK_SIZE`: Document chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 10)
- `EMBEDDING_MODEL_NAME`: HuggingFace embedding model
- `CUDA_MEMORY_FRACTION`: GPU memory allocation (if using CUDA)
- `BATCH_SIZE`: Processing batch size
- `LLM_TEMPERATURE`: Model temperature setting
- `LLM_TOP_K`: Top K results for similarity search

## 🛠️ Project Structure

```
docurachat/
├── main.py              # Main Streamlit application
├── chat_manager.py      # Chat session management
├── embedding_manager.py # Document embedding and retrieval
├── file_processor.py    # Document processing
├── models.py           # Data models
└── config.py           # Configuration settings
```

## 🔄 Usage Flow

1. Start a new chat session
2. Upload one or more supported documents
3. Ask questions about your documents
4. Get AI-powered responses with context from your documents
5. Manage multiple chat sessions
6. Export or delete conversations as needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
