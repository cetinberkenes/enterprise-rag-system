<<<<<<< HEAD
# enterprise-rag-system
=======
# RAG System with Huawei Cloud Integration

A production-ready Retrieval-Augmented Generation (RAG) system that seamlessly integrates with Huawei Cloud Object Storage Service (OBS) and local LLM inference using Ollama.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Huawei Cloud](https://img.shields.io/badge/Cloud-Huawei-red.svg)
![LLM](https://img.shields.io/badge/LLM-Llama%203.1-orange.svg)

## 🚀 Features

- **🌐 Cloud Integration**: Direct integration with Huawei Cloud Object Storage Service (OBS)
- **🤖 Local LLM Inference**: Uses Ollama for private, secure text generation
- **🔍 Multilingual Support**: Optimized for multilingual document processing
- **💾 Persistent Storage**: ChromaDB for efficient vector storage and retrieval
- **⚡ GPU Acceleration**: Automatic GPU detection and utilization
- **🛡️ Enterprise Ready**: Production-grade error handling and logging
- **📊 Monitoring**: Built-in system monitoring and statistics

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Huawei Cloud   │    │   PDF Document   │    │  Local Storage  │
│      OBS        │────┤    Processing    │────┤   ChromaDB      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Text Chunks    │
                       │   + Embeddings   │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Similarity Search│────│     Ollama      │
                       │   + Context      │    │   Llama 3.1     │
                       └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────────────────────────┐
                       │         Generated Answer             │
                       └──────────────────────────────────────┘
```

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 10GB+ free space for models and data

### Cloud Requirements
- **Huawei Cloud Account** with OBS access
- **OBS Bucket** with appropriate permissions

### Model Requirements
- **Ollama** installed and running
- **Llama 3.1** model (8B/13B/70B variants supported)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/enterprise-rag-system.git
cd enterprise-rag-system
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n rag-system python=3.10
conda activate rag-system

# Or using venv
python -m venv rag-system
source rag-system/bin/activate  # Linux/Mac
# rag-system\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

### 5. Download LLM Model
```bash
# Download Llama 3.1 8B (recommended for most use cases)
ollama pull llama3.1:8b

# For more powerful inference (requires more VRAM)
ollama pull llama3.1:13b
```

## ⚙️ Configuration

### 1. Environment Variables
Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your Huawei Cloud credentials:

```env
# Huawei Cloud OBS Configuration
OBS_ACCESS_KEY=your-access-key-here
OBS_SECRET_KEY=your-secret-key-here
OBS_ENDPOINT=obs.region.myhuaweicloud.com

# Optional: Model Configuration
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### 2. Huawei Cloud OBS Setup
1. Create an OBS bucket in your preferred region
2. Generate access credentials (Access Key ID and Secret Access Key)
3. Ensure your bucket has appropriate read permissions
4. Note your OBS endpoint URL

## 🚀 Quick Start

### Basic Usage with OBS

```python
from src.rag_system import EnterpriseRAGSystem

# Initialize the RAG system
rag = EnterpriseRAGSystem()

# Load document from Huawei Cloud OBS
chunk_count = rag.load_from_obs(
    bucket_name="your-bucket-name",
    object_key="path/to/your/document.pdf"
)

# Query the document
answer = rag.query("What are the main requirements mentioned in the document?")
print(answer)
```

### Local File Usage

```python
from src.rag_system import EnterpriseRAGSystem

# Initialize the RAG system
rag = EnterpriseRAGSystem()

# Load local PDF
chunk_count = rag.load_local_pdf("path/to/your/document.pdf")

# Query the document
answer = rag.query("Summarize the key points of this document")
print(answer)
```

### Interactive Testing

```bash
python src/test_interactive.py
```

## 📚 Usage Examples

### 1. Document Analysis
```python
# Technical document analysis
rag = EnterpriseRAGSystem(model_name="llama3.1:8b")
rag.load_from_obs("tech-docs", "system-requirements.pdf")

questions = [
    "What are the system requirements?",
    "What security measures are mentioned?",
    "What are the performance benchmarks?"
]

for question in questions:
    answer = rag.query(question, k=5)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### 2. Multilingual Documents
```python
# Optimized for multilingual content
rag = EnterpriseRAGSystem(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
rag.load_from_obs("docs", "multilingual-policy.pdf")

# Query in different languages
answer_en = rag.query("What is the privacy policy?")
answer_tr = rag.query("Gizlilik politikası nedir?")
```

### 3. Batch Processing
```python
# Process multiple documents
documents = [
    ("legal-docs", "contract-2024.pdf"),
    ("legal-docs", "policy-update.pdf"),
    ("legal-docs", "compliance-guide.pdf")
]

for bucket, document in documents:
    rag = EnterpriseRAGSystem()
    chunks = rag.load_from_obs(bucket, document)
    print(f"Processed {document}: {chunks} chunks")
```

## 🔍 API Reference

### EnterpriseRAGSystem Class

#### Constructor
```python
EnterpriseRAGSystem(
    model_name: str = "llama3.1:8b",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    vector_store_path: str = "./knowledge_base"
)
```

#### Methods

##### `load_from_obs(bucket_name, object_key, **kwargs) -> int`
Load and process a PDF from Huawei Cloud OBS.

- **bucket_name** (str): OBS bucket name
- **object_key** (str): Object path in bucket
- **Returns**: Number of text chunks created

##### `load_local_pdf(pdf_path) -> int`
Load and process a local PDF file.

- **pdf_path** (str): Path to local PDF file
- **Returns**: Number of text chunks created

##### `query(question, k=4, show_context=False, temperature=0.1) -> str`
Query the knowledge base using RAG.

- **question** (str): User question
- **k** (int): Number of relevant chunks to retrieve
- **show_context** (bool): Whether to log retrieved context
- **temperature** (float): LLM temperature for response generation
- **Returns**: Generated answer

##### `get_stats() -> Dict`
Get system statistics and information.

- **Returns**: Dictionary with system metrics

## 🛠️ Advanced Configuration

### Custom Model Configuration

```python
# Use different models for different use cases
rag_fast = EnterpriseRAGSystem(model_name="llama3.1:8b")    # Fast inference
rag_quality = EnterpriseRAGSystem(model_name="llama3.1:13b") # Higher quality

# Custom chunking strategy
rag_detailed = EnterpriseRAGSystem(
    chunk_size=500,     # Smaller chunks for detailed analysis
    chunk_overlap=100
)
```

### GPU Optimization

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# The system automatically uses GPU when available
# No manual configuration needed
```

## 📊 Performance Monitoring

### System Statistics
```python
# Get comprehensive system stats
stats = rag.get_stats()
print(f"Vector count: {stats['vector_count']}")
print(f"Model: {stats['model']}")
print(f"GPU available: {stats['gpu_available']}")
```

### Logging
The system uses Python's logging module for comprehensive monitoring:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Custom logger
logger = logging.getLogger("rag_system")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Test model
ollama run llama3.1:8b "Hello"
```

#### 2. OBS Connection Issues
```bash
# Verify credentials
echo $OBS_ACCESS_KEY
echo $OBS_ENDPOINT

# Test OBS connection manually
```

#### 3. GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Memory Issues
- Reduce `chunk_size` and `k` parameters
- Use smaller model variant (8B instead of 13B)
- Close other GPU-intensive applications

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Huawei Cloud** for robust cloud infrastructure
- **Meta AI** for the Llama 3.1 model
- **Ollama** for local LLM inference
- **LangChain** for RAG framework
- **ChromaDB** for vector storage


## 🗺️ Roadmap

- [ ] Web UI interface (Streamlit/Gradio)
- [ ] REST API endpoints
- [ ] Multi-document support
- [ ] Chat history and sessions
- [ ] Advanced filtering and search
- [ ] Integration with other cloud providers
- [ ] Docker containerization
- [ ] Kubernetes deployment

