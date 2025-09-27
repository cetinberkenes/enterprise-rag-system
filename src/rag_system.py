"""
Enterprise RAG System with Huawei Cloud OBS Integration
======================================================

A production-ready Retrieval-Augmented Generation system that integrates with:
- Huawei Cloud Object Storage Service (OBS)
- Local Ollama LLM inference (Llama 3.1:8B)
- ChromaDB vector storage
- Multilingual embeddings

Author: Enes Çetinberk
Created: 2025
License: MIT
"""

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import os
from dotenv import load_dotenv
from datetime import datetime
import psutil
from typing import Dict, List, Optional, Tuple
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseRAGSystem:
    """
    Enterprise-grade RAG system with cloud storage integration.
    
    Features:
    - Huawei Cloud OBS integration
    - Local LLM inference with Ollama
    - Multilingual document processing
    - Persistent vector storage
    - GPU acceleration support
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        vector_store_path: str = "./knowledge_base"
    ):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Ollama model name for text generation
            embedding_model: HuggingFace model for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            vector_store_path: Path for persistent vector storage
        """
        logger.info("🚀 Initializing Enterprise RAG System...")
        logger.info(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        
        # System information
        self._log_system_info()
        
        # Initialize embedding model
        logger.info("🔍 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = None
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        logger.info("✅ RAG System initialized successfully!")
    
    def _log_system_info(self) -> None:
        """Log system information including CPU, RAM, and GPU details."""
        # CPU/RAM info
        cpu_count = psutil.cpu_count()
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        logger.info(f"💻 System: {cpu_count} CPU cores, {ram_gb}GB RAM")
        
        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            logger.info(f"🎮 GPU: {gpu_name}, {vram_gb}GB VRAM")
            
            # GPU utilization (optional)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    logger.info(f"📊 GPU Load: {gpu.load*100:.1f}%, Memory: {gpu.memoryUtil*100:.1f}%")
            except ImportError:
                logger.info("📊 GPUtil not installed, skipping GPU monitoring")
            except Exception:
                pass
        else:
            logger.warning("⚠️ No GPU detected, using CPU for embeddings")
    
    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama model."""
        try:
            logger.info(f"🤖 Testing {self.model_name} connection...")
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user', 
                    'content': 'Hello! Please respond briefly to confirm you are working.'
                }],
                options={
                    'num_predict': 20,
                    'temperature': 0.1
                }
            )
            logger.info(f"✅ Model response: {response['message']['content'][:100]}...")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            raise ConnectionError("Ollama model is not accessible!")
    
    def load_from_obs(
        self,
        bucket_name: str,
        object_key: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> int:
        """
        Load PDF from Huawei Cloud Object Storage Service.
        
        Args:
            bucket_name: OBS bucket name
            object_key: Object path in bucket
            access_key: OBS access key (optional, reads from env)
            secret_key: OBS secret key (optional, reads from env)
            endpoint: OBS endpoint (optional, reads from env)
            
        Returns:
            Number of chunks created
            
        Raises:
            ImportError: If OBS SDK is not installed
            ConnectionError: If OBS connection fails
            FileNotFoundError: If object doesn't exist
        """
        try:
            from obs import ObsClient
        except ImportError:
            logger.error("❌ Huawei OBS SDK not installed!")
            logger.info("💡 Install with: pip install esdk-obs-python")
            raise ImportError("OBS SDK required for cloud storage access")
        
        logger.info("☁️ Loading document from Huawei Cloud OBS...")
        logger.info(f"🪣 Bucket: {bucket_name}")
        logger.info(f"📄 Object: {object_key}")
        
        # Get credentials
        access_key = access_key or os.getenv('OBS_ACCESS_KEY')
        secret_key = secret_key or os.getenv('OBS_SECRET_KEY')
        endpoint = endpoint or os.getenv('OBS_ENDPOINT')
        
        if not all([access_key, secret_key, endpoint]):
            logger.error("❌ OBS credentials missing!")
            logger.info("💡 Set environment variables:")
            logger.info("   OBS_ACCESS_KEY=your-access-key")
            logger.info("   OBS_SECRET_KEY=your-secret-key")
            logger.info("   OBS_ENDPOINT=obs.region.myhuaweicloud.com")
            raise ValueError("OBS credentials required")
        
        logger.info(f"📡 Endpoint: {endpoint}")
        
        # Initialize OBS client
        obs_client = ObsClient(
            access_key_id=access_key,
            secret_access_key=secret_key,
            server=endpoint
        )
        
        # Generate safe local filename
        safe_filename = object_key.replace('/', '_').replace(' ', '_')
        local_path = f"./{safe_filename}"
        
        try:
            logger.info("📥 Starting download...")
            
            # Download from OBS
            response = obs_client.getObject(bucket_name, object_key, downloadPath=local_path)
            
            if response.status < 300:
                file_size_mb = round(os.path.getsize(local_path) / (1024*1024), 2)
                logger.info("✅ Download successful!")
                logger.info(f"📊 File size: {file_size_mb}MB")
                logger.info(f"💾 Local path: {local_path}")
                
                # Process the downloaded PDF
                chunk_count = self.load_local_pdf(local_path)
                
                # Optionally remove temporary file
                # os.remove(local_path)
                
                return chunk_count
            else:
                error_msg = getattr(response, 'errorMessage', 'Unknown error')
                raise ConnectionError(f"OBS download failed: Status {response.status}, Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ OBS download error: {e}")
            
            # Log detailed error information
            if hasattr(e, 'errorCode'):
                logger.error(f"🔍 Error Code: {e.errorCode}")
            if hasattr(e, 'errorMessage'):
                logger.error(f"🔍 Error Message: {e.errorMessage}")
                
            raise
        
        finally:
            obs_client.close()
    
    def load_local_pdf(self, pdf_path: str) -> int:
        """
        Load and process a local PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks created
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        logger.info(f"📄 Loading PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check file size
        file_size_mb = round(os.path.getsize(pdf_path) / (1024*1024), 2)
        logger.info(f"📊 PDF size: {file_size_mb}MB")
        
        # Load PDF
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"📄 Loaded {len(documents)} pages")
        except Exception as e:
            logger.error(f"❌ PDF loading error: {e}")
            raise
        
        # Analyze content
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"📊 Total characters: {total_chars:,}")
        
        if total_chars < 1000:
            logger.warning("⚠️ Very little text found, OCR might be needed")
        
        # Show content sample
        if documents:
            sample_text = documents[0].page_content[:200].replace('\n', ' ')
            logger.info(f"📝 Content sample: {sample_text}...")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""],
            keep_separator=True
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"✂️ Created {len(chunks)} chunks")
        
        # Show chunk sample
        if chunks:
            sample_chunk = chunks[0].page_content.replace('\n', ' ')[:150]
            logger.info(f"📝 Chunk sample: {sample_chunk}...")
        
        # Create vector store
        logger.info("🔍 Creating embeddings and vector store...")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_store_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("✅ Knowledge base created successfully!")
            logger.info(f"📊 Vector count: {self.vectorstore._collection.count()}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Vector store creation error: {e}")
            raise
    
    def query(
        self,
        question: str,
        k: int = 4,
        show_context: bool = False,
        temperature: float = 0.1
    ) -> str:
        """
        Query the knowledge base using RAG.
        
        Args:
            question: User question
            k: Number of relevant chunks to retrieve
            show_context: Whether to log retrieved context
            temperature: LLM temperature for response generation
            
        Returns:
            Generated answer
            
        Raises:
            ValueError: If knowledge base is not loaded
            Exception: If query processing fails
        """
        if not self.vectorstore:
            raise ValueError("Knowledge base not loaded! Please load a document first.")
        
        logger.info(f"🔍 Processing query: {question}")
        
        try:
            # Similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
            
            if not docs_with_scores:
                return "❌ No relevant content found."
            
            # Filter by relevance score
            relevant_docs = [doc for doc, score in docs_with_scores if score < 0.8]
            
            if not relevant_docs:
                return "❌ No sufficiently relevant content found. Please try rephrasing your question."
            
            logger.info(f"📊 Found {len(relevant_docs)} relevant chunks")
            
            # Build context
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                clean_content = doc.page_content.replace('\n', ' ').strip()
                context_parts.append(f"[SOURCE {i}]\n{clean_content}")
            
            context = "\n\n".join(context_parts)
            
            # Create system prompt
            system_prompt = """You are an expert assistant specialized in analyzing technical documents. 
Provide accurate, detailed answers based only on the provided context."""

            user_prompt = f"""Based on the following document sources, answer the question:

{context}

QUESTION: {question}

INSTRUCTIONS:
- Use only information from the provided sources
- Provide detailed and technical answers when appropriate
- Indicate which source(s) you're referencing with [SOURCE X]
- If the information is not clearly available, state "The information is not clearly specified in the documents"

ANSWER:"""

            # Generate response using Ollama
            logger.info("🤖 Generating response...")
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                    'num_ctx': 4096
                }
            )
            
            answer = response['message']['content']
            
            # Optionally show context
            if show_context:
                logger.info(f"📚 Retrieved context:\n{context[:400]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system information
        """
        if not self.vectorstore:
            return {"status": "Knowledge base not loaded"}
        
        return {
            'vector_count': self.vectorstore._collection.count(),
            'model': self.model_name,
            'embedding_model': self.embeddings.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'vector_store_path': self.vector_store_path,
            'gpu_available': torch.cuda.is_available()
        }
    
    def list_collections(self) -> List[str]:
        """List available vector store collections."""
        if not self.vectorstore:
            return []
        
        try:
            return [collection.name for collection in self.vectorstore._client.list_collections()]
        except Exception:
            return []

def main():
    """Main function for testing the RAG system."""
    # Initialize RAG system
    rag = EnterpriseRAGSystem()
    
    # Configuration
    use_obs = True  # Set to False to use local file
    bucket_name = "your-bucket-name"
    object_key = "your-document.pdf"
    local_pdf_path = "your-document.pdf"
    
    # Load document
    try:
        if use_obs:
            logger.info("🌐 Loading document from OBS...")
            chunk_count = rag.load_from_obs(bucket_name, object_key)
        else:
            logger.info("📁 Loading local document...")
            chunk_count = rag.load_local_pdf(local_pdf_path)
        
        if chunk_count > 0:
            logger.info(f"✅ Successfully loaded {chunk_count} chunks")
            
            # Example queries
            example_questions = [
                "What are the main requirements mentioned in the document?",
                "What are the security criteria?",
                "How should the system infrastructure be designed?",
                "What are the backup and recovery procedures?",
                "What are the performance requirements?"
            ]
            
            logger.info("🎯 Running example queries...")
            
            for i, question in enumerate(example_questions, 1):
                print(f"\n{'='*70}")
                print(f"🔎 EXAMPLE QUERY {i}/{len(example_questions)}")
                print(f"Question: {question}")
                
                try:
                    answer = rag.query(question, k=3)
                    print(f"\n📝 Answer:\n{answer}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                print(f"{'='*70}")
            
            # System statistics
            print(f"\n📊 System Statistics:")
            stats = rag.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
                
        else:
            logger.error("❌ Document loading failed")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
