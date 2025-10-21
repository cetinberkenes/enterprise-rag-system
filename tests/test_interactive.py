"""
Interactive Testing Script for Enterprise RAG System
===================================================

This script provides an interactive command-line interface for testing
the Enterprise RAG System with various document types and queries.

Usage:
    python test_interactive.py

Features:
- Interactive Q&A session
- Multiple document loading options
- Sample queries for different document types
- Performance monitoring
- Error handling and recovery


Created: 2025
License: MIT
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional, List

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from rag_system import EnterpriseRAGSystem
except ImportError:
    print("❌ Error: Cannot import EnterpriseRAGSystem")
    print("💡 Make sure you're running this from the project root directory")
    print("💡 And that src/rag_system.py exists")
    sys.exit(1)

import logging

# Configure logging for interactive mode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class InteractiveRAGTester:
    """Interactive testing interface for the RAG system."""
    
    def __init__(self):
        """Initialize the interactive tester."""
        self.rag_system: Optional[EnterpriseRAGSystem] = None
        self.document_loaded = False
        self.query_count = 0
        
    def display_banner(self):
        """Display welcome banner."""
        print("\n" + "="*70)
        print("🚀 ENTERPRISE RAG SYSTEM - INTERACTIVE TESTER")
        print("="*70)
        print("🤖 AI-Powered Document Q&A System")
        print("☁️ Integrated with Huawei Cloud OBS")
        print("🔍 Powered by Llama 3.1 + ChromaDB")
        print(f"⏰ Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system with user preferences."""
        print("\n🔧 SYSTEM INITIALIZATION")
        print("-" * 30)
        
        try:
            # Model selection
            print("\n📋 Available Models:")
            print("1. llama3.1:8b  (Fast, balanced)")
            print("2. llama3.1:13b (Higher quality, slower)")
            print("3. llama3.1:70b (Best quality, requires powerful GPU)")
            print("4. Custom model")
            
            choice = input("\nSelect model (1-4, default=1): ").strip()
            
            model_mapping = {
                "1": "llama3.1:8b",
                "2": "llama3.1:13b", 
                "3": "llama3.1:70b",
                "": "llama3.1:8b"  # default
            }
            
            if choice in model_mapping:
                model_name = model_mapping[choice]
            elif choice == "4":
                model_name = input("Enter custom model name: ").strip()
                if not model_name:
                    model_name = "llama3.1:8b"
            else:
                print("⚠️ Invalid choice, using default: llama3.1:8b")
                model_name = "llama3.1:8b"
            
            print(f"✅ Selected model: {model_name}")
            
            # Initialize RAG system
            print("\n🚀 Initializing RAG system...")
            self.rag_system = EnterpriseRAGSystem(model_name=model_name)
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def load_document_menu(self) -> bool:
        """Display document loading options and handle user choice."""
        print("\n📄 DOCUMENT LOADING OPTIONS")
        print("-" * 35)
        print("1. Load from Huawei Cloud OBS")
        print("2. Load local PDF file")
        print("3. Use example document (demo)")
        print("4. Skip loading (use existing)")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            return self._load_from_obs()
        elif choice == "2":
            return self._load_local_file()
        elif choice == "3":
            return self._load_example_document()
        elif choice == "4":
            if self.document_loaded:
                print("✅ Using existing document")
                return True
            else:
                print("❌ No document currently loaded")
                return False
        else:
            print("❌ Invalid choice")
            return False
    
    def _load_from_obs(self) -> bool:
        """Load document from Huawei Cloud OBS."""
        print("\n☁️ HUAWEI CLOUD OBS CONFIGURATION")
        print("-" * 40)
        
        # Check environment variables
        if not all([os.getenv('OBS_ACCESS_KEY'), os.getenv('OBS_SECRET_KEY'), os.getenv('OBS_ENDPOINT')]):
            print("⚠️ OBS credentials not found in environment variables")
            print("💡 Please set OBS_ACCESS_KEY, OBS_SECRET_KEY, and OBS_ENDPOINT")
            
            setup_choice = input("Configure credentials now? (y/n): ").lower()
            if setup_choice == 'y':
                access_key = input("OBS Access Key: ").strip()
                secret_key = input("OBS Secret Key: ").strip()
                endpoint = input("OBS Endpoint (e.g., obs.ap-southeast-3.myhuaweicloud.com): ").strip()
                
                # Set for current session
                os.environ['OBS_ACCESS_KEY'] = access_key
                os.environ['OBS_SECRET_KEY'] = secret_key
                os.environ['OBS_ENDPOINT'] = endpoint
            else:
                return False
        
        bucket_name = input("Enter bucket name: ").strip()
        object_key = input("Enter object key (file path): ").strip()
        
        if not bucket_name or not object_key:
            print("❌ Bucket name and object key are required")
            return False
        
        try:
            print(f"\n📥 Loading {object_key} from bucket {bucket_name}...")
            chunk_count = self.rag_system.load_from_obs(bucket_name, object_key)
            
            if chunk_count > 0:
                print(f"✅ Successfully loaded {chunk_count} chunks")
                self.document_loaded = True
                return True
            else:
                print("❌ Failed to load document")
                return False
                
        except Exception as e:
            print(f"❌ OBS loading error: {e}")
            return False
    
    def _load_local_file(self) -> bool:
        """Load local PDF file."""
        print("\n📁 LOCAL FILE LOADING")
        print("-" * 25)
        
        # Show current directory files
        current_dir = os.getcwd()
        pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
        
        if pdf_files:
            print(f"\n📋 PDF files in current directory ({current_dir}):")
            for i, file in enumerate(pdf_files, 1):
                print(f"{i}. {file}")
            print(f"{len(pdf_files) + 1}. Enter custom path")
            
            choice = input(f"\nSelect file (1-{len(pdf_files) + 1}): ").strip()
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(pdf_files):
                    file_path = pdf_files[choice_num - 1]
                elif choice_num == len(pdf_files) + 1:
                    file_path = input("Enter full file path: ").strip()
                else:
                    print("❌ Invalid choice")
                    return False
            except ValueError:
                file_path = choice  # Assume direct path input
        else:
            print("📂 No PDF files found in current directory")
            file_path = input("Enter full file path: ").strip()
        
        if not file_path:
            print("❌ File path is required")
            return False
        
        try:
            print(f"\n📄 Loading {file_path}...")
            chunk_count = self.rag_system.load_local_pdf(file_path)
            
            if chunk_count > 0:
                print(f"✅ Successfully loaded {chunk_count} chunks")
                self.document_loaded = True
                return True
            else:
                print("❌ Failed to load document")
                return False
                
        except Exception as e:
            print(f"❌ File loading error: {e}")
            return False
    
    def _load_example_document(self) -> bool:
        """Load example document for demonstration."""
        print("\n🎯 EXAMPLE DOCUMENT LOADING")
        print("-" * 32)
        print("This option requires an example PDF file in the examples/ directory")
        
        example_files = [
            "examples/sample_technical_doc.pdf",
            "examples/sample_policy.pdf",
            "examples/sample_manual.pdf"
        ]
        
        for example_file in example_files:
            if os.path.exists(example_file):
                try:
                    chunk_count = self.rag_system.load_local_pdf(example_file)
                    if chunk_count > 0:
                        print(f"✅ Loaded example document: {example_file}")
                        print(f"📊 Created {chunk_count} chunks")
                        self.document_loaded = True
                        return True
                except Exception as e:
                    print(f"❌ Failed to load {example_file}: {e}")
                    continue
        
        print("❌ No example documents found")
        print("💡 Place a PDF file in the examples/ directory to use this option")
        return False
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries based on document type."""
        return [
            "What are the main topics discussed in this document?",
            "Can you summarize the key points?",
            "What are the requirements mentioned?",
            "What are the security considerations?",
            "How should the system be implemented?",
            "What are the performance criteria?",
            "What are the compliance requirements?",
            "What are the best practices mentioned?",
            "What are the technical specifications?",
            "What are the operational procedures?"
        ]
    
    def run_interactive_session(self):
        """Run the main interactive Q&A session."""
        if not self.document_loaded:
            print("❌ No document loaded. Please load a document first.")
            return
        
        print("\n💬 INTERACTIVE Q&A SESSION")
        print("-" * 35)
        print("🎯 Ask questions about your document!")
        print("💡 Type 'help' for sample questions")
        print("💡 Type 'stats' for system statistics")
        print("💡 Type 'quit' to exit")
        
        sample_queries = self.get_sample_queries()
        
        while True:
            try:
                question = input(f"\n🔍 Question #{self.query_count + 1}: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for testing the RAG system!")
                    break
                
                elif question.lower() == 'help':
                    print("\n📝 Sample Questions:")
                    for i, sample in enumerate(sample_queries[:5], 1):
                        print(f"{i}. {sample}")
                    print("...")
                    continue
                
                elif question.lower() == 'stats':
                    stats = self.rag_system.get_stats()
                    print("\n📊 System Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue
                
                # Process the question
                self.query_count += 1
                print(f"\n🤖 Processing query #{self.query_count}...")
                
                start_time = time.time()
                answer = self.rag_system.query(question, k=4)
                end_time = time.time()
                
                print(f"\n📝 Answer:")
                print("-" * 50)
                print(answer)
                print("-" * 50)
                print(f"⏱️ Response time: {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                continue
        
        print(f"\n📊 Session Summary:")
        print(f"   Total questions asked: {self.query_count}")
        print(f"   Session duration: {datetime.now().strftime('%H:%M:%S')}")
    
    def run(self):
        """Main entry point for the interactive tester."""
        self.display_banner()
        
        # Initialize system
        if not self.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Load document
        while not self.document_loaded:
            if not self.load_document_menu():
                retry = input("\n🔄 Would you like to try again? (y/n): ").lower()
                if retry != 'y':
                    print("👋 Exiting without loading document")
                    return
        
        # Run interactive session
        self.run_interactive_session()

def main():
    """Main function to run the interactive tester."""
    try:
        tester = InteractiveRAGTester()
        tester.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.exception("Fatal error in interactive tester")

if __name__ == "__main__":
    main()
