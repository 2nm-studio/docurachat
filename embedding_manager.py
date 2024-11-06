# embedding_manager.py
from pathlib import Path
import json
import torch
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.metadata_file = Path(config.VECTOR_STORE_DIR) / "metadata.json"
        self.metadata = self._load_metadata()

    def _init_embeddings(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Debug CUDA avec plus d'informations
            if torch.cuda.is_available():
                # Nettoyage mémoire préventif
                torch.cuda.empty_cache()
                
                logger.info(f"""
                CUDA Configuration:
                - Device: {device}
                - GPU Name: {torch.cuda.get_device_name(0)}
                - Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB
                - Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB
                - Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB
                - Memory Fraction: {self.config.CUDA_MEMORY_FRACTION}
                """)
            
            model_kwargs = {
                'device': device,
            }
            
            if device == "cuda":
                model_kwargs.update({
                    'use_auth_token': False,
                    # Force FP32 au lieu de FP16 pour plus de stabilité
                    'torch_dtype': torch.float32,
                })
            
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs={
                    'batch_size': self.config.BATCH_SIZE,
                    'device': device,
                    'show_progress_bar': True,
                    # Ajout d'un timeout plus long
                    'timeout': 300
                }
            )
            
            # Test d'embedding pour vérifier la configuration
            test_text = "Test embedding initialization"
            try:
                test_embedding = embeddings.embed_query(test_text)
                logger.info("Test embedding successful")
            except Exception as e:
                logger.error(f"Test embedding failed: {e}")
                raise
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.info("Falling back to CPU")
            return self._init_embeddings_cpu()

    def _init_embeddings_cpu(self):
        """Fallback pour initialisation CPU si GPU échoue"""
        return HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 4}
        )

    def _load_metadata(self) -> Dict:
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        return {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def get_chat_store_path(self, chat_id: str) -> Path:
        return Path(self.config.VECTOR_STORE_DIR) / f"store_{chat_id}"

    def process_chunks(self, chat_id: str, chunks: List, file_name: str) -> bool:
        store_path = self.get_chat_store_path(chat_id)
        
        try:
            logger.info(f"Processing {len(chunks)} chunks for {file_name}")
            
            # Utiliser torch.cuda.amp pour les calculs mixed-precision si disponible
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                with torch.cuda.amp.autocast():
                    return self._process_chunks_internal(store_path, chunks, chat_id, file_name)
            else:
                return self._process_chunks_internal(store_path, chunks, chat_id, file_name)
                
        except Exception as e:
            logger.error(f"Error processing chunks for {file_name}: {e}")
            return False

    def _process_chunks_internal(self, store_path, chunks, chat_id, file_name):
        batch_size = self.config.BATCH_SIZE
        
        try:
            if store_path.exists():
                vectorstore = FAISS.load_local(
                    str(store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Process chunks in smaller batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                    
                    # Clear CUDA cache before each batch if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    vectorstore.add_documents(batch)
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)

            # Save after processing
            vectorstore.save_local(str(store_path))
            
            # Update metadata
            if chat_id not in self.metadata:
                self.metadata[chat_id] = {"files": []}
            if file_name not in self.metadata[chat_id]["files"]:
                self.metadata[chat_id]["files"].append(file_name)
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in _process_chunks_internal: {e}")
            return False

    def get_retriever(self, chat_id: str):
        store_path = self.get_chat_store_path(chat_id)
        if not store_path.exists():
            return None
            
        try:
            vectorstore = FAISS.load_local(
                str(store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Configurer le retriever avec des paramètres optimisés
            return vectorstore.as_retriever(
                search_kwargs={
                    "k": self.config.LLM_TOP_K,
                    "fetch_k": self.config.LLM_TOP_K * 2,
                    "score_threshold": 0.5
                }
            )
        except Exception as e:
            logger.error(f"Error loading vector store for chat {chat_id}: {e}")
            return None

    def delete_chat_store(self, chat_id: str):
        try:
            store_path = self.get_chat_store_path(chat_id)
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)
            
            if chat_id in self.metadata:
                del self.metadata[chat_id]
                self._save_metadata()
                
            # Clear CUDA cache after deletion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error deleting chat store {chat_id}: {e}")