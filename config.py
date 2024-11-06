# config.py
import torch
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class Config:
    # Model settings
    MODEL = "llama2"  # ou votre modèle par défaut
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API settings
    OLLAMA_API_BASE_URL = "http://localhost:11434"
    
    # Storage settings
    CHAT_STORAGE_DIR = Path("chat_storage")
    VECTOR_STORE_DIR = Path("vectorstore")
    
    # Processing settings optimisés pour Quadro M1000M
    CHUNK_SIZE = 384
    CHUNK_OVERLAP = 38
    
    # LLM settings
    LLM_TEMPERATURE = 0
    LLM_TOP_K = 10
    LLM_TOP_P = 0.3
    LLM_NUM_CTX = 3072
    
    # GPU settings ajustés pour Quadro M1000M
    BATCH_SIZE = 8
    MAX_GPU_MEMORY = "2GiB"
    CUDA_MEMORY_FRACTION = 0.6
    CUDA_VISIBLE_DEVICES = "0"
    CUDA_LAUNCH_BLOCKING = "1"
    
    # Performance settings
    TORCH_THREADS = 4
    CUDA_KERNEL_TIMEOUT = 100
    FAISS_GPU_ID = 0 if torch.cuda.is_available() else None
    EMBEDDING_DIMENSION = 384
    
    # File settings
    MAX_FILE_SIZE_MB = 50
    
    # UI settings
    PROCESSING_TIMEOUT = 300
    
    # Monitoring settings 
    ENABLE_PERFORMANCE_LOGGING = True
    PERFORMANCE_LOG_INTERVAL = 5
    
    @classmethod
    def ensure_directories(cls):
        cls.CHAT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_cuda(cls):
        if torch.cuda.is_available():
            try:
                # Limite l'utilisation mémoire GPU
                torch.cuda.set_per_process_memory_fraction(cls.CUDA_MEMORY_FRACTION)
                # Active la libération automatique de la mémoire
                torch.cuda.empty_cache()
                # Optimisations basiques seulement
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                
                # Configuration des variables d'environnement
                os.environ["CUDA_VISIBLE_DEVICES"] = cls.CUDA_VISIBLE_DEVICES
                os.environ["CUDA_LAUNCH_BLOCKING"] = cls.CUDA_LAUNCH_BLOCKING
                os.environ["MAX_SPLIT_SIZE_MB"] = str(int(float(cls.MAX_GPU_MEMORY[:-3]) * 1024))
                
                logger.info(f"""CUDA Setup completed:
                    - Device: {torch.cuda.get_device_name(0)}
                    - Memory Fraction: {cls.CUDA_MEMORY_FRACTION}
                    - Max Memory: {cls.MAX_GPU_MEMORY}
                    """)
            except Exception as e:
                logger.warning(f"Failed to setup CUDA optimally: {e}")
                
    @classmethod
    def is_file_size_valid(cls, file_size_bytes: int) -> bool:
        return file_size_bytes <= (cls.MAX_FILE_SIZE_MB * 1024 * 1024)