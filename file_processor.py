# file_processor.py
import tempfile
import os
from typing import List, Optional
from pathlib import Path
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def process_file(self, file_obj, file_type: str):
        temp_path = None
        try:
            # Créer un fichier temporaire avec un suffixe approprié
            suffix = '.pdf' if file_type == "application/pdf" else '.txt'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_obj.getvalue())  # Utiliser getvalue() au lieu de read()
                temp_path = temp_file.name

            logger.info(f"Processing file at temporary path: {temp_path}")
            
            # Choisir le bon loader
            if file_type == "application/pdf":
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)

            # Charger et découper le document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Successfully processed {len(chunks)} chunks from {file_obj.name}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing file {file_obj.name}: {e}")
            return None

        finally:
            # Nettoyer le fichier temporaire
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file {temp_path}: {e}")