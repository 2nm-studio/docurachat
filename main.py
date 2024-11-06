# main.py
import streamlit as st
import ollama
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
from config import Config
from models import Chat, ChatMessage
from chat_manager import ChatManager
from file_processor import FileProcessor
from embedding_manager import EmbeddingManager
from langchain_ollama import ChatOllama  
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from performance_monitor import PerformanceMonitor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Logger spécifique pour les performances
performance_logger = logging.getLogger('performance')
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(logging.StreamHandler())

def check_cuda():
    cuda_info = []
    cuda_info.append(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        cuda_info.extend([
            f"Version CUDA: {torch.version.cuda}",
            f"GPU: {torch.cuda.get_device_name(0)}",
            f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB",
            f"Mémoire utilisée: {torch.cuda.memory_allocated(0)/1024**2:.0f} MB",
            f"Mémoire cache: {torch.cuda.memory_reserved(0)/1024**2:.0f} MB"
        ])
    
    return "\n".join(cuda_info)

def validate_file(file):
    """Valide un fichier uploadé"""
    if not Config.is_file_size_valid(file.size):
        st.error(f"Le fichier {file.name} dépasse la taille maximale autorisée ({Config.MAX_FILE_SIZE_MB}MB)")
        return False
    return True

def format_chat_history(messages):
    """Formate l'historique des messages pour le contexte"""
    formatted = []
    for msg in messages[-5:]:  # Garder uniquement les 5 derniers messages
        role = "Human" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def process_file(file, file_processor, embedding_manager, chat_id, chat_manager):
    """Traite un fichier uploadé"""
    try:
        if not validate_file(file):
            return None
            
        logger.info(f"Processing file: {file.name}")
        
        # Libérer la mémoire GPU avant le traitement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        chunks = file_processor.process_file(file, file.type)
        
        if chunks:
            context = "\n\n".join(chunk.page_content for chunk in chunks)
            
            with st.spinner(f"Création de l'index vectoriel pour {file.name}..."):
                success = embedding_manager.process_chunks(chat_id, chunks, file.name)
                
            if success:
                chat_manager.update_chat_context(chat_id, [file.name], context)
                return context
                
        return None
        
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {e}")
        st.error(f"Erreur lors du traitement de {file.name}: {str(e)}")
        return None

def initialize_system():
    """Initialise les composants système avec gestion des erreurs"""
    try:
        config = Config()
        Config.ensure_directories()
        
        # Configuration CUDA
        try:
            Config.setup_cuda()
        except Exception as e:
            logger.warning(f"CUDA setup warning: {e}")
        
        chat_manager = ChatManager(config.CHAT_STORAGE_DIR)
        file_processor = FileProcessor(config)
        embedding_manager = EmbeddingManager(config)
        performance_monitor = PerformanceMonitor(config)
        
        return config, chat_manager, file_processor, embedding_manager, performance_monitor
        
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        raise

def main():
    try:
        st.set_page_config(
            page_title="Document Chat Bot",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session states
        if 'current_chat_id' not in st.session_state: 
            st.session_state.current_chat_id = None
        if 'chat_initialized' not in st.session_state: 
            st.session_state.chat_initialized = False
        if 'processing_files' not in st.session_state: 
            st.session_state.processing_files = False
        if 'processed_files' not in st.session_state: 
            st.session_state.processed_files = set()
        if 'selected_chat_title' not in st.session_state: 
            st.session_state.selected_chat_title = None
        if 'editing_title' not in st.session_state: 
            st.session_state.editing_title = False
        
        # System initialization avec progress bar
        with st.spinner("Initialisation du système..."):
            config, chat_manager, file_processor, embedding_manager, performance_monitor = initialize_system()
            
        # Start performance monitoring if enabled
        if config.ENABLE_PERFORMANCE_LOGGING:
            performance_monitor.start()
        
        # Sidebar
        with st.sidebar:
            st.title("Advanced Document Chat Bot")
            
            # CUDA Info
            with st.expander("Configuration CUDA", expanded=False):
                st.code(check_cuda())
            
            # Model selection
            available_models = ollama.list()['models']
            selected_model = st.selectbox(
                "Sélectionner un modèle",
                [model['name'] for model in available_models],
                help="Choisir le modèle IA à utiliser"
            )
            
            # Chat selection
            chat_ids = list(chat_manager.chats.keys())
            if not st.session_state.current_chat_id and chat_ids:
                last_chat_id = chat_manager.get_last_active_chat_id()
                if last_chat_id:
                    st.session_state.current_chat_id = last_chat_id
                    st.session_state.selected_chat_title = chat_manager.chats[last_chat_id].title
                    st.session_state.chat_initialized = True
            
            chat_titles = ["Nouvelle conversation"] + [
                chat_manager.chats[cid].title for cid in chat_ids
            ]
            
            selected_index = 0
            if st.session_state.selected_chat_title in chat_titles:
                selected_index = chat_titles.index(st.session_state.selected_chat_title)
            
            selected_chat = st.selectbox(
                "Sélectionner une conversation",
                chat_titles,
                key='chat_selector',
                index=selected_index
            )
            
            if selected_chat != st.session_state.selected_chat_title:
                if selected_chat != "Nouvelle conversation":
                    current_chat_id = next(
                        cid for cid in chat_ids 
                        if chat_manager.chats[cid].title == selected_chat
                    )
                    st.session_state.current_chat_id = current_chat_id
                    st.session_state.selected_chat_title = selected_chat
                    st.session_state.chat_initialized = True
                    st.rerun()
            
            # File upload
            uploaded_files = st.file_uploader(
                "Ajouter des documents (PDF, TXT, MD, JS, PHP)",
                type=["pdf", "txt", "md", "js", "php"],
                accept_multiple_files=True,
                help=f"Taille maximale: {Config.MAX_FILE_SIZE_MB}MB"
            )
            
            # Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Nouvelle conversation", use_container_width=True):
                    new_chat_id = chat_manager.create_new_chat()
                    st.session_state.current_chat_id = new_chat_id
                    st.session_state.selected_chat_title = chat_manager.chats[new_chat_id].title
                    st.session_state.chat_initialized = True
                    st.session_state.processed_files = set()
                    st.session_state.editing_title = False
                    st.rerun()
            
            with col2:
                if st.button("Supprimer", type="secondary", use_container_width=True):
                    if st.session_state.current_chat_id:
                        chat_id = st.session_state.current_chat_id
                        embedding_manager.delete_chat_store(chat_id)
                        chat_manager.delete_chat(chat_id)
                        st.session_state.current_chat_id = None
                        st.session_state.chat_initialized = False
                        st.session_state.processed_files = set()
                        st.session_state.selected_chat_title = None
                        st.rerun()
        
        # Main content area
        if st.session_state.current_chat_id:
            current_chat = chat_manager.chats[st.session_state.current_chat_id]
            
            # Chat title and edit button
            col1, col2 = st.columns([20, 1])
            with col1:
                if st.session_state.editing_title:
                    new_title = st.text_input(
                        "",
                        value=current_chat.title,
                        label_visibility="collapsed",
                        key="chat_title_input"
                    )
                    if new_title != current_chat.title:
                        chat_manager.rename_chat(st.session_state.current_chat_id, new_title)
                        st.session_state.selected_chat_title = new_title
                        st.session_state.editing_title = False
                        st.rerun()
                else:
                    st.title(current_chat.title)
            
            with col2:
                if st.button("✏️", key="edit_title", help="Modifier le titre"):
                    st.session_state.editing_title = not st.session_state.editing_title
                    st.rerun()
            
            # Process uploaded files
            if uploaded_files and not st.session_state.processing_files:
                new_files = [
                    file for file in uploaded_files 
                    if file.name not in st.session_state.processed_files
                ]
                
                if new_files:
                    st.session_state.processing_files = True
                    
                    if not st.session_state.current_chat_id:
                        st.session_state.current_chat_id = chat_manager.create_new_chat()
                        st.session_state.chat_initialized = True
                    
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(new_files):
                        progress = (idx) / len(new_files)
                        progress_bar.progress(progress)
                        
                        with st.status(f"Traitement de {file.name}...") as status:
                            context = process_file(
                                file,
                                file_processor,
                                embedding_manager,
                                st.session_state.current_chat_id,
                                chat_manager
                            )
                            
                            if context:
                                st.session_state.processed_files.add(file.name)
                                status.update(label=f"✅ {file.name} traité avec succès!")
                            else:
                                status.update(label=f"❌ Erreur lors du traitement de {file.name}")
                            time.sleep(0.5)
                    
                    progress_bar.progress(1.0)
                    st.session_state.processing_files = False
                    st.rerun()
            
            # Display chat messages
            for message in current_chat.messages:
                with st.chat_message(message.role):
                    timestamp = datetime.fromisoformat(message.timestamp).strftime("%H:%M:%S")
                    st.markdown(f"**{timestamp}**")
                    st.markdown(message.content)
            
            # Chat input
            if prompt := st.chat_input(
                "Quelle est votre question ?",
                key="chat_input",
                disabled=st.session_state.processing_files
            ):
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.status("Réflexion en cours...") as status:
                        try:
                            # Activer le logging performence pendant la génération
                            performance_monitor._log_to_file = True
                            
                            # Initialize LLM
                            llm = ChatOllama(
                                temperature=config.LLM_TEMPERATURE,
                                base_url=config.OLLAMA_API_BASE_URL,
                                model=selected_model,
                                top_k=config.LLM_TOP_K,
                                top_p=config.LLM_TOP_P,
                                num_ctx=config.LLM_NUM_CTX
                            )
                            
                            # Get retriever
                            retriever = embedding_manager.get_retriever(st.session_state.current_chat_id)
                            
                            if retriever:
                                logger.info(f"Retriever found: {retriever}")
                                
                                if 'qa_chain' not in st.session_state:
                                    st.session_state.qa_chain = chat_manager.create_qa_chain(
                                        retriever,
                                        llm
                                    )
                                
                                try:
                                    response = chat_manager.process_question(
                                        st.session_state.current_chat_id,
                                        prompt,
                                        st.session_state.qa_chain
                                    )
                                    
                                except Exception as e:
                                    logger.error(f"Error in QA chain: {e}")
                                    response = f"Désolé, une erreur s'est produite : {str(e)}"
                            else:
                                # Fallback to normal chat
                                response = llm.predict(f"""
                                Conversation précédente:
                                {format_chat_history(current_chat.messages)}
                                
                                Human: {prompt}
                                
                                Assistant:""")
                                
                                chat_manager.add_message(st.session_state.current_chat_id, "user", prompt)
                                chat_manager.add_message(st.session_state.current_chat_id, "assistant", response)
                            
                            st.markdown(response)
                            status.update(label="✅ Réponse générée!", state="complete")
                            
                            # Désactiver le logging performance
                            performance_monitor._log_to_file = False
                            
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            st.error(f"Une erreur est survenue: {str(e)}")
                            status.update(label="❌ Erreur", state="error")
                            performance_monitor._log_to_file = False
                    
                        st.rerun()

    except Exception as e:
        logger.error(f"Main application error: {e}")
        st.error("Une erreur critique est survenue. Veuillez rafraîchir la page ou contacter l'administrateur.")
        
    finally:
        # Nettoyage final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        Config.setup_cuda()  # Configure CUDA
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        raise