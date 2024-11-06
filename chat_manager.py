# chat_manager.py
from typing import Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime
import uuid
import logging
from models import Chat, ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class Document:
    def __init__(self, content):
        self.page_content = content
class ChatManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_chat_id: Optional[str] = None
        self.chats: Dict[str, Chat] = {}
        self.load_all_chats()

    def create_new_chat(self, files: List[str] = None, initial_context: str = "") -> str:
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Générer un titre par défaut plus descriptif
        if files and len(files) > 0:
            if len(files) == 1:
                title = f"Chat about {files[0]}"
            else:
                title = f"Chat about {len(files)} files"
        else:
            title = f"New Chat ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
        chat = Chat(
            id=chat_id,
            title=title,
            messages=[],
            context=initial_context,
            files=files or [],
            created_at=timestamp,
            last_updated=timestamp
        )
        
        self.chats[chat_id] = chat
        self.current_chat_id = chat_id
        self.save_chat(chat_id)
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str) -> str:
        if chat_id not in self.chats:
            raise ValueError(f"Chat {chat_id} not found")
            
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self.chats[chat_id].messages.append(message)
        self.chats[chat_id].last_updated = datetime.now().isoformat()
        self.save_chat(chat_id)
        return message.message_id

    def get_chat_context(self, chat_id: str) -> str:
        if chat_id not in self.chats:
            return ""
            
        chat = self.chats[chat_id]
        context = chat.context + "\n\n"
        
        for msg in chat.messages:
            context += f"{msg.role}: {msg.content}\n"
            
        return context.strip()

    def save_chat(self, chat_id: str):
        if chat_id not in self.chats:
            return
            
        chat = self.chats[chat_id]
        chat_path = self.storage_dir / f"{chat_id}.json"
        
        with open(chat_path, 'w', encoding='utf-8') as f:
            json.dump(chat.to_dict(), f, ensure_ascii=False, indent=2)

    def load_chat(self, chat_id: str) -> Optional[Chat]:
        chat_path = self.storage_dir / f"{chat_id}.json"
        if not chat_path.exists():
            return None
            
        with open(chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
            
        messages = [
            ChatMessage(**msg) for msg in chat_data["messages"]
        ]
        
        return Chat(
            id=chat_data["id"],
            title=chat_data["title"],
            messages=messages,
            context=chat_data["context"],
            files=chat_data["files"],
            created_at=chat_data["created_at"],
            last_updated=chat_data["last_updated"]
        )

    def load_all_chats(self):
        for chat_file in self.storage_dir.glob("*.json"):
            chat_id = chat_file.stem
            chat = self.load_chat(chat_id)
            if chat:
                self.chats[chat_id] = chat

    def delete_chat(self, chat_id: str):
        if chat_id not in self.chats:
            return
            
        chat_path = self.storage_dir / f"{chat_id}.json"
        if chat_path.exists():
            chat_path.unlink()
            
        del self.chats[chat_id]
        if self.current_chat_id == chat_id:
            self.current_chat_id = None

    def update_chat_context(self, chat_id: str, files: List[str], new_context: str):
        """Met à jour le contexte d'un chat existant avec de nouveaux fichiers"""
        if chat_id not in self.chats:
            raise ValueError(f"Chat {chat_id} not found")
            
        chat = self.chats[chat_id]
        chat.files.extend(files)
        chat.context += f"\n\n{new_context}"
        if len(files) > 0:
            if len(chat.files) == 1:
                chat.title = f"Chat about {chat.files[0]}"
            else:
                chat.title = f"Chat about {len(chat.files)} files"
        chat.last_updated = datetime.now().isoformat()
        
        self.save_chat(chat_id)

    def rename_chat(self, chat_id: str, new_title: str):
        """Renomme un chat existant"""
        if chat_id not in self.chats:
            raise ValueError(f"Chat {chat_id} not found")
            
        self.chats[chat_id].title = new_title
        self.chats[chat_id].last_updated = datetime.now().isoformat()
        self.save_chat(chat_id)

    def get_last_active_chat_id(self) -> Optional[str]:
        """Retourne l'ID du chat le plus récemment mis à jour"""
        if not self.chats:
            return None
            
        return max(
            self.chats.items(),
            key=lambda x: x[1].last_updated
        )[0]
    


    def format_docs(self, docs):
        """Combine documents, ensuring each has 'page_content'."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                logger.info(f"Document {i+1} is a string; converting to Document.")
                formatted_docs.append(Document(doc))
            elif hasattr(doc, 'page_content'):
                formatted_docs.append(doc)
            else:
                logger.warning(f"Document {i+1} is of unexpected type {type(doc)}; converting to Document.")
                formatted_docs.append(Document(str(doc)))  # Convert any unexpected types to string

        # Format all documents by joining their 'page_content'
        return "\n\n".join(doc.page_content for doc in formatted_docs)

    def create_qa_chain(self, retriever, llm):
        """Crée une chaîne de question-réponse en incluant l'historique des conversations."""
        logger.info("Creating QA chain with conversation history...")
        
        prompt = ChatPromptTemplate.from_template("""
        Tu es un assistant expert qui aide à comprendre des documents.
        Utilise le contexte et l'historique de la conversation ci-dessous pour répondre à la question.
        Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
        
        Historique de la conversation:
        {chat_history}
        
        Contexte du document:
        {context}
        
        Question actuelle: {question}
        
        Réponse:""")

        def get_chat_history(chat_id):
            chat = self.chats.get(chat_id)
            if not chat:
                return ""
            # Prendre les 5 derniers échanges
            relevant_messages = chat.messages[-10:]
            formatted_history = []
            for msg in relevant_messages:
                role = "Human" if msg.role == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.content}")
            return "\n".join(formatted_history)
        

        def format_docs_with_debug(docs):
            """Format les documents avec des logs de debug"""
            logger.info(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs, 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                logger.info(f"Content length: {len(doc.page_content)}")
                logger.info(f"Content preview: {doc.page_content[:200]}...")
            
            formatted = self.format_docs(docs)
            logger.info(f"Total formatted context length: {len(formatted)}")
            return formatted

        qa_chain = (
            {
                "context": RunnablePassthrough() | self.format_docs,
                "chat_history": lambda x: get_chat_history(self.current_chat_id),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("QA chain with history created successfully")
        return qa_chain

    def process_question(self, chat_id: str, question: str, qa_chain) -> str:
        """Traite une question et enregistre la conversation."""
        try:
            logger.info(f"Processing question for chat {chat_id}")
            
            # Génère la réponse
            response = qa_chain.invoke(question)
            
            # Enregistre la question et la réponse
            self.add_message(chat_id, "user", question)
            self.add_message(chat_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise e