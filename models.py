# models.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str
    message_id: str = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Chat:
    id: str
    title: str
    messages: List[ChatMessage]
    context: str
    files: List[str]
    created_at: str
    last_updated: str
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "context": self.context,
            "files": self.files,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }