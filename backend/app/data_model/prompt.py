from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.db_connection import MongoDB
import torch
class Prompt:
    def __init__(self, prompt: str, db_client: MongoDB=None, load_from_db=True, id: str=None):
        
        self.prompt: str = prompt
        self.id: str = id if id else prompt[:20].replace(" ", "_").lower()
        self.creation_date: datetime = datetime.now()
        self.last_used_date: Optional[datetime] = None
        self.generated_image_ids: List[str] = []
        self.mean_embedding: Optional[List[float]] = None
        self.used_seed_ids: List[str] = []
        self.usage_count: int = 0
        self.tags: List[str] = []
        self.notes: str = ""
        self.metadata: Dict[str, Any] = {}
        self.clip_embedding = None
        self.diversity_score: float = 0.0  
        self.coherence_score: float = 0.0  
        
        
        self.db_client = db_client
        
        loaded = False
        if db_client:
            if load_from_db:
                loaded =self._load_from_db()
            if not loaded: 
                print(f"Prompt {self.id} not found in DB, creating new entry.")
                self.save_to_db()

    def _load_from_db(self):
        prompt_data = self.db_client.get_item_from_collection_by_id("prompts", self.id)
        
        if prompt_data:
            self.prompt = prompt_data.get("prompt", self.prompt)
            self.creation_date = prompt_data.get("creation_date", self.creation_date)
            self.last_used_date = prompt_data.get("last_used_date")
            self.generated_image_ids = prompt_data.get("generated_image_ids", [])
            self.used_seed_ids = prompt_data.get("used_seed_ids", [])
            self.usage_count = prompt_data.get("usage_count", 0)
            self.tags = prompt_data.get("tags", [])
            self.notes = prompt_data.get("notes", "")
            self.metadata = prompt_data.get("metadata", {})
            self.diversity_score = prompt_data.get("diversity_score", 0.0)
            self.coherence_score = prompt_data.get("coherence_score", 0.0)
            self.mean_embedding = prompt_data.get("mean_embedding", None)
            return True 
    
    
    def save_to_db(self):
        if not self.db_client:
            return False
        
        self.last_used_date = datetime.now()
        
        return self.db_client.update_item_in_collection_by_id("prompts", self.id, self.to_dict())
    
    def register_usage(self, seed_id: str, image_id: str):

        if seed_id not in self.used_seed_ids:
            self.used_seed_ids.append(seed_id)
        
        self.generated_image_ids.append(image_id)
        self.usage_count += 1
        
        if self.db_client:
            self.save_to_db()
    
    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)
            if self.db_client:
                self.save_to_db()
    
    def set_diversity_score(self, score: float):
        self.diversity_score = score
        if self.db_client:
            self.save_to_db()
    
    def get_prompt(self) -> str:
        return self.prompt
    
    def set_metadata(self, category: str, complexity: str, object: str):
        self.metadata = {
            "category": category,
            "complexity": complexity,
            "object": object
        }
        self.save_to_db()

    def get_id(self) -> str:
        return self.id
    
    def calculate_and_store_mean_embedding(self):
       
        if not self.db_client:
            return None
        
        
        embeddings = self.db_client.get_image_embeddings_for_prompt(self.id)
        if not embeddings:
            return None

        
        stacked = torch.tensor(embeddings)
        mean = torch.mean(stacked, dim=0)
        self.mean_embedding = mean.tolist()

        self.save_to_db()
        return self.mean_embedding

    def get_mean_embedding(self):
        
        if self.mean_embedding is not None:
            return self.mean_embedding
        print(f"Mean-Embedding für Prompt {self.id} nicht gefunden, berechne es neu.")
        return self.calculate_and_store_mean_embedding()


    def clean_mean_embedding(self):
        
        self.mean_embedding = None
       

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "prompt": self.prompt,
            "creation_date": self.creation_date,
            "last_used_date": self.last_used_date,
            "generated_image_ids": self.generated_image_ids,
            "used_seed_ids": self.used_seed_ids,
            "usage_count": self.usage_count,
            "tags": self.tags,
            "notes": self.notes,
            "diversity_score": self.diversity_score,
            "coherence_score": self.coherence_score,
            "clip_embedding": self.clip_embedding,
            "mean_embedding": self.mean_embedding,
            "metadata": self.metadata
        }
