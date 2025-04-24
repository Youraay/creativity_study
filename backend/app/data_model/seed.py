from pymongo import MongoClient
from utils import MongoDB
from datetime import datetime
from typing import List, Dict, Any, Optional

class Seed:
    def __init__(self, seed: int, db_client: MongoDB=None, load_from_db=True):
        
        self.id: str = str(seed)
        self.seed: int = seed
        
        
        self.creation_date: datetime = datetime.now()
        self.last_used_date: Optional[datetime] = None
        self.generated_image_ids: List[str] = []
        self.usage_count: int = 0
        self.associated_prompts: List[str] = []
        self.outlier_scores: Dict[str, float] = {}  
        self.coherence_scores: Dict[str, float] = {} 
        self.visual_quality_scores: Dict[str, float] = {} 
        self.tags: List[str] = []
        self.notes: str = ""
        self.parent_ids: List[str] = []
        self.mutation_origin: bool = False
        
        self.db_client = db_client
        loaded = False
        
        if db_client:
            if load_from_db:
                loaded =self._load_from_db()
            if not loaded: 
                print(f"Seed {self.id} not found in DB, creating new entry.")
                self.save_to_db()
    
    def _load_from_db(self):
        seed_data = self.db_client.get_item_from_collection_by_id("seeds", self.id)
        
        
        if seed_data:
            
            self.creation_date = seed_data.get("creation_date", self.creation_date)
            self.last_used_date = seed_data.get("last_used_date")
            self.generated_image_ids = seed_data.get("generated_image_ids", [])
            self.usage_count = seed_data.get("usage_count", 0)
            self.associated_prompts = seed_data.get("associated_prompts", [])
            self.outlier_scores = seed_data.get("outlier_scores", {})
            self.coherence_scores = seed_data.get("coherence_scores", {})
            self.visual_quality_scores = seed_data.get("visual_quality_scores", {})
            self.tags = seed_data.get("tags", [])
            self.notes = seed_data.get("notes", "")
            self.parent_ids = seed_data.get("parent_ids", [])
            self.mutation_origin = seed_data.get("mutation_origin", False)
            return True
    
    def save_to_db(self):
        if not self.db_client:
            return False
        
        self.last_used_date = datetime.now()
    
        return self.db_client.update_item_in_collection_by_id("seeds", self.id, self.to_dict())
        
    
    def add_generated_image(self, image_id: str, prompt: str):
        
        self.generated_image_ids.append(image_id)
        if prompt not in self.associated_prompts:
            self.associated_prompts.append(prompt)
        self.usage_count += 1
        
        
        if self.db_client:
            self.save_to_db()
    
    def set_outlier_score(self, prompt: str, score: float):
        self.outlier_scores[prompt] = score

    def set_coherence_score(self, prompt: str, score: float):
        
        self.coherence_scores[prompt] = score
        if self.db_client:
            self.save_to_db()

    def get_seed(self) -> int: 
        return self.seed
    def set_parents(self, parent1_id: str, parent2_id: str):
        self.parent_ids = [parent1_id, parent2_id]
        if self.db_client:
            self.save_to_db()

    def set_mutated(self, is_mutated: bool):
        self.mutation_origin = is_mutated
        if self.db_client:
            self.save_to_db()
    def get_id(self) -> str: 
        return self.id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "seed": self.seed,
            "creation_date": self.creation_date,
            "last_used_date": self.last_used_date,
            "generated_image_ids": self.generated_image_ids,
            "usage_count": self.usage_count,
            "associated_prompts": self.associated_prompts,
            "outlier_scores": self.outlier_scores,
            "coherence_scores": self.coherence_scores,
            "visual_quality_scores": self.visual_quality_scores,
            "tags": self.tags,
            "notes": self.notes,
            "parent_ids": self.parent_ids,
            "mutation_origin": self.mutation_origin
        }
    
    def get_average_outlier_score(self) -> float:
        
        if not self.outlier_scores:
            return 0.0
        
        return sum(self.outlier_scores.values()) / len(self.outlier_scores)

    def get_average_coherence_score(self) -> float:
        
        if not self.coherence_scores:
            return 0.0
        
        return sum(self.coherence_scores.values()) / len(self.coherence_scores)
    
    def get_average_quality_score(self) -> float:
        
        if not self.visual_quality_scores:
            return 0.0
        
        return sum(self.visual_quality_scores.values()) / len(self.visual_quality_scores)
    
    