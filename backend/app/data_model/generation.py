from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.db_connection import MongoDB
import uuid
import config


class Generation:
    def __init__(self,
                 experiment_id: str,
                 generation_index: int,
                 prompt_id: str,
                 seed_ids: List[str],
                 image_ids: List[str],
                 elite_seed_ids: List[str],
                 avg_fitness: float,
                 best_fitness: float,
                 diversity: float,
                 db_client: Optional[MongoDB] = None,
                 generation_id: Optional[str] = None,
                 load_from_db: bool = False):

        self.db_client = db_client
        self.id: str = generation_id or f"gen_{experiment_id}_{generation_index}"
        self.prompt_id: str = prompt_id
        self.experiment_id = experiment_id
        self.generation_index = generation_index
        self.seed_ids = seed_ids
        self.image_ids = image_ids
        self.elite_seed_ids = elite_seed_ids
        self.avg_fitness = avg_fitness
        self.best_fitness = best_fitness
        self.diversity = diversity
        self.timestamp = datetime.now()

        if db_client and load_from_db:
            self._load_from_db()
        elif db_client:
            self.save_to_db()

    def save_to_db(self):
        if self.db_client:
            self.db_client.update_item_in_collection_by_id("generations", self.id, self.to_dict())

    def _load_from_db(self):
        data = self.db_client.get_item_from_collection_by_id("generations", self.id)
        if data:
            self.seed_ids = data.get("seed_ids", [])
            self.prompt_id = data.get("prompt_id", self.prompt_id)
            self.image_ids = data.get("image_ids", [])
            self.elite_seed_ids = data.get("elite_seed_ids", [])
            self.avg_fitness = data.get("avg_fitness", 0.0)
            self.best_fitness = data.get("best_fitness", 0.0)
            self.diversity = data.get("diversity", 0.0)
            self.timestamp = data.get("timestamp", self.timestamp)

    def get_id(self) -> str:
        return self.id
    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "experiment_id": self.experiment_id,
            "generation_index": self.generation_index,
            "prompt_id": self.prompt_id,
            "seed_ids": self.seed_ids,
            "image_ids": self.image_ids,
            "elite_seed_ids": self.elite_seed_ids,
            "avg_fitness": self.avg_fitness,
            "best_fitness": self.best_fitness,
            "diversity": self.diversity,
            "timestamp": self.timestamp
        }
