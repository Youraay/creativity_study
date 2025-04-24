from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.db_connection import MongoDB
import uuid
import config

class Experiment:
    def __init__(self,
                 prompt_id: str,
                 prompt_text: str,
                 db_client: Optional[MongoDB] = None,
                 experiment_id: Optional[str] = None,
                 load_from_db: bool = False):

        self.db_client = db_client
        self.id: str = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text
        self.timestamp = datetime.now()
        self.status = "initialized"

        
        self.config: Dict[str, Any] = {
            "population_size": config.POPULATION_SIZE,
            "elite_size": config.ELITE_SIZE,
            "mutation_rate": config.MUTATION_RATE,
            "max_generations": config.MAX_GENERATIONS,
            "patience": config.PATIENCE,
            "min_improvement": config.MIN_IMPROVEMENT,
            "use_blip2_qba": config.USE_BLIP2_QNA,

        }
        self.config["finess_weight"] = {
            "global_outlier": config.GLOABL_OUTLIER_WEIGHT,
            "local_outlier": config.LOCAL_OUTLIER_WEIGHT,
            "coherence": config.COHERENCE_WEIGHT,
            "quality": config.QUALITY_WEIGHT
        }

        self.models: Dict[str, str] = {
            "base_model": config.SDXL_BASE_MODEL,
            "refiner_model": config.SDXL_REF_MODEL,
            "clip_model": config.CLIP_MODEL,
            "blip_model": config.BLIP2_MODEL
        }

        self.generation_config: Dict[str, Any] = {
            "n_steps": config.N_STEPS,
            "high_noise_frac": config.HIGH_NOISE_FRAC,
            "guidance_scale": config.GUIDANCE_SCALE,
            "negative_prompt": config.NEGATIVE_PROMPT,
            "use_half_precision": config.USE_HALF_PRECISION
        }

        self.metadata: Dict[str, str] = {
            "category": "",
            "complexity": "",
            "object": ""
        }

        self.generation_ids: List[str] = []
        self.result_seed_ids: List[str] = []

        if db_client and load_from_db:
            self._load_from_db()
        elif db_client:
            self.save_to_db()


    def set_metadata(self, category: str, complexity: str, object: str):
        self.metadata = {
            "category": category,
            "complexity": complexity,
            "object": object
        }
        self.save_to_db()

    def add_generation(self, generation_id: str):
        self.generation_ids.append(generation_id)
        self.save_to_db()

    def set_final_results(self, seed_ids: List[str]):
        self.result_seed_ids = seed_ids
        self.status = "completed"
        self.save_to_db()

    def set_status(self, status: str):
        self.status = status
        self.save_to_db()

    def save_to_db(self):
        if self.db_client:
            self.db_client.update_item_in_collection_by_id("experiments", self.id, self.to_dict())

    def _load_from_db(self):
        data = self.db_client.get_item_from_collection_by_id("experiments", self.id)
        if data:
            self.prompt_id = data.get("prompt_id", self.prompt_id)
            self.prompt_text = data.get("prompt_text", self.prompt_text)
            self.timestamp = data.get("timestamp", self.timestamp)
            self.status = data.get("status", self.status)
            self.config = data.get("config", self.config)
            self.models = data.get("models", self.models)
            self.generation_config = data.get("generation_config", self.generation_config)
            self.metadata = data.get("metadata", {})
            self.generation_ids = data.get("generation_ids", [])
            self.result_seed_ids = data.get("result_seed_ids", [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "timestamp": self.timestamp,
            "status": self.status,
            "config": self.config,
            "models": self.models,
            "generation_config": self.generation_config,
            "metadata": self.metadata,
            "generation_ids": self.generation_ids,
            "result_seed_ids": self.result_seed_ids
        }
        