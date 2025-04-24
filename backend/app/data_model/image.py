from datetime import datetime
from typing import Dict, Any, Optional, List
from utils.db_connection import MongoDB
import os
from PIL import Image as PILImage
import base64
import io
from data_model import Seed, Prompt
import numpy as np
import torch
import config
class Image:

    @classmethod
    def from_dict(cls, data: Dict[str, Any], db_client: MongoDB) -> "Image":
        
        dummy_seed = Seed(data.get("seed_id", ""), db_client)
        dummy_prompt = Prompt(data.get("prompt_id", ""), db_client)

        img = cls(
            seed=dummy_seed,
            prompt=dummy_prompt,
            generation_steps=data.get("generation_steps", 0),
            guidance_scale=data.get("guidance_scale", 0.0),
            use_refiner=data.get("use_refiner", False),
            db_client=db_client,
            custom_id=data.get("_id"),
            file_path=data.get("file_path")
        )

        img.creation_date = data.get("creation_date", datetime.now())
        img.width = data.get("width", 0)
        img.height = data.get("height", 0)
        img.coherence_score = data.get("coherence_score", 0.0)
        img.quality_score = data.get("quality_score", 0.0)
        img.local_outlier_score = data.get("local_outlier_score", 0.0)
        img.global_outlier_score = data.get("global_outlier_score", 0.0)
        img.tags = data.get("tags", [])
        img.notes = data.get("notes", "")
        img.clip_embedding = data.get("clip_embedding", None)
        img.blip_embedding = data.get("blip_embedding", None)
        img.seed_id = data.get("seed_id")
        img.prompt_id = data.get("prompt_id")
        img.prompt_text = data.get("prompt_text")

        return img
    
    
    def __init__(self, 
                 seed: Seed, 
                 prompt: Prompt, 
                 pil_image: PILImage = None,
                 generation_steps: int = 0,
                 guidance_scale: float = 0.0,
                 use_refiner: bool = False,
                 db_client: MongoDB = None, 
                 load_from_db: bool = False,
                 custom_id: str = None,
                 file_path: str = None):
        
        generation_steps = config.N_STEPS
        guidance_scale = config.GUIDANCE_SCALE
        
        self.id = custom_id or f"{prompt.get_id()}_{seed.get_id()}_{generation_steps}_{guidance_scale}"
        if use_refiner:
            self.id += "_refined"
        
        
        
        self.file_path = file_path or f"results/images/{self.id}.png"
        
        
        self.seed_id = seed.get_id()
        self.prompt_id = prompt.get_id()
        self.prompt_text = prompt.get_prompt()
        
        
        self.creation_date = datetime.now()
        self.generation_steps = generation_steps
        self.guidance_scale = guidance_scale
        self.width = 0
        self.height = 0
        self.use_refiner = use_refiner
        
        
        self.coherence_score = 0.0
        self.quality_score = 0.0
        self.local_outlier_score = 0.0
        self.global_outlier_score = 0.0
        self.tags = []
        self.notes = ""
        self.clip_embedding = None  
        self.blip_embedding = None  
        
        self.db_client = db_client
        
        
        if pil_image:
            self.width, self.height = pil_image.size
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                pil_image.save(self.file_path)
                pil_image.close()
                print(f"Image Saved to Disk: {self.file_path}")
            except (OSError, IOError) as e:
                print(f"Fehler beim Speichern des Bildes: {e}")
            
            seed.add_generated_image(self.id, prompt.get_prompt())
            prompt.register_usage(self.seed_id, self.id)
            
            if db_client:
                print(f"Saved Image to Database {self.id}")
                self.save_to_db()
        
        elif db_client and load_from_db:
            self._load_from_db()

        
    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)
        return self
    
    def get_id(self) -> str:
        return self.id
    def _load_from_db(self):
        image_data = self.db_client.get_item_from_collection_by_id("images", self.id)
        
        if image_data:
            self.file_path = image_data.get("file_path", self.file_path)
            self.seed_id = image_data.get("seed_id", "")
            self.prompt_id = image_data.get("prompt_id", "")
            self.prompt_text = image_data.get("prompt_text", "")
            self.creation_date = image_data.get("creation_date", self.creation_date)
            self.generation_steps = image_data.get("generation_steps", 0)
            self.guidance_scale = image_data.get("guidance_scale", 0.0)
            self.width = image_data.get("width", 0)
            self.height = image_data.get("height", 0)
            self.use_refiner = image_data.get("use_refiner", False)
            self.coherence_score = image_data.get("coherence_score", 0.0)
            self.quality_score = image_data.get("quality_score", 0.0)
            self.local_outlier_score = image_data.get("local_outlier_score", 0.0)
            self.global_outlier_score = image_data.get("global_outlier_score", 0.0)
            self.tags = image_data.get("tags", [])
            self.notes = image_data.get("notes", "")
            clip_emb = image_data.get("clip_embedding")
            if clip_emb is not None:
                self.clip_embedding = np.array(clip_emb)

            blip_emb = image_data.get("blip_embedding")

            if blip_emb is not None:
                self.blip_embedding = np.array(blip_emb)
            return True
    
    def save_to_db(self):
        if not self.db_client:
            return False
        
        try:
            return self.db_client.update_item_in_collection_by_id("images", self.id, self.to_dict())
        except Exception as e:
            print(f"Fehler beim Speichern in DB: {e}")
            return False
    
   
    
    def load_image(self):
        if self.file_path and os.path.exists(self.file_path):
            return PILImage.open(self.file_path)
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "file_path": self.file_path,
            "seed_id": self.seed_id,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "creation_date": self.creation_date,
            "generation_steps": self.generation_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "use_refiner": self.use_refiner,
            "coherence_score": self.coherence_score,
            "quality_score": self.quality_score,
            "local_outlier_score": self.local_outlier_score,
            "global_outlier_score": self.global_outlier_score,
            "tags": self.tags,
            "notes": self.notes,
            "clip_embedding": self.clip_embedding.tolist() if self.clip_embedding is not None else None,
            "blip_embedding": self.blip_embedding.tolist() if self.blip_embedding is not None else None,
             }