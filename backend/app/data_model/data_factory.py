from utils import MongoDB
from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Union
from data_model.seed import Seed
from data_model.prompt import Prompt
from data_model.image import Image
import random

class DataFactory():

    def __init__(self, mongo: MongoDB) -> None:
        self.mongo = mongo
        self.db = mongo.get_database()
    
    
    
    def create_seed(self, seed_value: int = None) -> Seed:
     
        if seed_value is None:
            seed_value = random.randint(1, 2147483647)
        
        return Seed(seed_value, db_client=self.mongo)
    
    def get_seed(self, seed_id: str) -> Optional[Seed]:
      
        return Seed(int(seed_id), db_client=self.mongo, load_from_db=True)
    
    
    def find_creative_seeds(self, min_outlier_score: float = 0.7) -> List[Seed]:
    
        creative_seeds = []
        all_seeds = self.mongo.get_collection("seeds").find({})
        
        for seed_doc in all_seeds:
            if any(score >= min_outlier_score for score in seed_doc.get("outlier_scores", {}).values()):
                creative_seeds.append(Seed(int(seed_doc["_id"]), db_client=self.mongo, load_from_db=True))
        
        return creative_seeds
    
    
    
    def create_prompt(self, prompt_text: str, id: str = None) -> Prompt:
       
        return Prompt(prompt=prompt_text, db_client=self.mongo, id=id)
    
    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        
        return Prompt(prompt="", id=prompt_id, db_client=self.mongo, load_from_db=True)
    
    def find_prompts_by_category(self, category: str) -> List[Prompt]:
      
        prompt_docs = self.mongo.get_items_from_collection_by_attr("prompts", "category", category)
        return [Prompt(prompt="", id=doc["_id"], db_client=self.mongo, load_from_db=True) for doc in prompt_docs]
    
   
    def find_images_by_seed_and_prompt(self, seed_id: str, prompt_id: str) -> List[Image]:
       
        image_docs = self.mongo.get_collection("images").find(
            {"seed_id": seed_id, "prompt_id": prompt_id}
        )
        return [self.get_image(doc["_id"]) for doc in image_docs]
    
    def create_image(self, 
                    seed: Seed, 
                    prompt: Prompt, 
                    pil_image, 
                    generation_steps: int,
                    guidance_scale: float,
                    use_refiner: bool) -> Image:
       
        return Image(
            seed=seed,
            prompt=prompt,
            pil_image=pil_image,
            generation_steps=generation_steps,
            guidance_scale=guidance_scale,
            use_refiner=use_refiner,
            db_client=self.mongo
        )
    
    def get_image(self, image_id: str) -> Optional[Image]:
       
    
        image_doc = self.mongo.get_item_from_collection_by_id("images", image_id)
        
        if not image_doc:
            return None
            
        seed = self.get_seed(image_doc["seed_id"])
        prompt = self.get_prompt(image_doc["prompt_id"])
        
      
        return Image(
            seed=seed,
            prompt=prompt,
            custom_id=image_id,
            file_path=image_doc["file_path"],
            generation_steps=image_doc["generation_steps"],
            guidance_scale=image_doc["guidance_scale"],
            use_refiner=image_doc["use_refiner"],
            db_client=self.mongo,
            load_from_db=True
        )
    
    def find_images_by_seed(self, seed_id: str) -> List[Image]:
        
        image_docs = self.mongo.get_items_from_collection_by_attr("images", "seed_id", seed_id)
        return [self.get_image(doc["_id"]) for doc in image_docs]
    
    def find_images_by_prompt(self, prompt_id: str) -> List[Image]:
        
        image_docs = self.mongo.get_items_from_collection_by_attr("images", "prompt_id", prompt_id)
        return [self.get_image(doc["_id"]) for doc in image_docs]
    
    def find_creative_images(self, min_outlier_score: float = 0.7) -> List[Image]:
       
        
        image_docs = self.mongo.get_collection("images").find(
            {"outlier_score": {"$gte": min_outlier_score}}
        )
        
        return [self.get_image(doc["_id"]) for doc in image_docs]
    
    def get_image_by_seed_and_prompt(self, seed_id: str, prompt_id: str) -> Optional["Image"]:
   
        result = self.mongo.get_items_from_collection_by_attrs("images", {
            "seed_id": seed_id,
            "prompt_id": prompt_id
        })

        if result:
            return Image.from_dict(result, self) 
        return None
    
    def get_image_embeddings_for_prompt(self, prompt_id: str) -> List[List[float]]:
        images = self.mongo.get_items_from_collection_by_attr("images", "prompt_id")
        return [img["embedding"] for img in images if "embedding" in img]
    

    def get_images_by_config(self, prompt_id: str, steps: int, guidance: float) -> List[Image]:
    
        image_docs = self.mongo.get_collection("images").find(
            {"prompt_id": prompt_id,
             "generation_steps": steps,
             "guidance_scale": float(guidance)}
        )
        
        return [Image.from_dict(doc, self.mongo) for doc in image_docs]
    
    def get_all_images_with_embedding(self) -> List["Image"]:
     
        query = {"blip_embedding": {"$ne": None}}
        image_docs = self.mongo.get_collection("images").find(query)
        return [Image.from_dict(doc, self.mongo) for doc in image_docs]
