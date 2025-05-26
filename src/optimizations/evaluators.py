from abc import ABC, abstractmethod
from custom_types import Argument, Noise, Evaluation
from typing import Generic
import torch
import torch.nn as nn
from models.manager import ModelManager
class Evaluator(Generic[Argument] , ABC):
    @abstractmethod
    def evaluate(self, noise: Argument) -> Evaluation:

        raise NotImplementedError("Method is not implementet yet")

class MaxDifferenceToMeanBlip2Embedding(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        
        self.mean_embds = torch.load(f"mean_embeddings/{mean_embds_file_name}")
        

    def evaluate(self, noise: Noise) -> Evaluation:
        
        inputs = self.blip_processor(images=noise.pil, return_tensors="pt")
        with torch.no_grad():
            outputs =self.blip2_model.get_image_features(**inputs)
            blip_embs = outputs.last_hidden_state.mean(dim=1)

        cos = torch.nn.functional.cosine_similarity(noise.image_embs, self.mean_embds)
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity = 1 - normalised_similarity

        return inversed_similarity
    
class MaxPromptCoherence(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.clip_model, self.clip_processor = model_manager.load_clip()

        

    def evaluate(self, noise: Noise, prompt: str) -> Evaluation:
        
        inputs = self.clip_processor(text=[prompt], images=noise.pil, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_emb = outputs.image_embeds  
            text_emb = outputs.text_embeds    
        
        cos = torch.nn.functional.cosine_similarity(image_emb, text_emb)
        similarity = cos.item()
        normalised_similarity = (similarity +1)/2
        

        return normalised_similarity