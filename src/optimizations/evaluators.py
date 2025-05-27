from abc import ABC, abstractmethod
from custom_types import Argument,Argument2, Noise, Evaluation
from typing import Generic
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.manager import ModelManager

class Evaluator(Generic[Argument] , ABC):
    @abstractmethod
    def evaluate(self, noise: Argument, *args, **kwargs) -> Evaluation:

        raise NotImplementedError("Method is not implementet yet")

class MaxMeanDivergenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        
        self.mean_embds = torch.load(f"mean_embeddings/{mean_embds_file_name}")
        

    def evaluate(self, noise: Noise, *args, **kwargs) -> Evaluation:
        
        inputs = self.blip_processor(images=noise.pil, return_tensors="pt")
        with torch.no_grad():
            outputs =self.blip2_model.get_image_features(**inputs)
            noise.image_embs = outputs.last_hidden_state.mean(dim=1)

        cos = F.cosine_similarity(noise.image_embs, self.mean_embds)
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity: Evaluation = 1 - normalised_similarity

        return inversed_similarity
    
class MaxPromptCoherenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.clip_model, self.clip_processor = model_manager.load_clip()
    
    def evaluate(self, noise: Noise, *args, **kwargs) -> Evaluation:
        
        inputs = self.clip_processor(
            text=[kwargs.get("prompt")],
            images=noise.pil,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embs = outputs.image_embeds
            text_embs = outputs.text_embeds
            
            similarity = F.cosine_similarity(image_embs, text_embs, dim=-1).item()
            normalised_similarity: Evaluation = (similarity +1)/2
        
        return normalised_similarity
    
class MaxQualityEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        
        self.mean_embds = torch.load(f"mean_embeddings/{mean_embds_file_name}")
        

    def evaluate(self, noise: Noise) -> Evaluation:
        
        pass
        

        