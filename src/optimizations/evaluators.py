from abc import ABC, abstractmethod
from ..custom_types import Argument,Argument2, Noise, Evaluation
from typing import Generic, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..models.manager import ModelManager


class Evaluator(Generic[Argument] , ABC):
    @abstractmethod
    def evaluate(self, noise: Argument, *args, **kwargs):

        raise NotImplementedError("Method is not implementet yet")

class MaxMeanDivergenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
        self.name = "BLIP2_Max_Mean_Divergence"
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        mean_embds_file_name = mean_embds_file_name.replace(" ", "-")
        self.mean_embds = torch.load(f"/scratch/dldevel/sinziri/creativity_study/src/mean_embeddings/{mean_embds_file_name}.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        
        

    def evaluate(self, noise: Noise, *args, **kwargs):
        # Generate BLIP embeddings if not already present
        if noise.image_embs is None:
            noise.scores[self.name] = 0.5
            # inputs = self.blip_processor(images=noise.pil, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            # with torch.no_grad():
            #     outputs = self.blip2_model.get_image_features(**inputs)
            #     noise.image_embs = outputs.last_hidden_state.mean(dim=1)
        img = noise.image_embs.unsqueeze(0)    # → [1, D]
        cos = F.cosine_similarity(img, self.mean_embds) # → [1, D]
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity: Evaluation = 1 - normalised_similarity
        noise.scores[self.name] = inversed_similarity
        
        

class MaxLocalMeanDivergenceEvaluator(Evaluator[Noise]):

    def __init__(self) -> None:
        
        self.name = "BLIP2_Max_Local_Mean_Divergence"
        
        
        
        

    def evaluate(self, noise: Noise, *args, **kwargs):
        
        local_mean_embs = kwargs.get("local_mean_embeddings")   
        if local_mean_embs is None:
            print("Warning: local_mean_embeddings not provided to MaxLocalMeanDivergenceEvaluator")
            return
            
        # Ensure image_embs is available
        if noise.image_embs is None:
            print("Warning: noise.image_embs is None in MaxLocalMeanDivergenceEvaluator")
            return

        cos = F.cosine_similarity(noise.image_embs, local_mean_embs)
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity: Evaluation = 1 - normalised_similarity
        noise.scores[self.name] = inversed_similarity
        


class MaxPromptCoherenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
        self.name = "CLIP_Max_Prompt_Coherence"
        self.clip_model, self.clip_processor = model_manager.load_clip()
    
    def evaluate(self, noise: Noise, *args, **kwargs):
        
        inputs = self.clip_processor(
            text=[kwargs.get("prompt")],
            images=noise.pil,
            return_tensors="pt",
            padding=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            noise.clip_embeddings = outputs.image_embeds
            text_embs = outputs.text_embeds
            
            similarity = F.cosine_similarity(noise.clip_embeddings, text_embs, dim=-1).item()
            normalised_similarity: Evaluation = (similarity +1)/2
            noise.scores[self.name] = normalised_similarity
        
    
class MaxQualityEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
        self.name = "MaxQualityEvaluator"
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        mean_embds_file_name = mean_embds_file_name.replace(" ", "_")
        self.mean_embds = torch.load(f"mean_embeddings/{mean_embds_file_name}.pt")
        

    def evaluate(self, noise: Noise) -> None:
        
        pass
        

class BLIP2ImageCaption(Evaluator[Noise]):

    def __init__(self):
        model_manager= ModelManager()
        self.blip2_model, self.blip_processor = model_manager.load_blip2_for_generation()

    def evaluate(self, noise: Noise, *args, **kwargs):
    
        inputs = self.blip_processor(noise.pil, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            generated_ids = self.blip2_model.generate(
                **inputs, 
                max_new_tokens=50,  
                do_sample=True,     
                num_beams=4,        
                temperature=0.7,    
                min_length=5,       
                repetition_penalty=1.2
            )
            generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"Generated Caption:{generated_text}")
        noise.image_caption = generated_text

