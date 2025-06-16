from abc import ABC, abstractmethod
from ..custom_types import Argument,Argument2, Noise, Evaluation
from typing import Generic, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..models.manager import ModelManager

class Evaluator(Generic[Argument] , ABC):
    @abstractmethod
    def evaluate(self, noise: Argument, *args, **kwargs) -> Tuple[Evaluation, str]:

        raise NotImplementedError("Method is not implementet yet")

class MaxMeanDivergenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
        self.name = "BLIP2_Max_Mean_Divergence"
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        mean_embds_file_name = mean_embds_file_name.replace(" ", "-")
        self.mean_embds = torch.load(f"/scratch/dldevel/sinziri/creativity_study/src/mean_embeddings/{mean_embds_file_name}.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        
        

    def evaluate(self, noise: Noise, *args, **kwargs) -> Tuple[Evaluation, str]:
        
            
        inputs = self.blip_processor(images=noise.pil, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs =self.blip2_model.get_image_features(**inputs)

            noise.image_embs = outputs.last_hidden_state.mean(dim=1)

        cos = F.cosine_similarity(noise.image_embs, self.mean_embds)
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity: Evaluation = 1 - normalised_similarity
        noise.scores[self.name] = normalised_similarity
        
        return inversed_similarity, self.name

class MaxLocalMeanDivergenceEvaluator(Evaluator[Noise]):

    def __init__(self) -> None:
        model_manager = ModelManager()
        self.name = "BLIP2_Max_Local_Mean_Divergence"
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        
        
        
        

    def evaluate(self, noise: Noise, *args, **kwargs) -> Tuple[Evaluation, str]:
        
        local_mean_embs = kwargs.get("local_mean_embeddings")   
        inputs = self.blip_processor(images=noise.pil, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs =self.blip2_model.get_image_features(**inputs)
            blip_embeddings = outputs.last_hidden_state.mean(dim=1)

        cos = F.cosine_similarity(blip_embeddings, local_mean_embs)
        similarity =cos.item()
        normalised_similarity = (similarity +1)/2
        inversed_similarity: Evaluation = 1 - normalised_similarity
        noise.scores[self.name] = normalised_similarity
        return inversed_similarity, self.name


class MaxPromptCoherenceEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
        self.name = "CLIP_Max_Prompt_Coherence"
        self.clip_model, self.clip_processor = model_manager.load_clip()
    
    def evaluate(self, noise: Noise, *args, **kwargs) -> Tuple[Evaluation, str]:
        
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
        return normalised_similarity, self.name
    
class MaxQualityEvaluator(Evaluator[Noise]):

    def __init__(self,
                mean_embds_file_name: str ) -> None:
        model_manager = ModelManager()
                
        self.blip2_model, self.blip_processor = model_manager.load_blip2()
        mean_embds_file_name = mean_embds_file_name.replace(" ", "_")
        self.mean_embds = torch.load(f"mean_embeddings/{mean_embds_file_name}.pt")
        

    def evaluate(self, noise: Noise) -> Evaluation:
        
        pass
        

class BLIP2ImageCaption(Evaluator[Noise]):

    def __init__(self):
        model_manager= ModelManager()
        self.blip2_model, self.blip_processor = model_manager.load_blip2_for_generation()

        def evaluate(self, noise: Noise, *args, **kwargs) -> Tuple[Evaluation, str]:
        
            inputs = self.blip_processor(noise.pil, return_tensor="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                generated_ids= self.blip2_model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(generated_text)
            noise.caption = generated_text

