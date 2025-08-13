
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
import pandas as pd 

from .abc_evaluator import Evaluator
from custom_types import Noise
from models import ModelManager


def _load_matrix(path:str) -> torch.Tensor:  

    t=torch.load(path)

    t.ndim


class KNNMEansEvaluator(Evaluator[Noise]):

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