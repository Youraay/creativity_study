import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import config
from data_model import Image

class SeedEvaluation:
    def __init__(self, db_client=None):
        
        self.db_client = db_client

    
    def calculate_creativity_index(self,seed_id: str,prompt_id: str,) -> float:
        """
        Calculates the creativity index of a seed based on the weighted sum of the outlier scores, coherence score, and quality score.
        TODO: Better solution?
        """
        
        image: Image = Image.from_dict(self.db_client.get_image_by_seed_and_prompt(seed_id, prompt_id), self.db_client)
        if image is None or image.blip_embedding is None:
            return 0.0

        

        
        local_outlier = image.local_outlier_score or 0.0
        global_outlier = image.global_outlier_score or 0.0

        
        coherence = image.coherence_score or 0.0
        quality = image.quality_score or 0.0

        
        creativity_score = (
            config.GLOABL_OUTLIER_WEIGHT * global_outlier +
            config.LOCAL_OUTLIER_WEIGHT * local_outlier +
            config.COHERENCE_WEIGHT * coherence +
            config.QUALITY_WEIGHT * quality
        )

        return float(creativity_score)