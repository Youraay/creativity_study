import torch
import numpy as np
from typing import List
from PIL import Image as PILImage
from utils import ModelManager, PromptAlignment, EmbeddingExtractor
from data_model import Seed, Prompt, Image
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from skimage.restoration import estimate_sigma
import statistics
from scipy import ndimage
import config
class ImageEvaluation:
    def __init__(self):
        self.model_manager = ModelManager()
        self.alignment = PromptAlignment()
        
        

    

    def calculate_coherence_score(self, image: PILImage.Image, prompt_text: str) -> float:
        """
        
        The score is normalized to a range of 0 to 1.
        """
        if config.USE_BLIP2_QNA:
            return self.alignment.score_with_blip2_qna(image, prompt_text)
        else:
            return self.alignment.score_with_clip(image, prompt_text)

    def calculate_outlier_score(self, target_embedding: np.ndarray, mean_embedding: np.ndarray) -> float:
    
        """
        Theo outlier score is calculated by comparing the image embedding with the mean_embedding of the prompt and configuration.
        The score is calculated using cosine similarity, where a lower score indicates a more outlier image.
        The cosin similarity normalized to a range of 0 to 1. (The score is reversed to indicate outlierness)
        The score is then transformed using a sigmoid function to make it more interpretable.
        """

        target_embedding = np.squeeze(target_embedding)
        mean_embedding = np.squeeze(mean_embedding)

        cos_img = cosine_similarity([target_embedding], [mean_embedding])[0][0]
        score = (1.0 - cos_img) / 2.0
        
        #sigmoid transformation
        score = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * (score - 0.5)))

        return score
    
    def calculate_quality_score(self, image: PILImage.Image) -> float:
        """
        The score is normalized to a range of 0 to 1.
        TODO: Use BRISQUE or NIQE 
        """
       
        return 0