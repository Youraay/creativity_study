from PIL import Image as PILImage
import numpy as np
import statistics
import torch
from utils import ModelManager

class PromptAlignment:
    def __init__(self):
        self.model_manager = self.model_manager = ModelManager()

    def score_with_clip(self, image: PILImage.Image, prompt: str) -> float:
        

        return 0

    def score_with_blip2_qna(self, image: PILImage.Image, prompt: str) -> float:
       
        return 0