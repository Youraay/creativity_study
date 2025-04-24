from PIL import Image as PILImage
import os
import random
from data_model import Seed, Prompt, Image
from utils.model_manager import ModelManager
import torch
import config
class ImageGeneration(): 

    def __init__(self):
        
        self.model_manager = ModelManager()
        
        
        self.n_steps = config.N_STEPS
        self.high_noise_frac = config.HIGH_NOISE_FRAC
        self.guidance_scale = config.GUIDANCE_SCALE
        self.negative_prompt = config.NEGATIVE_PROMPT
        self.use_half_precision = config.USE_HALF_PRECISION
        print(f"\nImageGeneration initialized with: \n n_steps={self.n_steps}\n high_noise_frac={self.high_noise_frac}\n guidance_scale={self.guidance_scale}\n negative_prompt='{self.negative_prompt}'")

    def generate_img(self, prompt: Prompt, seed: Seed):
        
        
        base_model = self.model_manager.get_sdxl_base()
        
        if self.use_half_precision:
            base_model.to(torch.float16)
        
        image = base_model(
            prompt=prompt.get_prompt(),
            seed=seed.get_seed(),
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        return image


    def generate_quick_image(self, prompt: Prompt, seed: Seed, steps=50) -> PILImage:
        
        base_model = self.model_manager.get_sdxl_base()
        
        
        seed_value = seed.get_seed() if hasattr(seed, 'get_seed') else seed["seed"]
        
        
        image = base_model(
            prompt=prompt.get_prompt(),
            seed=seed_value,
            num_inference_steps=steps,
            guidance_scale=self.guidance_scale,  
            output_type="pil", 
        ).images[0]

        return image

    def refine_img(self, prompt: Prompt, seed : Seed, steps=0, image=None) -> PILImage:
        
        refiner_model = self.model_manager.get_sdxl_refiner()
        
        if steps == 0: 
            steps = self.n_steps
            
        refined_image = refiner_model(
            prompt=prompt.get_prompt(),
            seed=seed.get_seed(),
            num_inference_steps=steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]
        
        return refined_image

    def save_img(self, pil_img: PILImage, prompt: Prompt, seed: Seed, refined: bool, db_client=None):
        
        image = Image(
        seed=seed,
        prompt=prompt,
        pil_image=pil_img,
        generation_steps=self.n_steps,
        guidance_scale=self.guidance_scale,
        use_refiner=refined,
        db_client=db_client
        )
        
        
        if self.negative_prompt:
            image.notes = f"Negative prompt: {self.negative_prompt}"
        
        image.save_to_db()
        return image
        
    
    def generate_and_refine(self, prompt: Prompt, seed) -> PILImage:
        
        latents = self.generate_img(prompt, seed)
        return self.refine_img(prompt, seed, image=latents)
    

    def set_steps(self, steps: int):
        self.n_steps = steps
        return self

    def set_high_noise_frac(self, frac: float):
        if 0.0 <= frac <= 1.0:
            self.high_noise_frac = frac
        else:
            raise ValueError("high_noise_frac muss zwischen 0.0 und 1.0 liegen")
        return self

    def set_guidance_scale(self, scale: float):
        self.guidance_scale = scale
        return self

    def set_negative_prompt(self, negative_prompt: str):
        
        self.negative_prompt = negative_prompt
        return self

    def reset_to_defaults(self):
        self.n_steps = 50
        self.high_noise_frac = 0.8
        self.guidance_scale = 0
        self.negative_prompt = ""
        return self