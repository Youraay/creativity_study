from diffusers import DiffusionPipeline
import torch
from PIL import Image
import random
from data_model import Seed, Prompt
class ImageGeneration(): 

    def __init__ (self):


        # load both base & refiner
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.base.to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")


        self.n_steps = 40
        self.high_noise_frac = 0.8
        self.guidance_scale = 0
        self.negative_prompt = ""



    def generate_img(self, prompt: Prompt, seed: Seed):

        image = self.base(
            prompt=prompt.get_prompt(),
            seed=seed.get_seed(),
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        return image


    def refine_img(self, prompt: Prompt, seed: Seed) -> Image:

        image = self.refiner(
            prompt=prompt.get_prompt(),
            seed=seed.get_seed(),
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]
        

        return image
    
    def save_img(self, img: Image, prompt: Prompt, seed: Seed): 

        img.save("allee.png")