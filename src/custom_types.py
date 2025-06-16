# types.py
from typing import TypeVar, Optional, List
import numpy as np
import PIL.Image
import torch
from dataclasses import dataclass, field
import PIL
import random
type Device = str
type Latents = torch.Tensor
Argument = TypeVar('Argument')
Argument2 = TypeVar('Argument2')
Result = TypeVar('Result')
type Fitness = float
type Evaluation = float

@dataclass()
class Noise():
    prompt: str
    noise_embeddings: Latents
    first_appearance: int
    last_appearance: int
    generator: Optional[Latents] = None
    id: int = random.randint(1,2**32)
    blip_embeddings: Optional[Latents] = None
    clip_embeddings: Optional[Latents] = None
    image_embs: Optional[Latents] = None
    parent_embs: Optional[List['Noise']] = field(default_factory=list)
    pil: Optional[PIL.Image.Image] = None 
    fitness: float = 0
    scores : dict[str,float] = field(default_factory=dict)
    image_caption = ""
    seed: int = 0
    

    @classmethod
    def from_seed(cls, 
                  seed: int,
                  prompt:str,
                  id: int,
                  generation: int,
                  batch_size : int = 1,
                  num_channels_latents  : int =4 ,
                  vae_scale_factor : int = 8,
                  init_noise_sigma : float = 1.0,
                  height: int = 1024,
                  width: int = 1024,
                  device : Device = 'cuda',
                  dtype = torch.float16):
        
        assert seed > 0, "Seed must be grater than 0"
        generator = torch.Generator(device).manual_seed(seed)

        latents_shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor, 
            width // vae_scale_factor
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
        # latents = latents * init_noise_sigma
        
        return cls(noise_embeddings=latents, prompt=prompt, id=id, seed=seed, first_appearance = generation, last_appearance= generation, generator=generator)
    

if __name__ == "__main__":
    device ="cuda" if torch.cuda.is_available() else "cpu"
    noise = Noise.from_seed(
        seed=42,
        prompt='dog',
        id=1,
        generation=1,
        device=device
    )

