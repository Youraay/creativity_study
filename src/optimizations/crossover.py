# mutators.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic
import torch
from ..custom_types import Latents, Argument, Noise

@dataclass
class Crossover(Generic[Argument], ABC):


    @abstractmethod
    def crossover(self, argument1: Argument, argument2: Argument, prompt:str, generation:int) -> Argument:
        
        raise NotImplementedError("Method is not implementet yet")
    
    def __str__(self):
        return self.__repr__()
    
def arithmetic_crossover(
        noise1: Latents,
        noise2: Latents,
        weight: float = 0.5,
        proportions: float = 1.0,
        normalize: bool = True
) -> Latents:
    
    assert noise1.shape == noise2.shape, "Noises must have the same size for crossover."
    assert 0 < proportions <=1, "0 <= proportion <= 1 must be fulfilled"

    device = noise1.device
    noise2 = noise2.to(device)
    
    if proportions == 1.0:
        return weight * noise1 + (1-weight) * noise2
    
    child = noise1.clone()
    num_elements = noise1.numel()
    num_crossover = int(num_elements *proportions)

    crossover_positions = torch.randperm(num_elements, device=device)[:num_crossover]

    flat_child = child.view(-1)
    flat_noise1 = noise1.view(-1)
    flat_noise2 = noise2.view(-1)

    flat_child[crossover_positions] =  weight * flat_noise1 + (1-weight) * flat_noise2
    child = flat_child.view_as(noise1)
    if normalize:
        # Z-Score-Normalisierung: Mittelwert 0, Standardabweichung 1
        mean = child.mean()
        std = child.std()
        if std > 1e-6:
            child = (child - mean) / std
    return child

def uniform_crossover(
        noise1: Latents,
        noise2: Latents,
        swap_rate: float = 0.5
) -> Latents:
    assert noise1.shape == noise2.shape, "Noises must have the same size for crossover."

    crossover_mask = torch.rand(noise1.shape, device=noise1.device) < swap_rate
    child = torch.where(crossover_mask, noise1, noise2)

    return child

def slerp_crossover(noise1: Latents,
          noise2: Latents,
          t:float
) -> Latents:
    n1_flat = noise1.view(-1)
    n2_flat = noise2.view(-1)

    n1_norm = n1_flat / torch.norm(n1_flat)
    n2_norm = n2_flat / torch.norm(n2_flat)

    dot = torch.clamp(torch.dot(n1_norm, n2_norm), -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.item() < 1e-5:
        return (1.0 - t) * noise1 + t * noise1
    
    sin_theta = torch.sin(theta)
    factor_n1 = torch.sin((1-t) * theta) /sin_theta
    factor_n2 = torch.sin(t * theta) / sin_theta

    return factor_n1 * noise1 + factor_n2* noise2

@dataclass
class ArithmeticCrossover(Crossover[Noise]):

    weight : float =0.5
    proportions : float = 1.0
    normalize: bool = True
    name: str = "ArithmeticCrossover"

    def __repr__(self):
        return (
            f"ArithmeticCrossover(weight={self.weight}, "
            f"proportions={self.proportions}, normalize={self.normalize})"
        )
    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int) -> Noise:

        return Noise(prompt=prompt,
            noise_embeddings=arithmetic_crossover(
                argument1.noise_embeddings,
                argument2.noise_embeddings,
                self.weight,
                self.proportions, 
            ),
            first_appearance=generation,
            last_appearance=generation
        )
    
@dataclass
class UniformCrossover(Crossover[Noise]):
    swap_rate : float
    name: str = "UniformCrossover"

    def __repr__(self):
        return f"UniformCrossover(swap_rate={self.swap_rate})"
    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int):

        return Noise(prompt=prompt,
                     noise_embeddings=uniform_crossover(
                        argument1.noise_embeddings,
                        argument2.noise_embeddings,
                        self.swap_rate
                    ), 
                    first_appearance= generation,
                    last_appearance=generation)
    

def quartered_crossover(n1: torch.Tensor,
                        n2: torch.Tensor,
                        blend: bool = False,
                        width: int = 8) -> torch.Tensor:
    if n1.shape != n2.shape:
        raise ValueError("Shapes differ")

    L = n1.shape[0]
    q = L // 4

    # --- harter Schnitt ----------------------------------------------------
    if not blend or width == 0:
        child = torch.cat((n1[:q],
                           n2[q:2*q],
                           n1[2*q:3*q],          # <- hier wieder n1
                           n2[3*q:]), dim=0).clone()  # clone() = garantiert neues Memory
        return child

    # --- weiches Blending --------------------------------------------------
    def blend_zone(a, b, w):
        α = torch.linspace(1, 0, w, device=a.device).unsqueeze(1)
        return α * a + (1 - α) * b

    w2 = width // 2
    a1 = n1[:q - w2]
    ab = blend_zone(n1[q - w2:q + w2], n2[q - w2:q + w2], width)
    b1 = n2[q + w2:2*q - w2]
    bc = blend_zone(n2[2*q - w2:2*q + w2], n1[2*q - w2:2*q + w2], width)  # <-- gefixt
    c1 = n1[2*q + w2:3*q - w2]
    cd = blend_zone(n1[3*q - w2:3*q + w2], n2[3*q - w2:3*q + w2], width)
    d1 = n2[3*q + w2:]

    return torch.cat((a1, ab, b1, bc, c1, cd, d1), dim=0).clone()

@dataclass
class QuarteredCrossover(Crossover[Noise]):
    blend: bool = False
    blend_range: int = 8

    name: str = "QuarteredCrossover"
    def __repr__(self):
        return f"QuarteredCrossover(blend={self.blend}, blend_range={self.blend_range})"
    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int):

        return Noise(prompt=prompt,
                     noise_embeddings=quartered_crossover(
                        argument1.noise_embeddings,
                        argument2.noise_embeddings,
                        self.blend,
                        self.blend_range
                    ), 
                    first_appearance= generation,
                    last_appearance= generation)
    
@dataclass
class SlerpCrossover(Crossover[Noise]):

    t: float = 0.5
    name: str = "SlerpCrossover"
    
    def __repr__(self):
        return f"SlerpCrossover(t={self.t})"
    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int) -> Noise:

        return Noise(prompt=prompt,
            noise_embeddings=slerp_crossover(
                argument1.noise_embeddings,
                argument2.noise_embeddings,
                self.t
            ),
            first_appearance=generation,
            last_appearance=generation
        )