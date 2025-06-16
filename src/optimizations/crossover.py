# mutators.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic
import torch
from ..custom_types import Latents, Argument, Noise

class Crossover(Generic[Argument], ABC):

    @abstractmethod
    def crossover(self, argument1: Argument, argument2: Argument, prompt:str, generation:int) -> Argument:
        
        raise NotImplementedError("Method is not implementet yet")
    

def arithmetic_crossover(
        noise1: Latents,
        noise2: Latents,
        weight: float = 0.5,
        proportions: float = 1.0
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

    return flat_child

def uniform_crossover(
        noise1: Latents,
        noise2: Latents,
        swap_rate: float = 0.5
) -> Latents:
    assert noise1.shape == noise2.shape, "Noises must have the same size for crossover."

    crossover_mask = torch.rand(noise1.shape, device=noise1.device) < swap_rate
    child = torch.where(crossover_mask, noise1, noise2)

    return child

@dataclass
class ArithmeticCrossover(Crossover[Noise]):

    weight : float =0.5
    proportions : float = 1.0

    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int) -> Noise:

        return Noise(prompt=prompt,
            arithmetic_crossover(
                argument1.latents,
                argument2.latents,
                self.weight,
                self.proportions, 
            ),
            first_appearance=generation,
            last_appearance=generation
        )
    
@dataclass
class UniformCrossover(Crossover[Noise]):

    swap_rate : float

    def crossover(self, argument1: Noise, argument2: Noise, prompt:str, generation:int):

        return Noise(prompt=prompt,
                     uniform_crossover(
                        argument1.latents,
                        argument2.latents,
                        self.swap_rate
                    ), 
                    first_appearance= generation,
                    last_appearance=generation)