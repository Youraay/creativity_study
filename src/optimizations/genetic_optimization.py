import PIL.Image
from ..custom_types import Device, Noise, Latents
from typing import List, Tuple
from .evaluators import Evaluator
from .mutators import Mutator
from .crossover import Crossover
from .selector import Selector
from diffusers import StableDiffusionXLPipeline
import random
import torch
import copy
from datetime import datetime 
import os
import PIL
class GeneticOptimization():

    def __init__ (self,
                  generations: int, 
                  population_size: int,
                  prompt: str,
                  image_pipeline : StableDiffusionXLPipeline, 
                  selector: Selector[float],
                  evaluators: List[Evaluator[Noise]],
                  evaluation_weights: List[float],
                  mutator : Mutator[Latents], 
                  crossover_function : Crossover[Noise],
                  base_population: List[Noise] | None = None, 
                  device: Device = "cuda", 
                  initial_mutation_rate: float = 0.1,
                  crossover_rate: float = 0.1,
                  elitism_count: int = 0,
                  strict_osga: bool = False, 
                  random_seed : int = 42
                  ):
        
        assert generations > 0, "Number of Generations must be greater than 0"
        assert population_size > 0, "Population Size must be greater than 0"
        if base_population:
            assert len(base_population) <= population_size, "Base Population must not exceed population size"

        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.generations = generations
        self.completed_generations = 0
        self.prompt = prompt

        self.pipe = image_pipeline
        
        self.population_size = population_size
        self.population = base_population if base_population else []

        self.evaluators = evaluators
        self.evaluation_weights = evaluation_weights

        self.selector = selector

        self.mutator = mutator
        self.mutation_rate = initial_mutation_rate
        self.crossover_function = crossover_function
        self.crossover_rate = crossover_rate
        
        self.elitism_count = elitism_count

        self.strict_osga = strict_osga

        self.device = device

    def save_images(self) -> None:
        """
        Saves generated image to export folder for visualisations
        """
        print(f"save images for generation {self.completed_generations}")
        for i, noise in enumerate(self.population):
            filename = f"image_{self.completed_generations:02d}_{i:03d}_{noise.fitness:4f}.png"
            path = os.path.join(self.output_path, filename)
            noise.pil.save(path)
        print(f"{self.population_size} images saved")

    def create_output_folder(self) -> None:
        """
        Inizialize output folder for this experiment
        """
        safe_prompt = self.prompt.replace(" ", "_")
        self.timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_prompt}_{self.timestemp}"
        self.output_path = os.path.join("outputs", folder_name)
        os.makedirs(self.output_path, exist_ok=True)

    def initialize_population(self) -> None:
        """
        Inizialize the start population.
        If a set of noises is given the population will be filled with the the set of noises and random 
        """

        if self.population:
            missing = self.population_size - len(self.population)
            print(f"population is present: {missing} will be generated")
        else:
            missing = self.population_size
            print(f"all Noises will be generated: {missing}")

        if missing > 0:
            num_channel_latents = self.pipe.unet.config.in_channels
            vae_scale_factor = self.pipe.vae_scale_factor
            new_population = [Noise.from_seed(random.randint(1, 2**32), id=id, num_channels_latents=num_channel_latents, vae_scale_factor=vae_scale_factor) for id in range(missing)]  
            self.population += new_population
            print(f"Base Population with {len(self.population)} Candidates")

        for noise in self.population:
            
            noise.pil = self.generate_image(noise.latents)
            
            noise.fitness, noise.scores = self.calculate_fitness(noise)
        print("Base Population created")

    def calculate_fitness(self, noise: Noise) -> Tuple[float, List[float]]:
        """
        Will calculat all evaluation scores and combine them to a weighted sum - the fitness score

        args:
        noise: Noise the Candidate for the evaluation

        return:
        fitness: The weighted sum of the evaluation scores for the genetic selection process
        evaluations: for further visualisations and analytics
        """
        
        evaluations: List[float] = [evaluator.evaluate(noise, prompt=self.prompt) for evaluator in self.evaluators]
        
        fitness: float = sum(score * weight for score,  weight in zip(evaluations, self.evaluation_weights))

        return fitness, evaluations

    def generate_image(self, latents : Latents) -> PIL.Image.Image:

        """
        Generates an Image from a Latetns with the given Pipeline

        args:
        latents: Latents = the initial noise for the diffusion pipeline

        return:
        image: PIL = the pixel representation of the generated image. Used for CLIP/BLIP Evaluators

        """

        return self.pipe(
                    prompt=self.prompt,
                    latents = latents,
                    output_type="pil",
                    num_inference_steps=4,
                    guidance_scale =0.0
                    ).images[0]
    
    def best_solution(self) -> Noise:
        """
        return:
        noise: Noise = the noise with the highest fitness of this generation
        """

        return max(self.population, key=lambda noise: noise.fitness)

    def perform_generation(self) -> None:
        """
        simulates one generation of latents
        if set selects a group of elits, with will be passed to the next generation.
        Further the next generation is fild with childs of the current population. 
        """
        print(self.best_solution())
        self.completed_generations += 1
        elites : List[Noise] = sorted(self.population, key= lambda noise: noise.fitness , reverse=True)[:self.elitism_count] if self.elitism_count else []

        child_population = elites.copy()

        while len(child_population) < self.population_size:
            print (f"the population has a size of {len(self.population)}")
            parent1: Noise = self.selector.select(self.population)

            parent2 : Noise | None = None
            child: Noise | None= None
            crossed = False
            mutated = False
            if random.random() <= self.crossover_rate:
                parent2 = self.selector.select(self.population)
                child = self.crossover_function.crossover(parent1, parent2)
                crossed = True
            
            else:
                child= copy.deepcopy(parent1)

            if random.random() <= self.mutation_rate:
                child.latents = self.mutator.mutate(child.latents)
                mutated = True

            if crossed or mutated:
                child.pil = self.generate_image(child.latents)
                child.fitness, child.scores = self.calculate_fitness(child)

                max_parent_fitness = max(parent1.fitness, parent2.fitness) if parent2 else parent1.fitness
                if self.strict_osga:
                    if child.fitness > max_parent_fitness:
                        child_population.append(child)
                else:
                    child_population.append(child)
            else:
                child_population.append(child)
        
        
        self.population = child_population        

    def run(self) -> Noise:
        """
        Runs the Simulation and returns the best Latents
        """
        self.initialize_population()
        self.create_output_folder()

        for generation in range(self.generations -1):
            self.save_images()
            self.perform_generation()
        self.save_images()
        return self.best_solution()
