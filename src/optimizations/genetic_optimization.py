import PIL.Image
from ..custom_types import Device, Noise, Latents
from ..models.manager import ModelManager
from typing import List, Tuple
from .evaluators import Evaluator, MaxLocalMeanDivergenceEvaluator, BLIP2ImageCaption
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
import json
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import logging

def pool_image_embs(tensor: torch.Tensor) -> torch.Tensor:
    # Erwartet shape [batch, seq_len, dim] oder [seq_len, dim] oder [dim]
    # Zielt auf [dim] ab
    # 1) Bei Batch > 1: batch-mean
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=0)     # -> [seq_len, dim]
    # 2) Bei Seq-Len > 1: token-mean
    if tensor.dim() == 2:
        tensor = tensor.mean(dim=0)     # -> [dim]
    # Jetzt sollte tensor.dim()==1 sein
    return tensor.to("cuda")

def latents_to_rgb(latents):
    """
    Convert the SDXL latents (4 channels: Luminance, Cyan, Red, Pattern Structure) to RGB tensors (3 channels).
    Explanation: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#the-4-channels-of-the-sdxl-latents
    """
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    
    # Handle batch dimension - squeeze if batch size is 1
    if rgb_tensor.dim() == 4 and rgb_tensor.shape[0] == 1:
        rgb_tensor = rgb_tensor.squeeze(0)  # Remove batch dimension: [1, 3, H, W] -> [3, H, W]
    
    # Ensure we have [C, H, W] format before transpose to [H, W, C]
    if rgb_tensor.dim() == 3:
        image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected tensor shape: {rgb_tensor.shape}. Expected [C, H, W] or [1, C, H, W]")

    return PIL.Image.fromarray(image_array)

class GeneticOptimization():

    def __init__ (self,
                  generations: int, 
                  population_size: int,
                  prompt: str,
                  image_pipeline : StableDiffusionXLPipeline, 
                  job_id: str,
                  selector: Selector[float],
                  evaluators: List[Evaluator[Noise]],
                  evaluation_weights: List[float],
                  mutator : Mutator[Latents], 
                  crossover_function : Crossover[Noise],
                  ts:str,
                  base_population: List[Noise] | None = None, 
                  device: Device = "cuda", 
                  initial_mutation_rate: float = 0.1,
                  crossover_rate: float = 0.8,
                  elitism_count: int = 0,
                  strict_osga: bool = False, 
                  random_seed : int = 42,
                  num_steps : int = 50,
                  guidance : float = 7.0,
                  sigma_scaling : bool = False,
                  ):
        
        assert generations > 0, "Number of Generations must be greater than 0"
        assert population_size > 0, "Population Size must be greater than 0"
        if base_population:
            assert len(base_population) <= population_size, "Base Population must not exceed population size"

        random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.next_id = 1
        self.job_id = job_id
        model_manager = ModelManager()
        self.blip2_model, self.blip_processor = model_manager.load_blip2()

        self.num_steps = num_steps
        self.guidance = guidance

        self.generations = generations
        self.completed_generations = 0
        self.prompt = prompt

        self.pipe = image_pipeline
        self.pipe.set_progress_bar_config(disable=True)
        
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
        self.generation_map: Dict[int, List[Noise]] = defaultdict(list)
        self.timestemp = ts

        self.caption_generator = BLIP2ImageCaption()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def generate_blip_embedding(self, noise: Noise):
        inputs = self.blip_processor(images=noise.pil, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output =self.blip2_model.get_image_features(**inputs)
            raw = output.last_hidden_state.cpu()

            noise.image_embs = pool_image_embs(raw)
        

    def save_images(self) -> None:
        """
        Saves generated image to export folder for visualisations
        """
        # self.logger.info(f"save images for generation {self.completed_generations}")
        for i, noise in enumerate(self.population):
            filename = f"images/image_{self.completed_generations:02d}_{noise.id:03d}_{noise.fitness:4f}.png"
            path = os.path.join(self.output_path, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)  
            if noise.pil is not None:
                noise.pil.save(path)
            else:
                self.logger.warning(f"Warning: No PIL image found for noise {noise.id}")
        self.logger.info(f"{self.population_size} images saved")

    def save_noises(self) -> None:
        """
        Saves generated image to export folder for visualisations
        """
        # self.logger.info(f"save images for generation {self.completed_generations}")
        for i, noise in enumerate(self.population):
            filename = f"noises/noise_{self.completed_generations:02d}_{noise.id:03d}_{noise.fitness:4f}.png"
            path = os.path.join(self.output_path, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)  
            img = latents_to_rgb(noise.noise_embeddings)
            img.save(path)  
        self.logger.info(f"{self.population_size} noises saved")

    def create_output_folder(self) -> None:
        """
        Inizialize output folder for this experiment
        """
        safe_prompt = self.prompt.replace(" ", "_")
        evs = ""
        for i in self.evaluators:
            evs += i.name
            evs += "_"
        folder_name = f"{self.job_id}_{safe_prompt}_{self.crossover_function.name}_{self.selector.name}_{evs}"
        self.output_path = os.path.join("/scratch/dldevel/sinziri/creativity_study/files", folder_name)
        os.makedirs(self.output_path, exist_ok=True)
        self.logger.info(f"Directory created: {self.output_path}")

    def save_config(self) -> None:
        """
        Speichert die wichtigsten Konfigurationsparameter in einer JSON-Datei zur Nachvollziehbarkeit des Experiments.
        Muss nach Erstellung des Output-Ordners aufgerufen werden.
        """
        config = {
            "generations": self.generations,
            "population_size": self.population_size,
            "prompt": self.prompt,
            "image_pipeline": self.pipe._name_or_path,
            "selector": type(self.selector).__name__,
            "mutator": type(self.mutator).__name__,
            "crossover_function": type(self.crossover_function).__name__,
            "device": self.device,
            "initial_mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elitism_count,
            "strict_osga": self.strict_osga,
            "evaluation_weights": self.evaluation_weights,
            "evaluators": [type(e).__name__ for e in self.evaluators],
            "random_seed": torch.initial_seed(),
            "timestamp": self.timestemp,
            "output_path": self.output_path
        }

        config_path = os.path.join(self.output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def log_generation_result(self) -> None:
        """
        Appends best fitness and optional metadata of the current generation
        to the configuration JSON.
        """
        log_entry = {
            "generation": self.completed_generations,
            "best_fitness": self.best_solution().fitness,
            "timestamp": datetime.now().isoformat()
        }

        
        config_path = os.path.join(self.output_path, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        
        if "generation_log" not in config_data:
            config_data["generation_log"] = []

        
        config_data["generation_log"].append(log_entry)

        
        with open(config_path, "w") as file:
            json.dump(config_data, file, indent=4)

    def log_error(self, error: Exception) -> None:
        """
        Protokolliert einen Fehler in die Konfigurationsdatei.
        """
        config_path = os.path.join(self.output_path, "config.json")
        
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        if "errors" not in config_data:
            config_data["errors"] = []

        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "generation": self.completed_generations
        }

        config_data["errors"].append(error_entry)

        with open(config_path, "w") as file:
            json.dump(config_data, file, indent=4)
            
    def initialize_population(self) -> None:
        """
        Inizialize the start population.
        If a set of noises is given the population will be filled with the the set of noises and random 
        """

        if self.population:
            missing = self.population_size - len(self.population)
            self.logger.info(f"population is present: {missing} will be generated")
        else:
            missing = self.population_size
            self.logger.info(f"all Noises will be generated: {missing}")

        if missing > 0:
            num_channel_latents = self.pipe.unet.config.in_channels
            vae_scale_factor = self.pipe.vae_scale_factor
            new_population = []  
            for id in range(missing):
                new_population.append( Noise.from_seed(random.randint(1, 2**32), prompt=self.prompt,id=self.next_id, generation=0, num_channels_latents=num_channel_latents, vae_scale_factor=vae_scale_factor))
                self.next_id += 1
            self.population += new_population
            self.logger.info(f"Base Population with {len(self.population)} Candidates")

        count_img_per_generation=1
        for noise in self.population:
            
            noise.pil = self.generate_image(noise.noise_embeddings)
            self.logger.debug(f"Image {count_img_per_generation} of {self.population_size} for Generation {self.completed_generations}")
            count_img_per_generation +=1

        for child in self.population:
                self.generate_blip_embedding(child)

        child_blip_embeddings = [child.image_embs for child in self.population if child.image_embs is not None]
        

        if len(child_blip_embeddings) == 0:
            self.logger.warning("Warning: No valid BLIP embeddings found in child population")
            # Skip local divergence evaluation but continue with other processing
            local_mean_embeddings = None
        else:
            child_embeddings_matrix = torch.stack(child_blip_embeddings)
            local_mean_embeddings = torch.mean(child_embeddings_matrix, axis=0)

        for child in self.population:
            for evaluator in self.evaluators:
                evaluator.evaluate(child, prompt=self.prompt, local_mean_embeddings=local_mean_embeddings)

            self.calculate_fitness(child)
        self.logger.info("Base Population created")

    def calculate_fitness(self, noise: Noise):
        """
        Will calculat all evaluation scores and combine them to a weighted sum - the fitness score

        args:
        noise: Noise the Candidate for the evaluation

        return:
        fitness: The weighted sum of the evaluation scores for the genetic selection process
        evaluations: for further visualisations and analytics
        """
        
        if not noise.scores:
            self.logger.warning(f"Warning: No scores found for noise {noise.id}, skipping fitness calculation")
            noise.fitness = 0.0
            return
            
        if len(noise.scores) != len(self.evaluation_weights):
            self.logger.warning(f"Warning: Score count ({len(noise.scores)}) doesn't match weight count ({len(self.evaluation_weights)}) for noise {noise.id}")
            noise.fitness = 0.0
            return
        
        fitness: float = sum(score * weight for score,  weight in zip(noise.scores.values(), self.evaluation_weights))
        noise.fitness = fitness
        # Am Ende von calculate_fitness hinzufügen:
        self.logger.debug(f"Scores: {noise.scores}")
        self.logger.debug(f"Weighted calculation: {list(zip(noise.scores.values(), self.evaluation_weights))}")
        self.logger.debug(f"Fitness for {noise.id}: {fitness}, Scores: {noise.scores}")
        

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
                    num_inference_steps=self.num_steps,
                    guidance_scale =self.guidance,
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
        self.completed_generations += 1
        elites : List[Noise] = sorted(self.population, key= lambda noise: noise.fitness , reverse=True)[:self.elitism_count] if self.elitism_count else []
        elite_ids = {elite.id for elite in elites}
        child_population = [copy.deepcopy(e) for e in elites]

        while len(child_population) < self.population_size:
            parent1: Noise = self.selector.select(self.population)

            parent2 : Optional[Noise] = None
            child: Optional[Noise] = None
            crossed = False
            mutated = False
            
            
            if random.random() <= self.crossover_rate:
                parent2 = self.selector.select(self.population)
                while parent2 is parent1:
                    parent2 = self.selector.select(self.population)
                if parent2.fitness > parent1.fitness:
                    parent1, parent2 = parent2, parent1
                child = self.crossover_function.crossover(parent1, parent2, self.prompt, self.completed_generations)
                child.parent_embs = [parent1, parent2]
                child.parents =[parent1.id, parent2.id]
                child.id = self.next_id
                self.next_id += 1
                child.crossed = True
            else:
                
                child = copy.deepcopy(parent1)
                child.last_appearance = self.completed_generations
                child.id = self.next_id
                self.next_id += 1
                child.parent_embs = [parent1]
                child.parents =[parent1.id]
                child.crossed = False

            
            if random.random() <= self.mutation_rate:
                child.noise_embeddings = self.mutator.mutate(child.noise_embeddings)
                child.mutated = True
            else:
                child.mutated = False
            
            child_population.append(child)

        

        for child in child_population:
            # Generate images for all children that were modified (crossed, mutated, or newly created)
            # Skip only elites that were copied directly without modification
            if child.crossed or child.mutated or child.id not in elite_ids:
                child.pil = self.generate_image(child.noise_embeddings)

        self.population.clear()

        for child in child_population:
            
            if child.crossed or child.mutated or child.id not in elite_ids:
                self.generate_blip_embedding(child)
                
        
        child_blip_embeddings = [child.image_embs for child in child_population if child.image_embs is not None]
        if len(child_blip_embeddings) == 0:
            self.logger.warning("Warning: No valid BLIP embeddings found in child population")
            
            local_mean_embeddings = None
        else:
            child_embeddings_matrix = torch.stack(child_blip_embeddings)
            local_mean_embeddings = torch.mean(child_embeddings_matrix, axis=0)


        for child in child_population:
            
            if child.crossed or child.mutated or child.id not in elite_ids:
                for evaluator in self.evaluators:
                    evaluator.evaluate(child, prompt=self.prompt, local_mean_embeddings=local_mean_embeddings)

        for child in child_population:
            
            self.calculate_fitness(child)
            
            
            if child.parent_embs and len(child.parent_embs) > 0:
                max_parent_fitness = max([p.fitness for p in child.parent_embs]) # type: ignore
            else:
                max_parent_fitness = 0.0  
            
            if self.strict_osga:
                if child.fitness > max_parent_fitness:
                    self.population.append(child)
            else:
                self.population.append(child)
            
            self.logger.debug(f"Image {len(child_population)} of {self.population_size} for Generation {self.completed_generations}")
        
        self.generation_map[self.completed_generations] = copy.deepcopy(self.population)
       
    def run(self) -> Tuple[Dict[int, List[Noise]], str]:
        """
        Runs the Simulation and returns the best Latents
        """
        try:
            self.create_output_folder()
            self.save_config()
            self.initialize_population()
            
            self.save_images()
            self.save_noises()
            self.generation_map[self.completed_generations] = copy.deepcopy(self.population)
            for generation in range(self.generations):
                
                self.perform_generation()
                self.log_generation_result()
                self.save_images()
                self.save_noises()
            
                self.logger.info(f"Generation{self.completed_generations} of {self.generations} finished")

            return self.generation_map , self.output_path
        except Exception as e:
            self.log_error(e)
            raise
        
