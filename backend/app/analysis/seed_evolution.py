import random
import numpy as np
import math
from typing import List, Dict, Tuple
from data_model import Seed, Prompt, Image, Experiment, Generation, DataFactory
from analysis.image_evaluation import ImageEvaluation
from analysis.seed_evaluation import SeedEvaluation
from utils import ImageGeneration, EmbeddingExtractor
import torch
import gc
import os
import config
class SeedEvolution:
    def __init__(self, db_client, image_generator: ImageGeneration, image_evaluator=None):
        
        self.db_client = db_client
        self.image_generator = image_generator
        
        self.image_evaluator = image_evaluator or ImageEvaluation()
        
        self.seed_evaluator = SeedEvaluation(db_client)
        self.embedding_extractor = EmbeddingExtractor()
        
        self.population_size = config.POPULATION_SIZE
        self.elite_size = config.ELITE_SIZE
        self.mutation_rate = config.MUTATION_RATE
        self.min_improvement = config.MIN_IMPROVEMENT
        self.max_generations = config.MAX_GENERATIONS
        self.patience = config.PATIENCE

         
        seed = config.RANDOM_SEED  
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"\nEvolutionsparameter:")
        print(f"Populationsgröße: {self.population_size} Seeds")
        print(f"Elite-Größe: {self.elite_size} Seeds")
        print(f"Mutationsrate: {self.mutation_rate:.2%}")
        print(f"Max. Generationen: {self.max_generations}")
        print(f"Abbruch nach {self.patience} Generationen ohne min. {self.min_improvement:.2%} Verbesserung")
        print("\n")
        
    def initialize_population(self, prompt: Prompt) -> List[Seed]:
        """
        Creats the first Population of seeds
        The seed population is created randomly 
        as in the evolution simulation. Images are generated and evaluated.
        For the evaluation the BLIP-Embeddings are used for outlier detection
        For the semantic coherence the CLIP-Model is used
        For quality the sharpeness, contrast and noise level are used.
        """
        
        
        factory = DataFactory(self.db_client)
        
        self.prompt = prompt
        population = []
        images: List[Image] = []

        print(f"Initializing population for prompt: {prompt.get_prompt()}")

        for _ in range(self.population_size):
            seed = factory.create_seed()
            print(f"Generating image for {prompt.get_prompt()}:{seed.get_seed()}...")
            
            image = self.image_generator.generate_quick_image(prompt, seed)
            img_obj = self.image_generator.save_img(image, prompt, seed, refined=False, db_client=self.db_client)
            images.append(img_obj)
            population.append(seed)

        print(f"Extracting embeddings for {len(images)} images...")
        valid_images = []
        for img in images:
            image_data = None
            try:
                with img.load_image() as pil_image:
                    image_data = pil_image.convert("RGB").copy()  
                    emb = self.embedding_extractor.extract_image_embedding(image_data)
            except Exception as e:
                print(f"Fehler beim Laden von {img.get_id()}: {e}")
                continue

            
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu()

            img.blip_embedding = emb
            img.save_to_db()
            valid_images.append(img)

        if not valid_images:
            raise RuntimeError("Keine gültigen Bilder mit BLIP-Embeddings in der Population.")

        print(f"Berechne lokale Mittelwerte für {len(valid_images)} valide Bilder...")
        all_embeddings = torch.stack([img.blip_embedding for img in valid_images])
        local_mean_embedding = all_embeddings.mean(dim=0).cpu().numpy()
        global_mean_embedding = prompt.get_mean_embedding()

        print(f"Berechne Scores für {len(valid_images)} Bilder...")
        for img in valid_images:
            img.coherence_score = self.image_evaluator.calculate_coherence_score(img.load_image(), prompt.get_prompt())
            img.global_outlier_score = self.image_evaluator.calculate_outlier_score( img.blip_embedding.cpu().numpy(), global_mean_embedding)
            img.local_outlier_score = self.image_evaluator.calculate_outlier_score( img.blip_embedding.cpu().numpy(), local_mean_embedding)
            img.quality_score = self.image_evaluator.calculate_quality_score(img.load_image())
            img.save_to_db()

        
            print(f"Scores: {img.get_id()} - Coherence: {img.coherence_score:.4f}, Global Outlier: {img.global_outlier_score:.4f}, Local Outlier: {img.local_outlier_score:.4f}, Quality: {img.quality_score:.4f}")
        
      
        prompt.clean_mean_embedding()
        images.clear()
        torch.cuda.empty_cache()
        gc.collect()

        return population
    def evaluate_fitness(self, population: List[Seed], prompt: Prompt) -> Dict[str, float]:
        """
        Calculates the fitnness of the population.
        the score is a combination of the (local, global) outlier score, coherence score and quality score.
        """

        fitness_scores = {}
        
        for seed in population:
            
            creativity_score = self.seed_evaluator.calculate_creativity_index(seed.get_id(), prompt.get_id())
            fitness_scores[seed.get_id()] = creativity_score
            
        return fitness_scores
    
    def select_parents(self, population: List[Seed], fitness_scores: Dict[str, float]) -> List[Seed]:
        """
        The First Group of Parents contains the elite seeds of the current generation, defined by a cut-off value.
        The Second Group is selected by a weighted random choice from the remaining seeds.
        A strong Seed has a chance to be parent of multiple children. 
        Using Softmax with temperature to give weaker seeds a chance to be selected.
        TODO: Select Elite with a randomized approach
        TODO: Implemenmt a Hall-of-Fame for good seeds along the generations
        TODO: Implement SUS 
        """

        sorted_population = sorted(
            population, 
            key=lambda seed: fitness_scores.get(seed.get_id(), 0.0),
            reverse=True
        )
        
        selected = sorted_population[:self.elite_size]
        
        
        selection_pool = sorted_population[self.elite_size:]
        
        num_to_select = self.population_size - self.elite_size

        if selection_pool and num_to_select > 0:
            
            fits = [fitness_scores.get(s.get_id(), 0.0) for s in selection_pool]
            
            # p∝ exp(f_i / T)
            # calculates the selection probability with temperatur 
            scores = [math.exp(f / config.TEMPERATURE) for f in fits]
            total = sum(scores)
            probs = [score / total for score in scores]

            
            additional = random.choices(selection_pool, weights=probs, k=num_to_select)
            selected.extend(additional)

        
        return selected
    
    def crossover(self, parent1: Seed, parent2: Seed) -> int:
        """
        Combine the Genoms of two parents to create a child.
        Randomly selects one of the parrents
        If no parent is selected the child is created by a two-point crossover.
        If the parents are not simular, a uniform crossover is used.
        """

        # convert to 32 bit room        
        parent_bit_1 = format(parent1.get_seed(), '032b')
        parent_bit_2 = format(parent2.get_seed(), '032b')

        
        # randomly one of the parents is living for the next generation
        if random.random() > config.CROSSOVER_RATE:  
            return parent1.get_seed() if random.random() < 0.5 else parent2.get_seed()

        
        sim = 1.0 - self.calculate_population_diversity([parent1, parent2])
        


        if sim > 0.8:
            pts = sorted(random.sample(range(1, 31), 2))
            # randomly combines chunks of the parents at 2 points
            child = parent_bit_1[:pts[0]] + parent_bit_2[pts[0]:pts[1]] + parent_bit_1[pts[1]:]
        else:
            # over the whole seed lengh bite by bite one of the bites is choosen
            child = ''.join(random.choice([bit1, bit2]) for bit1, bit2 in zip(parent_bit_1, parent_bit_2))
        return int(child, 2)
    
    def mutate(self, seed_value: int) -> int:
        """
        Simulates the mutation of a seed.
        If the parents are too similar, the mutation is more radical (creates an independent seed).
        With a healthy relationship between the parents, bitwise mutations are used.
        """

        
        seed_bits = list(format(seed_value, '032b'))
            
        for i in range(len(seed_bits)):
            if random.random() < self.mutation_rate:
                seed_bits[i] = '1' if seed_bits[i] == '0' else '0'
            
        return int(''.join(seed_bits), 2)
    
    def create_next_generation(self, parents: List[Seed]) -> List[Seed]:
        """
        Create the next generation of seeds by combining the fittest seeds of the current generation.
        The seed breeding is done by utilizing seed crossover and mutation
        """

        from data_model.data_factory import DataFactory
        self.factory = DataFactory(self.db_client)
        
        
        next_generation = parents[:self.elite_size]
        
        self._cached_elite_images = {
            seed.get_id(): Image.from_dict( self.db_client.get_image_by_seed_and_prompt(seed.get_id(), self.prompt.get_id()), self.db_client)
            for seed in next_generation
        }

        while len(next_generation) < self.population_size:
            
            parent1, parent2 = random.sample(parents, 2)
            child_value = self.crossover(parent1, parent2)
            child_value_m = self.mutate(child_value)
            child_seed = self.factory.create_seed(child_value_m)
            child_seed.set_parents(parent1.get_id(), parent2.get_id())
            if child_value != child_value_m:
                child_seed.set_mutated(True)
            next_generation.append(child_seed)
        
        return next_generation
    
    def calculate_population_diversity(self, population: List[Seed]) -> float:
        """
        Calculates the Hamming-Distance between all seeds in the population.
        (Bitwise diffrence of the population)
        Uses the Hamming-Distance to measure the diversity of the population.
        0.0 = all seeds are identical
        1.0 = all seeds are different
        """

        seed_values = [seed.get_seed() for seed in population]
        bit_strings = [format(val, '032b') for val in seed_values]
        
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(bit_strings)):
            for j in range(i+1, len(bit_strings)):
                
                distance = sum(b1 != b2 for b1, b2 in zip(bit_strings[i], bit_strings[j]))
                total_distance += distance
                comparisons += 1
        
        
        return total_distance / (comparisons * 32) if comparisons else 0.0
    
    def evolve(self, prompt: Prompt, experiment: Experiment) -> Tuple[List[Seed], Dict]:
        """
        The main function for the evolution of the seeds.
        Uses a adaptive evolution strategy to create a population of seeds.
        The evolution is based on the fitness of the seeds.
        The Mutation rate is adapted to the diversity of the population.
        The evolution is stopped if no significant improvement is detected for a certain number of generations.
        """
        
        self.experiment = experiment
        population = self.initialize_population(prompt)
        
        
        evolution_stats = {"best_fitness": [], "avg_fitness": [], "best_seeds": []}
        
        
        best_fitness_ever = 0.0
        avg_fitness_ever = 0.0
        generations_without_improvement_best = 0
        generations_without_improvement_avg = 0

        
        for generation_idx in range(self.max_generations):
            print(f"Generation {generation_idx+1}/{self.max_generations}")
            
            
            fitness_scores = self.evaluate_fitness(population, prompt)
            
            
            best_fitness = max(fitness_scores.values()) if fitness_scores else 0
            avg_fitness = sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0
            best_seed_id = max(fitness_scores.items(), key=lambda x: x[1])[0] if fitness_scores else None
            
            
            evolution_stats["best_fitness"].append(best_fitness)
            evolution_stats["avg_fitness"].append(avg_fitness)
            evolution_stats["best_seeds"].append(best_seed_id)
            
            print(f" Best fitness: {best_fitness:.4f}, Average: {avg_fitness:.4f}")
            
            diversity = self.calculate_population_diversity(population)
            if diversity < config.MIN_DIVERSITY:  
                print(f"Population diversity too low: {diversity:.4f}")
                self.mutation_rate = min(0.25, self.mutation_rate * 1.5) 
            elif diversity > config.MAX_DIVERSITY: 
                print(f"Population diversity too high: {diversity:.4f}")
                self.mutation_rate = max(0.01, self.mutation_rate * 0.8)
            


            if best_fitness - best_fitness_ever > self.min_improvement:
                best_fitness_ever = best_fitness
                generations_without_improvement_best = 0
                print(f"Significant best-fitness improvement: +{best_fitness - best_fitness_ever:.4f}")
            else:
                generations_without_improvement_best += 1
                print(f"No best-fitness improvement for {generations_without_improvement_best} generations")

            
            if avg_fitness - avg_fitness_ever > self.min_improvement:
                avg_fitness_ever = avg_fitness
                generations_without_improvement_avg = 0
                print(f"Significant avg-fitness improvement: +{avg_fitness - avg_fitness_ever:.4f}")
            else:
                generations_without_improvement_avg += 1
                print(f"No avg-fitness improvement for {generations_without_improvement_avg} generations")

            
            if (generations_without_improvement_best >= self.patience and
                 generations_without_improvement_avg >= self.patience):
                print(f"Early termination after {generation_idx+1} generations - stagnation or low diversity (diversity={diversity:.3f})")
                break
            
            
            if generation_idx == self.max_generations - 1:
                break
                
            
            parents = self.select_parents(population, fitness_scores)
            
            
            population = self.create_next_generation(parents)
            
            images : List[Image] = []
            for i, seed in enumerate(population):
                if seed.get_id() in self._cached_elite_images:
                    img_obj = self._cached_elite_images[seed.get_id()]
                    print(f"Use saved elite-image: {img_obj.get_id()}")
                else:
                    print(f"Generating image for {prompt.get_prompt()}:{seed.get_seed()}...")
                    # use_refiner = i < self.elite_size
                    use_refiner = False
                
                    if use_refiner:
                        
                        image = self.image_generator.generate_and_refine(prompt, seed)
                    else:
                        
                        image = self.image_generator.generate_quick_image(prompt, seed)
                
                    
                    img_obj = self.image_generator.save_img(image, prompt, seed, refined=False, db_client=self.db_client)
                images.append(img_obj)
                
            print(f"Extracting embeddings for {len(images)} images")

            for img  in images:
                image_data = None
                try:
                    with img.load_image() as pil_image:
                        image_data = pil_image.convert("RGB").copy()  
                        emb = self.embedding_extractor.extract_image_embedding(image_data)
                except Exception as e:
                    print(f"Error with: {img.get_id()}: {e}")
                    continue

                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu()
                img.blip_embedding = emb
                img.save_to_db()

            all_embeddings = [img.blip_embedding for img in images]
            local_mean_embedding = torch.stack(all_embeddings).mean(dim=0).cpu().numpy()
            global_mean_embedding = prompt.get_mean_embedding()
            print(f"Calculating scores for {len(images)} images...")
            for img in images:
                img.coherence_score = self.image_evaluator.calculate_coherence_score(img.load_image(), prompt.get_prompt())
                img.global_outlier_score = self.image_evaluator.calculate_outlier_score(img.blip_embedding.cpu().numpy(), global_mean_embedding) 
                img.local_outlier_score =  self.image_evaluator.calculate_outlier_score(img.blip_embedding.cpu().numpy(), local_mean_embedding)
                img.quality_score = self.image_evaluator.calculate_quality_score(img.load_image())
                img.save_to_db()
                print(f"Scores: {img.get_id()} - Coherence: {img.coherence_score:.4f}, Global Outlier: {img.global_outlier_score:.4f}, Local Outlier: {img.local_outlier_score:.4f}, Quality: {img.quality_score:.4f}")

            generation = Generation(
                experiment_id=self.experiment.id,
                generation_index=generation_idx,
                prompt_id=prompt.get_id(),
                seed_ids=[seed.get_id() for seed in population],
                elite_seed_ids=[s.get_id() for s in parents[:self.elite_size]],
                image_ids=[img.get_id() for img in images],
                avg_fitness=avg_fitness,
                best_fitness=best_fitness,
                diversity=self.calculate_population_diversity(population),
                db_client=self.db_client
            )
            
            generation.save_to_db()
            self.experiment.add_generation(generation.get_id())
            self.experiment.save_to_db()
            images.clear()
            torch.cuda.empty_cache()
            gc.collect()

            prompt.clean_mean_embedding()



        final_fitness = self.evaluate_fitness(population, prompt)
        best_seeds = sorted(
            population,
            key=lambda seed: final_fitness.get(seed.get_id(), 0.0),
            reverse=True
        )
                
        self.experiment.set_final_results([seed.get_id() for seed in best_seeds])
        
        print(f"Population Diversity: {diversity:.2f}, Mutation Rate: {self.mutation_rate:.2f}")
        return best_seeds, evolution_stats