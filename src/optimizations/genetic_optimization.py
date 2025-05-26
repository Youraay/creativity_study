from custom_types import Device, Noise, Argument
from typing import List, Generic, TypeVar
from evaluators import Evaluator
from mutators import Mutator
from crossover import Crossover
from .selector import Selector
import random
class GeneticOptimizaion():

    def __init__ (self,
                  generations: int, 
                  population_size: int,
                  selector: Selector[Noise],
                  evaluators: List[Evaluator[Noise]],
                  mutator : Mutator[Noise], 
                  crossover_function : Crossover[Noise],
                  base_population: List[Noise], 
                  device: Device = "cuda", 
                  initial_mutation_rate: float = 0.1,
                  crossover_rate: float = 0.1,
                  elitism_count: int = 0,
                  stric_osga: bool = False, 
                  ):
        if base_population:
            self.population = base_population
        else:
            self.population = self.initialize_population(population_size)
        self.evaluators = evaluators

    def initialize_population(self, population_size: int) -> List[Noise]:
        
        return [Noise.from_seed(random.randint(1, 2**32)) for _ in range(population_size)]  
        

    def perform_generation(self):
        pass

    def calculate_fitness(self, noise: Noise) -> float:
        
        evaluations = [evaluator.evaluate(noise) for evaluator in self.evaluators]
        pass
    
   
    