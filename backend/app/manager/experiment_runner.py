from data_model import Experiment
from analysis.seed_evolution import SeedEvolution
from data_model import Prompt, Seed, DataFactory
from typing import Dict, List, Tuple
class ExperimentRunner:
    def __init__(self, db_client, image_generator):
        self.db_client = db_client
        self.image_generator = image_generator
        self.factory = DataFactory(db_client)

    def run_single_prompt_evolution(self, prompt: Prompt):
        
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt.get_prompt()}")
        print(f"{'='*80}")
        
        
        
        experiment = Experiment(prompt.get_id(), prompt.get_prompt(), db_client=self.db_client)
        experiment.set_metadata(**prompt.metadata)
        

        evolution = SeedEvolution(db_client=self.db_client, image_generator=self.image_generator)
        best_seeds, stats = evolution.evolve(prompt, experiment)
        
        return best_seeds, stats
    