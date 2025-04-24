import os

import pymongo
import time
import sys
from utils import MongoDB, ImageGeneration, ModelManager
from manager.prompt_manager import PromptManager
from manager.experiment_runner import ExperimentRunner
from data_model import Prompt
from datetime import datetime


if __name__ == "__main__":

    mm = ModelManager()
    db_client = MongoDB()
    image_generator = ImageGeneration()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prompt_manager = PromptManager()
    runner = ExperimentRunner(db_client, image_generator)


    
    category = "living_beings"
    prompt_sets = prompt_manager.generate_prompt_set()
    prompt_dict = prompt_sets[category][0]  # Level of complexity: [0 - 3]
    prompt = Prompt( "person", db_client=db_client)
    prompt.set_metadata(category=category,complexity=prompt_dict["complexity"], object="person")

    prompt.save_to_db()
    best_seeds, stats = runner.run_single_prompt_evolution(prompt)

    result_data = {
    "prompt_id": prompt.get_id(),
    "prompt_text": prompt.get_prompt(),
    "category": category,
    "complexity": prompt_dict["complexity"],
    "timestamp": datetime.now(),
    "best_seeds": [seed.get_id() for seed in best_seeds],
    "evolution_stats": stats
    }
    