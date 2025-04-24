import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, Any
from datetime import datetime
from utils.db_connection import MongoDB
import random
from data_model import Prompt, Seed, Image, DataFactory
class PromptManager:
    def __init__(self):
        
        pass
    def generate_prompt_set(self):
        
        prompt_sets = {}
        
        

        prompts = {
               "living_beings": [
                "dog",  
                "cat",  
                "bird",  
                "person" 
            ],
            "nature": [
                "tree",  
                "flower", 
                "mountain",
                "cloud" 
            ],
            "everyday_objects": [
                "chair",
                "table",
                "cup",
                "book"
            ],
            "abstract_concepts": [
                "time",
                "dream",
                "motion",
                "happiness"
            ]
        }
        adjectives = {
            "living_beings": ["small", "large", "colorful", "spotted", "unusual"],
            "nature": ["tall", "green", "blooming", "ancient", "twisted"],
            "everyday_objects": ["wooden", "modern", "broken", "ornate", "glass"],
            "abstract_concepts": ["flowing", "bright", "dark", "chaotic", "serene"]
        }
        
        contexts = {
            "living_beings": ["in a field", "underwater", "in space", "in a city", "in the forest"],
            "nature": ["in fog", "at sunset", "in winter", "underwater", "in desert"],
            "everyday_objects": ["on a table", "in a museum", "floating", "in ruins", "in nature"],
            "abstract_concepts": ["in a dream", "visualized", "as a pattern", "as a statue", "in motion"]
        }
        
        
        for category, objects in prompts.items():
            prompt_sets[category] = []
            
            for obj in objects:
        
                prompt_sets[category].append({
                    "text": obj,
                    "object": obj,
                    "complexity": "basic"
                })
                
        
                adj = random.choice(adjectives[category])
                prompt_sets[category].append({
                    "text": f"{adj} {obj}",
                    "object": obj,
                    "complexity": "descriptive",
                    "adjective": adj
                })
                
        
                context = random.choice(contexts[category])
                prompt_sets[category].append({
                    "text": f"{obj} {context}",
                    "object": obj,
                    "complexity": "contextual",
                    "context": context
                })
                
        
                adj = random.choice(adjectives[category])
                context = random.choice(contexts[category])
                prompt_sets[category].append({
                    "text": f"{adj} {obj} {context}",
                    "object": obj,
                    "complexity": "complex",
                    "adjective": adj,
                    "context": context
                })
        
        return prompt_sets
    