from utils import MongoDB
from pymongo import MongoClient 

class DataFactory():

    def __init__(self, mongo: MongoDB) -> None:

        self.mongo = mongo
        self.db = mongo.get_database()

    def get_seeds(self) -> dict:
        return self.mongo.get_collection('seeds').find()
    
    def get_prompts(self) -> dict:
        return self.mongo.get_collection('prompts').find()
    
    def get_imgs(self) -> dict:
        return self.mongo.get_collection('imgs').find()
    
    def get_seed_by_id(self, id:str) -> dict:
        return self.mongo.get_item_from_collection_by_id('seeds', id)

    def get_seed_by_attr(self, attr:str, value:str)-> dict:
        return self.mongo.get_collection('seeds').find({attr: value})
    
    def get_prompt_by_id(self, id:str) -> dict:
        return self.mongo.get_item_from_collection_by_id('prompts', id)
    
    def get_prompt_by_attr(self, attr:str, value:str)-> dict:
        return self.mongo.get_collection('prompts').find({attr: value})
    
    def get_img_by_id(self, id:str) -> dict:
        return self.mongo.get_item_from_collection_by_id('imgs', id)

    def get_img_by_attr(self, attr:str, value:str)-> dict:
        return self.mongo.get_items_from_collection_by_attr('imgs', attr, value)

    def insert_seed(self, seed:dict) -> bool:
        return self.mongo.insert_item_into_collection('seeds', seed)
    
    def insert_prompt(self, prompt:dict) -> bool:
        return self.mongo.insert_item_into_collection('prompts', prompt)
    
    def insert_img(self, img:dict) -> bool:
        return self.mongo.insert_item_into_collection('imgs', img)
    
    def insert_many_seeds(self, seeds: list) -> bool:
        print(seeds)
        return self.mongo.insert_many_items_into_collection('seeds', seeds)
 

