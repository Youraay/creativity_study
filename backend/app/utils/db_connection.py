from pymongo import MongoClient , errors
import numpy as np
from typing import List
import time
import config
class MongoDB():

    def __init__(self):

        CONNECTION_STRING = config.MONGODB_CONNECTION_STRING
        self.client = MongoClient(CONNECTION_STRING)
        self.db_name = config.MONGODB_DB_NAME
        self.db = self.client['sample_database']
        
    def get_database(self, db='sample_database'):

        return self.client[db]
    
    def get_collection(self, collection):
         
        return self.db[collection]

    def get_item_from_collection_by_id(self, collection:str, id:str) -> dict:
        return self.get_collection(collection).find_one({"_id": id})
    
    def get_items_from_collection_by_attr(self, collection:str, attr:str, value:str) -> dict:
        return self.get_collection(collection).find({attr: value})
    
    def get_items_from_collection_by_attrs(self, collection:str, attrs) -> dict:
        return self.get_collection(collection).find(attrs)
    
    def insert_item_into_collection(self, collection:str, item:dict) -> bool:
        return bool(self.get_collection(collection).insert_one(item))
    
    def insert_many_items_into_collection(self, collection:str, items:list) -> bool:
        return bool(self.get_collection(collection).insert_many(items))
    
    def update_item_in_collection_by_id(self, collection:str, id:str, item:dict)-> bool:
        return bool(self.get_collection(collection).update_one({"_id":id}, {"$set": item}, upsert=True))
    
    def update_items_in_collection(self, collection:str, query:dict, item:dict)-> bool:
        return bool(self.get_collection(collection).update_many(query, {"$set":item}, upsert=True))
    
    def get_image_embeddings_for_prompt(self, prompt_id: str, guidance_scale =None, steps = None) -> List[List[float]]:
        guidance = guidance_scale if guidance_scale is not None else config.GUIDANCE_SCALE
        steps = steps if steps is not None else config.N_STEPS
        images = self.get_items_from_collection_by_attrs("images", {"prompt_id": prompt_id, "guidance_scale":guidance, "generation_steps": steps})
        embeddings = []
        for img in images:
            emb = img.get("blip_embedding")
            if emb and isinstance(emb, list):
                arr = np.array(emb)
                if not np.isnan(arr).any():
                    embeddings.append(emb)
        return embeddings
    
    def get_all_image_embeddings_for_prompt(self, prompt_id: str) -> List[np.ndarray]:
        
        results = self.get_items_from_collection_by_attrs("images", {"prompt_id": prompt_id})
        return [np.array(img["embedding"]) for img in results if "embedding" in img]
    

    def get_image_by_seed_and_prompt(self, seed_id: str, prompt_id: str):
        # TODO: Add config for filter
        result = self.get_collection("images").find_one({
            "seed_id": seed_id,
            "prompt_id": prompt_id
        })

        if result:
            
            return result
        return None
    
    def get_all_image_embeddings_for_prompt(self, prompt_id: str) -> List[np.ndarray]:
        # TODO: Add config for filter
        
        results = self.get_collection("images").find({
            "prompt_id": prompt_id,
            "embedding": {"$exists": True}
        })

        return [np.array(r["embedding"]) for r in results if "embedding" in r]
    
    def check_database_connection(self, max_retries=3, retry_delay=2):
       
        for attempt in range(max_retries):
            try:
                
                
                
                
                self.client.admin.command('ping')
                
                db = self.client[self.db_name]
                collections = db.list_collection_names()
                print("MongoDB-Verbindung hergestellt!")
                print("\nVerfügbare Collections:")
                for collection in collections:
                    count = db[collection].count_documents({})
                    print(f"  • {collection}: {count} Einträge")

                
                return True
                
            except errors.ServerSelectionTimeoutError as e:
                print(f"Verbindung fehlgeschlagen: MongoDB-Server nicht erreichbar")
                if attempt < max_retries - 1:
                    print(f"Neuer Versuch in {retry_delay} Sekunden...")
                    time.sleep(retry_delay)
                else:
                    print("Maximale Anzahl an Verbindungsversuchen erreicht.")
                    return False
                    
            except errors.OperationFailure as e:
                print(f"Authentifizierungsproblem: {e}")
                return False
                
            except Exception as e:
                print(f"Unerwarteter Fehler bei Datenbankverbindung: {e}")
                return False
