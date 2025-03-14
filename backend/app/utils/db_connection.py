from pymongo import MongoClient 

class MongoDB():

    def __init__(self):

        CONNECTION_STRING = "mongodb://admin:secret@192.168.0.2:27017/"
        self.client = MongoClient(CONNECTION_STRING)
        self.db = self.client['sample_database']
        
    def get_database(self, db='sample_database'):

        return self.client[db]
    
    def get_collection(self, collection):
         
        return self.db[collection]

    def get_item_from_collection_by_id(self, collection:str, id:str) -> dict:
        return self.get_collection(collection).find_one({"_id": id})
    
    def get_items_from_collection_by_attr(self, collection:str, attr:str, value:str) -> dict:
        return self.get_collection(collection).find({attr: value})
    
    def insert_item_into_collection(self, collection:str, item:dict) -> bool:
        return bool(self.get_collection(collection).insert_one(item))
    
    
    def update_item_in_collection_by_id(self, collection:str, id:str, item:dict)-> bool:
        return bool(self.get_collection(collection).update_one({"_id":id}, {"$set": item}))
    
    def update_items_in_collection(self, collection:str, query:dict, item:dict)-> bool:
        return bool(self.get_collection(collection).update_many(query, {"$set":item}))