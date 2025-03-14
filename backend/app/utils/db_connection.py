from pymongo import MongoClient 

class MongoDB():

    def __init__(self):

        CONNECTION_STRING = "mongodb://admin:secret@192.168.0.2:27017/"
        self.client = MongoClient(CONNECTION_STRING)
        self.db = self.client['sample_database']
        
    def get_database(self, db='sample_database'):

        return self.client[db]
    
    def change_database(self, db='sample_database'):
            self.db = self.client[db]

    def get_collection(self, collection):
         
        return self.db[collection]

