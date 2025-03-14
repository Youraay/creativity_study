class Seed:

    def __init__(self, seed: str):
        self.id: str = seed 
        self.seed : str = seed
        self.dict = {"_id": self.id, "seed": self.seed}

    def __init__(self, dict:dict):
        self.id: str = dict["_id"]
        self.seed : str = dict["seed"]
        self.dict = dict

    def get_seed(self) -> str: return self.seed
    
    def get_id(self) -> str: return self.id 

    def get_attr(self, attr:str) -> str: return self.dict[attr]

    def set_attr(self, attr:str, value:str) -> None: 
        if attr not in ["_id", "seed"]:
            self.dict[attr] = value