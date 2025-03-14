class Seed:

    def __init__(self, id: str, seed: str):
        self.id: str = id 
        self.seed : str = seed

    def get_seed(self) -> str:

        return self.seed
    
    def get_id(self) -> str:

        return self.id 