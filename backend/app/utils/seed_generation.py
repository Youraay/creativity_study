from data_model import Seed
import datetime

class SeedGeneration():


    """
       Generate seeds after diffrerent schematics and return them as a list of dictionaries. 
    """
    def __init__ (self):

        pass


    def generate_evenly_distributed_seeds(self, n: int, bit:int) -> list:

        """
            Generate n seeds evenly distributed over the range of 2^bit.

            Args:
                n: number of seeds to generate
                bit: the bit length of the seeds
        """
        list = []
        for i in range(n):
            seed_dict = {
                "_id": (i*(2**bit)//n),
                "seed": (i*(2**bit)//n),
                "generated_images": [],
                "created_at": datetime.now()
            }
            seed = Seed(seed_dict)
            list.append(seed)

        return self.seed_list