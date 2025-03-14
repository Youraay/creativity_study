from utils import MongoDB, SeedGeneration
from data_model import DataFactory, Seed
if __name__ == "__main__":
    print("Öffne Datenbank")
    db = MongoDB()
    dF = DataFactory(db)
    seedGen = SeedGeneration()

    seeds = seedGen.generate_evenly_distributed_seeds(n=1000, bit=32)
    dF.insert_many_seeds(seeds)