from utils import MongoDB

if __name__ == "__main__":
    print("Öffne Datenbank")
    db = MongoDB()
    db.get_collection("seeds")