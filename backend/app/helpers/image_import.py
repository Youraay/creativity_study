import os
import re
from PIL import Image as PILImage

from utils.embedding_extractor import EmbeddingExtractor
from utils.db_connection import MongoDB
import config

def import_generated_images_from_folder(folder_path: str, db_client):
    from data_model import Prompt, Seed, Image, DataFactory
    print("Start Import of genrated images")
    pattern = re.compile(r"^(?P<prompt>.+)_(?P<seed>\d+)_(?P<epoch>\d+)_(?P<guidance>[\d.]+)\.png$")
    factory = DataFactory(db_client)
    embedding_extractor = EmbeddingExtractor()

    imported_count = 0 

    for filename in os.listdir(folder_path):
        if not filename.endswith(".png"):
            continue

        match = pattern.match(filename)
        if not match:
            print(f"Überspringe Datei (falsches Format): {filename}")
            continue

        prompt_text = match.group("prompt")
        seed_value = int(match.group("seed"))
        steps = int(match.group("epoch"))
        guidance = float(match.group("guidance"))

        print(f"Importiere: {filename} | Prompt: '{prompt_text}' | Seed: {seed_value}")

        
        prompt = Prompt(prompt_text, db_client=db_client, load_from_db=True)
        prompt.save_to_db()

        
        seed = Seed(seed_value, db_client=db_client, load_from_db=True)
        seed.save_to_db()

        
        file_path = os.path.join(folder_path, filename)
        image_data = PILImage.open(file_path)

        img_obj = Image(
            seed=seed,
            prompt=prompt,
            pil_image=image_data,
            generation_steps=steps,
            guidance_scale=guidance,
            use_refiner=False,
            db_client=db_client,
            file_path=file_path,
        )


        embedding = embedding_extractor.extract_image_embedding(image_data)
        if embedding is not None:
            img_obj.blip_embedding = embedding
        else:
            print(f"Kein Embedding extrahiert für {filename}")


        img_obj.save_to_db()


        prompt.register_usage(seed.get_id(), img_obj.get_id())

        seed.add_generated_image(img_obj.get_id(), prompt.get_prompt())

        imported_count += 1

    print(f"Import abgeschlossen. {imported_count} Bild(er) erfolgreich importiert.")
