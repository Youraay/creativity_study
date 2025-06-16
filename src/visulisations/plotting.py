import torch
import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List
from custom_types import Noise
from collections import defaultdict
import os
import csv
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

OUTPUT_PATH = "/scratch/dldevel/sinziri/creativity_study/plots"
OUTPUT_FILE_PATH = "/scratch/dldevel/sinziri/creativity_study/files"
def generate_umap(generation_map: Dict[int, List[Noise]], file_name:str):
    valid_noises = []

    for generation in generation_map.values():
        valid_noises.extend(generation)

    if not valid_noises:
        print("Keine gültigen Noises für UMAP – Abbruch.")
        return

    id_to_index = {n.id: idx for idx, n in enumerate(valid_noises)}

    try:
        embeddings = torch.stack([
            n.image_embs for n in valid_noises if isinstance(n.image_embs, torch.Tensor)
        ]).cpu().numpy()
    except Exception as e:
        print(f"Fehler beim Stacken der image_embs: {e}")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    generations = [n.first_appearance for n in valid_noises]
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=generations, cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, label='Generation')

    for child in valid_noises:
        child_idx = id_to_index[child.id]
        for parent_id in getattr(child, "parents", []):
            if parent_id in id_to_index:
                parent_idx = id_to_index[parent_id]
                x = [emb_2d[parent_idx, 0], emb_2d[child_idx, 0]]
                y = [emb_2d[parent_idx, 1], emb_2d[child_idx, 1]]
                plt.plot(x, y, color='gray', alpha=0.4, linewidth=1)

    plt.title("UMAP Trajectory of Noise Evolution")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches='tight')


def generate_pca(generation_map: Dict[int, List[Noise]], file_name:str):
    valid_noises = []

    for generation in generation_map.values():
        valid_noises.extend(generation)

    if not valid_noises:
        print("Keine gültigen Noises für PCA – Abbruch.")
        return

    pooled_embeddings = []
    for n in valid_noises:
        if n.noise_embeddings is not None and n.noise_embeddings.ndim > 0:
            pooled_embeddings.append(n.noise_embeddings.mean(dim=0))
    if not pooled_embeddings:
        print("Keine gültigen Noise-Embeddings – PCA abgebrochen.")
        return

    embeddings_np = torch.stack(pooled_embeddings).cpu().numpy()

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(embeddings_np)

    generations = [n.first_appearance for n in valid_noises]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=generations, cmap='plasma', s=30, alpha=0.8)
    plt.colorbar(scatter, label='Generation')
    plt.title("PCA of Noise Embeddings")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches='tight')


def generate_mean_and_standard_deviation(generation_map: Dict[int, List[Noise]], file_name:str):
    valid_noises = []

    for generation in generation_map.values():
        valid_noises.extend(generation)

    if not valid_noises:
        print("Keine gültigen Noises für Statistik – Abbruch.")
        return

    generation_fitness = defaultdict(list)

    for noise in valid_noises:
        try:
            generation_fitness[noise.first_appearance].append(noise.fitness)
        except AttributeError:
            print(f"Fehlende Fitness oder Generation bei Noise {getattr(noise, 'id', '?')}")

    sorted_generations = sorted(generation_fitness.keys())
    mean_fitness = [np.mean(generation_fitness[g]) for g in sorted_generations]
    std_fitness = [np.std(generation_fitness[g]) for g in sorted_generations]

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_generations, mean_fitness, label='Mean Fitness', marker='o')
    plt.fill_between(sorted_generations,
                     np.array(mean_fitness) - np.array(std_fitness),
                     np.array(mean_fitness) + np.array(std_fitness),
                     color='orange', alpha=0.3, label='±1 StdDev')
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches='tight')


def save_noises_to_csv(all_noises: Dict[int, List[Noise]], csv_name: str = "noises.csv"):
    output_dir = OUTPUT_FILE_PATH 
    os.makedirs(output_dir, exist_ok=True)

    subfolders = {
        "noise_embeddings": "noise_embs",
        "clip_embeddings": "clip_embs",
        "blip_embeddings": "blip_embs",
        "image_embs": "image_embs",
        "images": "images"
    }
    for folder in subfolders.values():
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "id", "prompt", "seed", "first_appearance", "last_appearance",
        "fitness", "image_caption", "parents", "scores",
        "noise_embeddings_path", "clip_embeddings_path",
        "blip_embeddings_path", "image_embs_path", "image_path"
    ]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for generation, noises in all_noises.items():
            for noise in noises:
                row = {
                    "id": getattr(noise, "id", ""),
                    "prompt": getattr(noise, "prompt", ""),
                    "seed": getattr(noise, "seed", ""),
                    "first_appearance": getattr(noise, "first_appearance", -1),
                    "last_appearance": getattr(noise, "last_appearance", -1),
                    "fitness": getattr(noise, "fitness", None),
                    "image_caption": getattr(noise, "image_caption", ""),
                    "parents": ",".join(str(p) for p in getattr(noise, "parents", [])),
                    "scores": repr(getattr(noise, "scores", {}))
                }

                def save_tensor(tensor, subdir):
                    if tensor is None:
                        return ""
                    path = os.path.join(output_dir, subdir, f"{noise.id}.pt")
                    try:
                        torch.save(tensor.cpu(), path)
                    except Exception as e:
                        print(f"Fehler beim Speichern von {path}: {e}")
                        return ""
                    return path

                row["noise_embeddings_path"] = save_tensor(getattr(noise, "noise_embeddings", None), subfolders["noise_embeddings"])
                row["clip_embeddings_path"] = save_tensor(getattr(noise, "clip_embeddings", None), subfolders["clip_embeddings"])
                row["blip_embeddings_path"] = save_tensor(getattr(noise, "blip_embeddings", None), subfolders["blip_embeddings"])
                row["image_embs_path"] = save_tensor(getattr(noise, "image_embs", None), subfolders["image_embs"])

                try:
                    if getattr(noise, "pil", None) is not None:
                        image_path = os.path.join(output_dir, subfolders["images"], f"{noise.id}.png")
                        noise.pil.save(image_path)
                    else:
                        image_path = ""
                    row["image_path"] = image_path
                except Exception as e:
                    print(f"Fehler beim Speichern des Bildes für {noise.id}: {e}")
                    row["image_path"] = ""

                writer.writerow(row)

    print(f"Alle Noises und Embeddings wurden gespeichert unter: {output_dir}")