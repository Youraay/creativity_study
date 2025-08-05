import torch
import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List
from src.custom_types import Noise
from collections import defaultdict
import os
import csv
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

OUTPUT_PATH = "/scratch/dldevel/sinziri/creativity_study/files"
OUTPUT_FILE_PATH = "/scratch/dldevel/sinziri/creativity_study/files"
OUTPUT_PATH = ""
OUTPUT_FILE_PATH = ""
def generate_umap(generation_map: Dict[int, List[Noise]], file_name:str):
    valid_noises = []

    for generation in generation_map.values():
        valid_noises.extend(generation)

    if not valid_noises:
        print("Keine gültigen Noises für UMAP – Abbruch.")
        return

    id_to_index = {n.id: idx for idx, n in enumerate(valid_noises)}

    try:
        # Modify this part to reduce dimensions
        embeddings = torch.stack([
            n.image_embs.mean(dim=0) if n.image_embs.ndim > 1 else n.image_embs 
            for n in valid_noises if isinstance(n.image_embs, torch.Tensor)
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
            # Handle embeddings based on their dimensionality
            if n.noise_embeddings.ndim == 1:
                # Already a vector
                pooled_embeddings.append(n.noise_embeddings)
            elif n.noise_embeddings.ndim == 2:
                # 2D tensor - take mean along first dimension
                pooled_embeddings.append(n.noise_embeddings.mean(dim=0))
            else:
                # Higher dimensional tensor - flatten to 1D
                # First take mean along all but the last dimension
                embedding = n.noise_embeddings
                while embedding.ndim > 1:
                    embedding = embedding.mean(dim=0)
                pooled_embeddings.append(embedding)
                
    if not pooled_embeddings:
        print("Keine gültigen Noise-Embeddings – PCA abgebrochen.")
        return

    embeddings_np = torch.stack(pooled_embeddings).cpu().numpy()
    
    # Debug to check final shape
    print(f"PCA input shape: {embeddings_np.shape}")
    
    # Rest of the function remains the same
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


def save_noises_to_csv(all_noises: Dict[int, List[Noise]], path: str,csv_name: str = "noises.csv"):
    output_dir = f"{OUTPUT_FILE_PATH}/{path}/"
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
        "id", "prompt","generation", "seed", "first_appearance", "last_appearance",
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
                    "generation": generation,
                    "seed": getattr(noise, "seed", ""),
                    "first_appearance": getattr(noise, "first_appearance", -1),
                    "last_appearance": getattr(noise, "last_appearance", -1),
                    "fitness": getattr(noise, "fitness", 0.0),
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

                # try:
                #     if getattr(noise, "pil", None) is not None:
                #         image_path = os.path.join(output_dir, subfolders["images"], f"{noise.id}.png")
                #         noise.pil.save(image_path)
                #     else:
                #         image_path = ""
                #     row["image_path"] = image_path
                # except Exception as e:
                #     print(f"Fehler beim Speichern des Bildes für {noise.id}: {e}")
                #     row["image_path"] = ""

                writer.writerow(row)

    print(f"Alle Noises und Embeddings wurden gespeichert unter: {output_dir}")

# === Zusatz-Imports =========================================================
import torch.nn.functional as F
from itertools import combinations

# ---------------------------------------------------------------------------
def generate_best_fitness_curve(generation_map: Dict[int, List[Noise]],
                                file_name: str):
    """
    Zeigt die jeweils beste Fitness pro Generation als Linie.
    """
    best_per_gen = {g: max(n.fitness for n in noises)
                    for g, noises in generation_map.items()
                    if noises}
    if not best_per_gen:
        print("Keine Fitness-Werte gefunden – Plot übersprungen.")
        return

    gens = sorted(best_per_gen.keys())
    best = [best_per_gen[g] for g in gens]

    plt.figure(figsize=(10, 5))
    plt.plot(gens, best, marker="o")
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
def generate_diversity_metric(generation_map: Dict[int, List[Noise]],
                              file_name: str):
    """
    Durchschnittliche paarweise Distanz (1–Cos-Sim) der BLIP-Embeddings pro Generation.
    """
    diversity = {}
    for g, noises in generation_map.items():
        embs = [n.image_embs for n in noises
                if isinstance(n.image_embs, torch.Tensor)]
        if len(embs) < 2:
            continue
        mat = torch.stack(embs)
        mat = F.normalize(mat, dim=1)
        sim = mat @ mat.T                       # (N×N) Cos-Sim-Matrix
        dist = (1 - sim).triu(1)                # obere Dreiecksmatrix
        diversity[g] = dist[dist > 0].mean().item()

    if not diversity:
        print("Zu wenig Embeddings – Plot übersprungen.")
        return

    gens = sorted(diversity.keys())
    divs = [diversity[g] for g in gens]

    plt.figure(figsize=(10, 5))
    plt.plot(gens, divs, marker="o")
    plt.title("Embedding-Diversity per Generation")
    plt.xlabel("Generation")
    plt.ylabel("⌀ pairwise distance (1 – cos sim)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
def generate_mutation_vs_crossover_boxplot(generation_map: Dict[int,
                                      List[Noise]], file_name: str):
    """
    Vergleicht Fitness-Verteilungen je Entstehungsart (Crossover, Mutation,
    unverändert/Elite).
    """
    cross_fitness, mut_fitness, copy_fitness = [], [], []
    for noises in generation_map.values():
        for n in noises:
            crossed = getattr(n, "crossed", False)      # :contentReference[oaicite:2]{index=2}
            mutated = getattr(n, "mutated", False)
            if crossed:
                cross_fitness.append(n.fitness)
            elif mutated:
                mut_fitness.append(n.fitness)
            else:
                copy_fitness.append(n.fitness)

    data, labels = [], []
    if cross_fitness:
        data.append(cross_fitness); labels.append("Crossover")
    if mut_fitness:
        data.append(mut_fitness);   labels.append("Mutation")
    if copy_fitness:
        data.append(copy_fitness);  labels.append("Elite/Copy")

    if len(data) < 2:
        print("Nicht genug Kategorien für Boxplot – übersprungen.")
        return

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Fitness-Verteilung nach Entstehungsart")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
def generate_evaluator_contributions(generation_map: Dict[int, List[Noise]],
                                     file_name: str):
    """
    Gestapelte Fläche der Mittelwerte aller Evaluator-Scores je Generation.
    """
    # Score-Schlüssel bestimmen
    first_scores = next((n.scores for noises in generation_map.values()
                         for n in noises if getattr(n, "scores", None)), None)
    if not first_scores:
        print("Keine Scores gefunden – Plot übersprungen.")
        return
    eval_names = list(first_scores.keys())

    means = {name: [] for name in eval_names}
    gens  = sorted(generation_map.keys())
    for g in gens:
        noises = generation_map[g]
        for name in eval_names:
            vals = [n.scores.get(name, 0) for n in noises if n.scores]
            means[name].append(np.mean(vals) if vals else 0.0)

    plt.figure(figsize=(10, 6))
    stacks = [means[n] for n in eval_names]
    plt.stackplot(gens, *stacks, labels=eval_names)
    plt.title("Mittlere Evaluator-Scores (gestapelt)")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------------------
def generate_delta_to_first_generation(generation_map: Dict[int, List[Noise]],
                                       file_name: str):
    """
    Zeigt pro Generation die Differenz (Δ) von Mittelwert, Varianz und Maximum
    der Fitness relativ zu Generation 0.

    • ΔMean   = mean_g - mean_0
    • ΔVar    = var_g  - var_0
    • ΔMax    = max_g  - max_0
    """
    if 0 not in generation_map or not generation_map[0]:
        print("Generation 0 fehlt – Plot übersprungen.")
        return

    # Kennzahlen für Generation 0
    base_fitness = [n.fitness for n in generation_map[0]]
    base_mean = np.mean(base_fitness)
    base_var  = np.var(base_fitness)
    base_max  = np.max(base_fitness)

    gens, d_mean, d_var, d_max = [], [], [], []
    for g in sorted(generation_map.keys()):
        fitness = [n.fitness for n in generation_map[g]]
        if not fitness:
            continue
        gens.append(g)
        d_mean.append(np.mean(fitness) - base_mean)
        d_var.append(np.var(fitness)  - base_var)
        d_max.append(np.max(fitness)  - base_max)

    plt.figure(figsize=(10, 6))
    plt.plot(gens, d_mean, label="Δ Mean", marker="o")
    plt.plot(gens, d_var,  label="Δ Var",  marker="s")
    plt.plot(gens, d_max,  label="Δ Max",  marker="^")
    plt.axhline(0, color="black", lw=0.8)        # Baseline
    plt.title("Fitness-Unterschiede zu Generation 0")
    plt.xlabel("Generation")
    plt.ylabel("Differenz zur Start-Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{file_name}.png", dpi=300,
                bbox_inches="tight")
