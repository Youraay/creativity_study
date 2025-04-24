from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymongo import MongoClient
from typing import List, Optional
from datetime import datetime
import os

"""A super‑lightweight API for your Stable‑Diffusion experiment data.

Endpoints
---------
GET  /api/images            → list of image docs (minified)
GET  /api/prompts           → list of prompt docs
GET  /api/generations/{id}  → single generation doc
GET  /api/experiments/{id}  → single experiment doc
GET  /results/{path}        → serves PNGs stored on disk (static)

Run with:
    MONGODB_URI=mongodb://localhost:27017 \
    MONGO_DB=imagedb \
    STATIC_ROOT=/absolute/path/to/project \
    uvicorn app:app --reload

The frontend can live in the same directory. If you run uvicorn on the
same host/port, leave `API_BASE` in index.html empty – fetch() will hit the
same origin automatically.
"""

# ─── MongoDB connection ────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "imagedb")
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

images_col = db["images"]
prompts_col = db["prompt"]  # collection name in your sample
generations_col = db.get_collection("generation", None)
experiments_col = db.get_collection("experiment", None)

# ─── FastAPI setup ────────────────────────────────────────────────────────
app = FastAPI(title="Image‑Experiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ───────────────────────────────────────────────────────────────

def serialise(doc: dict) -> dict:
    """Make Mongo docs JSON‑serialisable (ObjectId → str, dates → iso)."""
    if not doc:
        return doc
    doc["_id"] = str(doc["_id"])
    for key in ("creation_date", "timestamp"):
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = doc[key].isoformat()
    return doc

# ─── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/images")
async def get_images(prompt_id: Optional[str] = None, limit: int = 500):
    query = {"prompt_id": prompt_id} if prompt_id else {}
    projection = {
        "_id": 1,
        "file_path": 1,
        "prompt_id": 1,
        "prompt_text": 1,
        "seed_id": 1,
        "quality_score": 1,
        "guidance_scale": 1,
        "generation_steps": 1,
        "creation_date": 1,
        "coherence_score": 1,
    }
    docs = (
        images_col.find(query, projection)
        .sort("creation_date", -1)
        .limit(limit)
    )
    return [serialise(d) for d in docs]


@app.get("/api/prompts")
async def get_prompts():
    docs = prompts_col.find({}, {"_id": 1, "prompt": 1})
    return [serialise(d) for d in docs]


@app.get("/api/generations/{gen_id}")
async def get_generation(gen_id: str):
    if generations_col is None:
        raise HTTPException(404, "Collection 'generation' not configured")
    doc = generations_col.find_one({"_id": gen_id})
    if not doc:
        raise HTTPException(404, "Generation not found")
    return serialise(doc)


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    if experiments_col is None:
        raise HTTPException(404, "Collection 'experiment' not configured")
    doc = experiments_col.find_one({"_id": exp_id})
    if not doc:
        raise HTTPException(404, "Experiment not found")
    return serialise(doc)


STATIC_ROOT = os.getenv("STATIC_ROOT", ".")

@app.get("/results/{file_path:path}")
async def serve_file(file_path: str):
    full_path = os.path.join(STATIC_ROOT, "results", file_path)
    if not os.path.isfile(full_path):
        raise HTTPException(404, "File not found")
    return FileResponse(full_path)
