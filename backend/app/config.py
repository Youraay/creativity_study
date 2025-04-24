# ─────────────────────────────────────────
# STABLE DIFFUSION SETTINGS
# ─────────────────────────────────────────

N_STEPS = 50
HIGH_NOISE_FRAC = 0.8
GUIDANCE_SCALE = 7.5  
NEGATIVE_PROMPT = ""
USE_HALF_PRECISION = True

# ─────────────────────────────────────────
# EVOLUTIONARY ALGORITHM SETTINGS
# ─────────────────────────────────────────

POPULATION_SIZE = 16
ELITE_SIZE = 3
MUTATION_RATE = 0.08
MAX_GENERATIONS = 20
PATIENCE = 4
MIN_IMPROVEMENT = 0.02
MIN_DIVERSITY = 0.05
MAX_DIVERSITY = 0.4
TEMPERATURE = 0.5 #For the softmax function in the parent selection
CROSSOVER_RATE = 0.5 #dDfines if the one parent is passed forward or a child is created
RANDOM_SEED = 42 # Set a random seed for reproducibility
# ─────────────────────────────────────────
# DATABASE SETTINGS
# ─────────────────────────────────────────

MONGODB_CONNECTION_STRING = "mongodb://admin:secret@localhost:27017/"
MONGODB_DB_NAME = "sample_database"

# ─────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────

SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REF_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BLIP2_MODEL = "Salesforce/blip2-flan-t5-xl"
USE_BLIP2_QNA = False

# ─────────────────────────────────────────
# EVALUATION SETTINGS
# ─────────────────────────────────────────

GLOABL_OUTLIER_WEIGHT = 0.9
LOCAL_OUTLIER_WEIGHT =0
COHERENCE_WEIGHT =0
QUALITY_WEIGHT =0.1 
SIGMOID_K = 10 # Defines the steepness of the sigmoid function (How distinct the outlier score should be)




