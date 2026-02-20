import os
from pathlib import Path
import logging

# ───────────────────────────────────────────────
#  Professional logging setup
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
#  Root project name — MUST match specification
# ───────────────────────────────────────────────
PROJECT_ROOT = "Movie-Hybrid-Recommendation-System"

# ───────────────────────────────────────────────
#  All directories (using mkdir -p style paths)
# ───────────────────────────────────────────────
directories = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data",
    "src/models",
    "src/serving",
    "src/utils",
    "models/cbf",
    "models/cf",
    "models/sentiment",
    "app/routes",
    "app/services",
    "migrations",
    "mlruns",
    "scripts",
    "tests",
    "docker",
    ".github/workflows",
    "requirements",
]

# ───────────────────────────────────────────────
#  Root-level files that MUST exist (empty or minimal)
# ───────────────────────────────────────────────
root_files = [
    ".gitignore",
    ".env.example",
    "Procfile",
    "runtime.txt",
    "README.md",
]

# ───────────────────────────────────────────────
#  Docker & GitHub workflow placeholder files
# ───────────────────────────────────────────────
docker_and_workflow_files = [
    "docker/Dockerfile.app",
    "docker/Dockerfile.trainer",
    "docker/docker-compose.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/retrain.yml",
]

# ───────────────────────────────────────────────
#  Requirements files (empty for now — populate later)
# ───────────────────────────────────────────────
requirements_files = [
    "requirements/base.txt",
    "requirements/training.txt",
    "requirements/production.txt",
]

# ───────────────────────────────────────────────
#  Exactly the 7 notebooks mandated in SECTION 0
# ───────────────────────────────────────────────
notebooks = [
    "notebooks/01_data_ingestion_eda.ipynb",
    "notebooks/02_feature_engineering_cbf.ipynb",
    "notebooks/03_cbf_model_training.ipynb",
    "notebooks/04_feature_engineering_cf.ipynb",
    "notebooks/05_cf_model_training.ipynb",
    "notebooks/06_hybrid_fusion.ipynb",
    "notebooks/07_evaluation.ipynb",
]

# ───────────────────────────────────────────────
#  All files that should have .gitkeep in empty dirs
# ───────────────────────────────────────────────
gitkeep_dirs = [
    "data/raw",
    "data/processed",
    "data/external",
    "models/cbf",
    "models/cf",
    "models/sentiment",
    "migrations",
    "mlruns",
]

# ───────────────────────────────────────────────
#  Execute creation
# ───────────────────────────────────────────────
def create_project_structure():
    # 1. Create root if it doesn't exist
    root_path = Path(PROJECT_ROOT)
    root_path.mkdir(exist_ok=True)
    logger.info(f"Project root ensured: {root_path.resolve()}")

    os.chdir(root_path)

    # 2. Create all directories
    for d in directories:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/ensured: {path}")

    # 3. Create .gitkeep in specified empty directories
    for d in gitkeep_dirs:
        gitkeep = Path(d) / ".gitkeep"
        gitkeep.touch(exist_ok=True)
        logger.info(f".gitkeep created: {gitkeep}")

    # 4. Create root files (empty or minimal)
    for f in root_files + requirements_files + docker_and_workflow_files + notebooks:
        path = Path(f)
        path.parent.mkdir(parents=True, exist_ok=True)  # safety
        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logger.info(f"File created (empty): {path}")
        else:
            logger.info(f"File already exists: {path}")

    logger.info("───────────────────────────────────────────────")
    logger.info("Phase 0 initialization COMPLETE")
    logger.info("Next step: populate .gitignore, README.md, and .env.example")
    logger.info("Then commit: git add . && git commit -m 'Phase 0: Initialize exact project structure per SECTION 0'")
    logger.info("After commit, you may proceed to SECTION 1 — DATA COLLECTION")

if __name__ == "__main__":
    create_project_structure()