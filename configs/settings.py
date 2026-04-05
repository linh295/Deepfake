import os
import pathlib
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Get CPU count for default workers
try:
    from multiprocessing import cpu_count
    DEFAULT_WORKERS = max(1, cpu_count() - 1)
except:
    DEFAULT_WORKERS = 1

class Settings(BaseSettings):
    ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent
    ENV_PATH: pathlib.Path = ROOT_DIR / ".env"
    
    # Directory paths
    RAW_DATA_DIR: pathlib.Path = ROOT_DIR / "FaceForensics++_C23 2"
    FRAME_DATA_DIR: pathlib.Path = ROOT_DIR / "frame_data"
    CROP_DATA_DIR: pathlib.Path = ROOT_DIR / "crop_data"
    AUDIT_FILE: pathlib.Path = ROOT_DIR / "audit" / "face_detection_audit.csv"
    MODEL_DIR: pathlib.Path = ROOT_DIR / "models"
    FIGURE_DIR: pathlib.Path = ROOT_DIR / "figures"
    
    # Frame extraction settings
    TARGET_FPS: int = 5
    FRAME_EXTRACTION_METADATA_CSV: str = "frame_extraction_metadata.csv"
    FRAME_EXTRACTION_AUDIT_CSV: str = "frame_extraction_audit.csv"

    # Face detection settings
    FACE_DETECTION_THRESHOLD: float = 0.9
    FACE_DETECTION_BACKUP_SUFFIX: str = ".bak.csv"
    RETINAFACE_WEIGHT_DIR: pathlib.Path = MODEL_DIR / "retinaface_weight"
    
    # Performance tuning
    JPEG_QUALITY: int = 85  # 75=fastest, 85=balanced, 95=best quality (slower)
    NUM_WORKERS: int = DEFAULT_WORKERS  # Parallel workers (set to 1 to disable multiprocessing)
    MAX_VIDEOS_PER_CATEGORY: Optional[int] = None  # Limit for testing (None = all, 10 = test with 10 videos)
    
    # FaceForensics++ categories
    DATASET_CATEGORIES: Dict[str, str] = {
        'original': 'real',
        'DeepFakeDetection': 'fake',
        'Deepfakes': 'fake',
        'Face2Face': 'fake',
        'FaceShifter': 'fake',
        'FaceSwap': 'fake',
        'NeuralTextures': 'fake'
    }
    
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        extra="allow",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
settings = Settings()
    
    
    
    
