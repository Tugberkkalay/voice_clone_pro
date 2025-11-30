# app/config.py
import os
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_SPEAKERS_DIR = BASE_DIR / "speakers"

# Veri klasörleri
DATA_DIR = BASE_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Klasörleri oluştur
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# XTTS model adı
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Varsayılan dil
DEFAULT_LANGUAGE = "tr"

# OpenAI API anahtarı (environment'tan okunacak)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MIN_DURATION_SECONDS = 50.0