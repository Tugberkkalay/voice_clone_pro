# clean_metadata.py
from pathlib import Path

from app.metadata_cleaner import clean_metadata_file

def main():
    # Burayı kendi speaker klasörüne göre ayarla
    meta = Path("data/training_data/spk_bd53f4a2/metadata.csv")
    clean_metadata_file(meta)

if __name__ == "__main__":
    main()
