# build_dataset.py
from pathlib import Path
import uuid

from app.config import RAW_SPEAKERS_DIR
from app.dataset_builder import build_training_dataset_for_person


def main():
    # Burada speaker klasörünü manuel seçiyoruz
    person_dir = RAW_SPEAKERS_DIR / "speaker_1"  # örnek
    speaker_id = f"spk_{uuid.uuid4().hex[:8]}"

    print(f"[INFO] Dataset üretimi başlıyor.")
    print(f"  - Kişi klasörü : {person_dir}")
    print(f"  - Speaker ID   : {speaker_id}")

    metadata_path = build_training_dataset_for_person(
        person_dir=person_dir,
        speaker_id=speaker_id,
        model_name="medium",   # istersen "small" ile başlarsın
        language="tr",
    )

    print(f"[DONE] metadata.csv -> {metadata_path}")


if __name__ == "__main__":
    main()
