# app/dataset_builder.py
from pathlib import Path
from typing import Tuple, List

import whisper
from pydub import AudioSegment

from .config import VOICES_DIR, DATA_DIR
from .audio_preprocess import (
    list_audio_files,
    load_and_concat_files,
    basic_denoise_and_normalize,
    trim_leading_trailing_silence,
)
import ssl
# SADECE MODEL DOWNLOAD İÇİN: SSL doğrulamayı devre dışı bırak
ssl._create_default_https_context = ssl._create_unverified_context

def build_training_dataset_for_person(
    person_dir: Path,
    speaker_id: str,
    model_name: str = "medium",
    language: str = "tr",
) -> Path:
    """
    Bir kişi klasöründen (speakers/speaker_X) eğitim datası üretir.

    Adımlar:
    1) Tüm ses dosyalarını birleştir
    2) Denoise + normalize + sessizlik kırp
    3) Geçici tek bir long_wav olarak diske yaz
    4) Whisper ile transcribe et (segment segment)
    5) Her segment için küçük wav dosyası üret ve metadata.csv'ye yaz

    Dönüş: metadata.csv'nin yolu
    """
    # 1) Kişi seslerini yükle ve birleştir
    files = list_audio_files(person_dir)
    combined = load_and_concat_files(files, target_sr=24000)

    # 2) Temizleme
    cleaned = basic_denoise_and_normalize(combined)
    cleaned = trim_leading_trailing_silence(cleaned)

    # 3) Geçici long wav olarak kaydet
    tmp_dir = VOICES_DIR / f"{speaker_id}_training_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    long_wav_path = tmp_dir / "long_cleaned.wav"
    cleaned.export(long_wav_path, format="wav")

    # 4) Whisper modeli
    print(f"[INFO] Whisper modeli yükleniyor: {model_name}")
    asr_model = whisper.load_model(model_name)

    print(f"[INFO] Transkripsiyon başlıyor: {long_wav_path}")
    result = asr_model.transcribe(
        str(long_wav_path),
        language=language,
        verbose=False,
    )

    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("Whisper transkripsiyon sonucu segment içermiyor.")

    # 5) Eğitim datası klasör yapısı
    train_root = DATA_DIR / "training_data" / speaker_id
    audio_dir = train_root / "audio"
    train_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = train_root / "metadata.csv"

    # 6) Her ASR segmentini ayrı wav + metadata satırı olarak kaydet
    #    Not: segment start/end saniye cinsinden, pydub için ms'e çevireceğiz
    print(f"[INFO] {len(segments)} segment bulundu. Eğitim datası üretiliyor...")

    with metadata_path.open("w", encoding="utf-8") as mf:
        for idx, seg in enumerate(segments, start=1):
            start_s = seg["start"]
            end_s = seg["end"]
            text = seg["text"].strip()

            # Çok kısa segmentleri atla (< 1.5 sn gibi)
            if (end_s - start_s) < 1.5:
                continue

            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)

            audio_seg: AudioSegment = cleaned[start_ms:end_ms]

            # Küçük fade-in/out ile kesimleri yumuşat
            audio_seg = audio_seg.fade_in(10).fade_out(10)

            utt_name = f"utt_{idx:04d}.wav"
            utt_path = audio_dir / utt_name
            audio_seg.export(utt_path, format="wav")

            # metadata satırı: path|text
            rel_path = f"audio/{utt_name}"
            mf.write(f"{rel_path}|{text}\n")

    print(f"[OK] Eğitim datası hazır:")
    print(f"  - Kök klasör : {train_root}")
    print(f"  - metadata   : {metadata_path}")
    print(f"  - audio      : {audio_dir}")

    return metadata_path
