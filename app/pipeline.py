# app/pipeline.py
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from .config import OUTPUTS_DIR
from .audio_preprocess import extract_speaker_segments
from .tts_engine import synthesize_to_wav, LanguageCode
# İstersen sonra açarız:
# from .llm_cleaner import clean_text_for_tts


@dataclass
class VoiceProfile:
    voice_id: str
    person_dir: Path
    speaker_wav_paths: List[Path]   # çoklu referans segmentler
    total_duration_sec: float


# Basit bir registry: bu session içinde oluşturulan profilleri tutuyoruz
VOICE_REGISTRY: Dict[str, VoiceProfile] = {}


def enroll_from_person_folder(person_dir: Path) -> VoiceProfile:
    """
    Aynı kişiye ait bir klasörden (içinde mp3/mp4/m4a/wav vb.)
    voice profile oluşturur.

    - Tüm kayıtları birleştirir
    - Denoise + normalize + sessizlik kırpma yapar
    - 8 sn'lik chunk'lara böler
    - Konuşma skoruna göre en iyi N chunk'ı seçer
    - Bu segmentleri XTTS referansı olarak saklar
    """
    voice_id = str(uuid.uuid4())

    ref_paths, total_dur = extract_speaker_segments(
        person_dir=person_dir,
        voice_id=voice_id,
        segment_sec=8.0,   # her segment ~8 saniye
        max_segments=12,   # en iyi 12 segmenti al (toplam ~1.5–2 dk referans)
    )

    profile = VoiceProfile(
        voice_id=voice_id,
        person_dir=person_dir,
        speaker_wav_paths=ref_paths,
        total_duration_sec=total_dur,
    )

    VOICE_REGISTRY[voice_id] = profile
    return profile


def synthesize_with_voice(
    voice_id: str,
    text: str,
    language: LanguageCode = "tr",
) -> Path:
    """
    Verilen voice_id profili ile metni okutur.
    - (İstersek) metni önce LLM ile temizleyebiliriz
    - XTTS-v2'yi, çoklu referans segment ile kullanır
    """
    if voice_id not in VOICE_REGISTRY:
        raise ValueError(f"Geçersiz voice_id: {voice_id}")

    profile = VOICE_REGISTRY[voice_id]

    # Şimdilik LLM temizliğini kapalı tutalım, OpenAI tarafı ayrı stabil olunca açarız.
    # cleaned_text = clean_text_for_tts(text, target_lang=language)
    cleaned_text = text

    out_id = str(uuid.uuid4())
    out_path = OUTPUTS_DIR / f"{voice_id}_{out_id}.wav"

    # 🔥 ÖNEMLİ: Artık BİR DOSYA DEĞİL, ÇOKLU REFERANS veriyoruz
    synthesize_to_wav(
        text=cleaned_text,
        speaker_wav=profile.speaker_wav_paths,
        out_path=out_path,
        language=language,
    )

    return out_path
