# app/audio_preprocess.py
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment

from .config import VOICES_DIR, MIN_DURATION_SECONDS

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4")


def list_audio_files(person_dir: Path) -> List[Path]:
    """
    Kişi klasöründeki tüm desteklenen ses dosyalarını (mp3, mp4, m4a, wav...) listeler.
    """
    if not person_dir.exists():
        raise FileNotFoundError(f"Kişi klasörü bulunamadı: {person_dir}")

    files = [
        p for p in sorted(person_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"{person_dir} klasöründe desteklenen uzantıda ses dosyası bulunamadı."
        )

    return files


def load_and_concat_files(files: List[Path], target_sr: int = 24000) -> AudioSegment:
    """
    Verilen dosyaları sırayla okuyup tek bir AudioSegment halinde birleştirir.
    Hepsini mono + target_sr'e çevirir.
    """
    combined = None
    for f in files:
        seg = AudioSegment.from_file(f)
        seg = seg.set_frame_rate(target_sr).set_channels(1)
        if combined is None:
            combined = seg
        else:
            combined += seg
    return combined


def basic_denoise_and_normalize(audio: AudioSegment) -> AudioSegment:
    """
    Basit ama işe yarar bir temizlik:
    - High-pass filter ile 80 Hz altını kes (dip uğultu)
    - Low-pass filter ile 8000 Hz üstünü kes (çok tiz gürültü)
    - dBFS seviyesini sabitle (ör: -20 dBFS civarı)
    """
    audio = audio.high_pass_filter(80)
    audio = audio.low_pass_filter(8000)

    target_dbfs = -20.0
    change_in_dbfs = target_dbfs - audio.dBFS
    audio = audio.apply_gain(change_in_dbfs)

    return audio


def trim_leading_trailing_silence(
    audio: AudioSegment,
    silence_thresh_dbfs: float = -45.0,
    chunk_ms: int = 300,
) -> AudioSegment:
    """
    Baş ve sondaki uzun sessizlikleri kırpar.
    """
    if len(audio) <= chunk_ms:
        return audio

    # Baştan
    start_ms = 0
    while start_ms + chunk_ms < len(audio):
        chunk = audio[start_ms:start_ms + chunk_ms]
        if chunk.dBFS > silence_thresh_dbfs:
            break
        start_ms += chunk_ms

    # Sondan
    end_ms = len(audio)
    while end_ms - chunk_ms > start_ms + chunk_ms:
        chunk = audio[end_ms - chunk_ms:end_ms]
        if chunk.dBFS > silence_thresh_dbfs:
            break
        end_ms -= chunk_ms

    return audio[start_ms:end_ms]


def split_into_chunks(audio: AudioSegment, chunk_ms: int) -> List[AudioSegment]:
    chunks = []
    for start in range(0, len(audio), chunk_ms):
        end = min(start + chunk_ms, len(audio))
        chunks.append(audio[start:end])
    return chunks


def compute_speech_score(
    chunk: AudioSegment,
    silence_threshold_dbfs: float = -45.0,
    frame_ms: int = 200,
) -> float | None:
    """
    Bir chunk için 'konuşma skorunu' hesaplar.
    - Çok sessizse None döner
    - Orta-yüksek enerji + düşük sessizlik oranı = yüksek skor
    """
    if len(chunk) < frame_ms:
        return None

    frames = []
    for start in range(0, len(chunk), frame_ms):
        end = min(start + frame_ms, len(chunk))
        frame = chunk[start:end]
        frames.append(frame)

    if not frames:
        return None

    silent_frames = 0
    loudness_values = []

    for fr in frames:
        loud = fr.dBFS
        loudness_values.append(loud)
        if loud < silence_threshold_dbfs:
            silent_frames += 1

    avg_loudness = sum(loudness_values) / len(loudness_values)
    silence_ratio = silent_frames / len(frames)

    if silence_ratio > 0.7:
        # %70'ten fazlası sessizlikse konuşma sayma
        return None

    # Skor: daha yüksek ses + daha az sessizlik daha iyi
    score = avg_loudness - silence_ratio * 20.0
    return float(score)


def save_wav(audio: AudioSegment, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(out_path, format="wav")
    return out_path


def extract_speaker_segments(
    person_dir: Path,
    voice_id: str,
    segment_sec: float = 8.0,
    max_segments: int = 12,
) -> Tuple[List[Path], float]:
    """
    Aynı kişiye ait klasördeki tüm sesleri:
    1) Birleştirir
    2) Denoise + normalize + baş/son sessizliği kırpar
    3) 8s'lik chunk'lara böler
    4) Her chunk için 'konuşma skoru' hesaplar
    5) En iyi N chunk'ı ref_01.wav, ref_02.wav... olarak kaydeder
    """
    files = list_audio_files(person_dir)
    combined = load_and_concat_files(files, target_sr=24000)

    cleaned = basic_denoise_and_normalize(combined)
    cleaned = trim_leading_trailing_silence(cleaned)

    duration_sec = len(cleaned) / 1000.0
    if duration_sec < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Toplam süre çok kısa ({duration_sec:.2f} sn). "
            f"En az {MIN_DURATION_SECONDS} sn olmalı."
        )

    chunk_ms = int(segment_sec * 1000)
    chunks = split_into_chunks(cleaned, chunk_ms=chunk_ms)

    scored_chunks: List[tuple[float, int, AudioSegment]] = []
    for idx, ch in enumerate(chunks):
        score = compute_speech_score(ch)
        if score is None:
            continue
        scored_chunks.append((score, idx, ch))

    if not scored_chunks:
        raise RuntimeError("Konuşma içeren uygun segment bulunamadı.")

    # Skora göre sırala, en iyileri al
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:max_segments]

    voice_dir = VOICES_DIR / voice_id
    ref_paths: List[Path] = []

    for rank, (score, idx, ch) in enumerate(top, start=1):
        seg = ch.fade_in(30).fade_out(30)
        ref_path = voice_dir / f"ref_{rank:02d}.wav"
        save_wav(seg, ref_path)
        ref_paths.append(ref_path)

    return ref_paths, duration_sec
