# app/tts_engine.py
from pathlib import Path
from functools import lru_cache
from typing import Literal, Union, List

import torch
from TTS.api import TTS

from .config import TTS_MODEL_NAME


LanguageCode = Literal[
    "tr",
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_tts() -> TTS:
    """
    XTTS-v2 modelini yükler.
    İlk çağrıda HuggingFace'ten indirir, sonra cache'den kullanır.
    """
    tts = TTS(TTS_MODEL_NAME, progress_bar=False).to(get_device())
    return tts


def synthesize_to_wav(
    text: str,
    speaker_wav: Union[Path, List[Path]],
    out_path: Path,
    language: LanguageCode = "tr",
) -> Path:
    """
    Metni, tek bir referans ya da çoklu referans segment kullanarak sese çevirir.
    """
    tts = get_tts()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(speaker_wav, list):
        speaker_arg = [str(p) for p in speaker_wav]
    else:
        speaker_arg = str(speaker_wav)

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_arg,
        language=language,
        file_path=str(out_path),
    )
    return out_path
