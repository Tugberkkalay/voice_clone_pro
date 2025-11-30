# app/llm_cleaner.py
from typing import Optional

from openai import OpenAI

from .config import OPENAI_API_KEY


_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """OpenAI client'ı tekil (singleton gibi) oluşturur."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY environment değişkeni tanımlı değil.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def clean_text_for_tts(text: str, target_lang: str = "tr") -> str:
    """
    LLM ile TTS öncesi metin temizliği:
    - Emojileri, internet argosunu, 'loooool' vb. kaldır
    - Mümkünse sayıları yazıya çevir
    - Cümleleri düzgün noktalama ile bitir
    - Sadece temizlenmiş metni döndür
    """
    client = get_client()
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                f"Sen {target_lang} dilinde çalışan bir metin temizleme asistansın. "
                "Metni metin-okuma (TTS) için hazırlıyorsun.\n"
                "- Emojileri ve chat kısaltmalarını kaldır.\n"
                "- Gerekirse rakamları yazıyla ifade et.\n"
                "- Cümleleri düzgün noktalama ile bitir.\n"
                "- ÇIKTI OLARAK SADECE temizlenmiş metni ver."
            ),
            input=text,
        )
        cleaned = response.output_text
        return cleaned.strip()
    except Exception:
        # LLM'de hata olursa servisi çökertmemek için orijinal metni kullan.
        return text.strip()
