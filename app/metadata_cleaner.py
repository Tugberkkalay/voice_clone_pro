# app/metadata_cleaner.py
from pathlib import Path
import os
import time
from typing import Tuple

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_text_with_llm(raw_text: str) -> str:
    """
    Whisper çıkışı bir cümleyi LLM ile temizler:
    - Yazım ve noktalama düzelt
    - Türkçe karakterleri düzelt
    - 'eee, ııı, şey, yani' gibi doldurucu ifadeleri mümkün olduğunca kaldır
    - Anlamı koru, kısaltma/özetleme yapma
    """
    prompt = f"""
Aşağıdaki Türkçe cümleyi, sesli konuşma transkriptini eğitim için temizlemeni istiyorum.

Kurallar:
- Yazım ve noktalama hatalarını düzelt.
- Türkçe karakterleri düzelt (ı, ğ, ş, ö, ü, ç).
- "ııı", "eee", "şey", "yani", "hani" gibi tamamen doldurucu ifadeleri mümkün olduğunca kaldır.
- Cümlenin anlamını ve kelime dizilimini olabildiğince koru.
- Yeni bilgi ekleme, cümleyi özetleme veya kısaltma.
- Sayıları ve özel isimleri değiştirme.
- Sadece düzeltilmiş cümleyi döndür, açıklama yazma.

METİN:
\"\"\"{raw_text}\"\"\"    
    """.strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",  # veya hesabında hangisi uygunsa
        input=prompt,
    )

    cleaned = resp.output[0].content[0].text.strip()
    return cleaned


def clean_metadata_file(
    metadata_path: Path,
    out_path: Path | None = None,
    sleep_between: float = 0.2,
) -> Path:
    """
    metadata.csv içindeki tüm satırları LLM ile temizler.
    Girdi formatı:  audio/utt_0001.wav|Merhaba esra simdi sana guzel bir hikaye anlatacagim
    Çıktı formatı:  audio/utt_0001.wav|Merhaba Esra, şimdi sana güzel bir hikâye anlatacağım.

    Orijinal dosyaya dokunmaz, yeni bir .cleaned.csv oluşturur.
    """
    metadata_path = metadata_path.resolve()
    if out_path is None:
        out_path = metadata_path.with_suffix(".cleaned.csv")

    print(f"[INFO] Metadata temizleme başlıyor:")
    print(f"  Girdi : {metadata_path}")
    print(f"  Çıktı : {out_path}")

    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    cleaned_lines: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if "|" not in line:
            continue

        audio_rel, raw_text = line.split("|", 1)
        raw_text = raw_text.strip()

        if not raw_text:
            continue

        print(f"[{idx}/{len(lines)}] LLM temizliği: {audio_rel}")
        try:
            cleaned_text = clean_text_with_llm(raw_text)
        except Exception as e:
            print(f"[WARN] Satır temizlenemedi, orijinali kullanılıyor: {e}")
            cleaned_text = raw_text

        cleaned_lines.append(f"{audio_rel}|{cleaned_text}")
        time.sleep(sleep_between)  # oranı çok zorlamamak için

    out_path.write_text("\n".join(cleaned_lines), encoding="utf-8")
    print(f"[OK] Temizlenmiş metadata kaydedildi: {out_path}")
    return out_path
