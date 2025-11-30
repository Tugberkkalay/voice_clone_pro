# main.py
from pathlib import Path
import sys

from app.config import RAW_SPEAKERS_DIR
from app.pipeline import enroll_from_person_folder, synthesize_with_voice


def select_person_folder() -> Path:
    speakers_root = RAW_SPEAKERS_DIR

    if not speakers_root.exists():
        print(f"[HATA] Speaker klasörü bulunamadı: {speakers_root}")
        sys.exit(1)

    person_dirs = [d for d in sorted(speakers_root.iterdir()) if d.is_dir()]

    if not person_dirs:
        print("Hiç kişi klasörü bulunamadı. 'speakers/' içine klasör ekle.")
        sys.exit(1)

    print("Kişi klasörleri:")
    for i, d in enumerate(person_dirs, start=1):
        print(f"  {i}) {d.name}")

    while True:
        choice = input(f"Hangi kişiyi kullanmak istiyorsun? (1-{len(person_dirs)}): ").strip()
        if not choice.isdigit():
            print("Sayı gir.")
            continue
        idx = int(choice)
        if not (1 <= idx <= len(person_dirs)):
            print("Aralık dışı.")
            continue
        return person_dirs[idx - 1]


def ask_text() -> str:
    text = input("Okutulacak metni yaz (boş = varsayılan): ").strip()
    if not text:
        text = "Seni çok seviyorum. Şimdi sana güzel bir hikâye anlatacağım."
        print(f"[Bilgi] Varsayılan metin kullanılacak:\n{text}\n")
    return text


def run_interactive():
    person_dir = select_person_folder()

    print("\n>>> Enroll başlıyor...\n")
    profile = enroll_from_person_folder(person_dir)

    print(f"Voice ID: {profile.voice_id}")
    print(f"Kişi klasörü: {profile.person_dir}")
    print(f"Toplam süre (sn): {profile.total_duration_sec:.2f}\n")

    print("Kullanılan referans segmentler:")
    for p in profile.speaker_wav_paths:
        print(f"  - {p}")
    print()

    text = ask_text()

    print("\n>>> Sentez başlıyor...\n")
    out_path = synthesize_with_voice(
        voice_id=profile.voice_id,
        text=text,
        language="tr",
    )

    print(f"[✓] Çıktı dosyası hazır: {out_path}")
    print("Bu dosyayı bir medya oynatıcı ile açıp klonu dinleyebilirsin.\n")


if __name__ == "__main__":
    run_interactive()
