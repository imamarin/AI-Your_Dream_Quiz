import os
import json
import re
from typing import List, Dict, Any

import requests
import streamlit as st

# ------------------------------------------------------------
# Config & Helpers
# ------------------------------------------------------------
st.set_page_config(page_title="AI Kuis Cita-cita", page_icon="🎯", layout="wide")

SUBJECTS = [
    "Matematika",
    "IPA",
    "IPS",
    "Bahasa Indonesia",
    "Bahasa Inggris",
    "PPKn",
    "Informatika",
    "Seni Budaya",
]

LEVELS = ["SD", "SMP", "SMA"]

DEFAULT_INSTRUCTIONS = (
    "Jawablah setiap soal. Soal disusun HOTS dan/atau berbasis gambar sesuai mata pelajaran, jenjang, "
    "dan dikaitkan dengan cita-cita yang kamu tulis."
)

# ------------------------------------------------------------
# Gemini (Google Generative Language API) Client
# ------------------------------------------------------------

def build_prompt(subject: str, level: str, aspiration: str, n: int = 5) -> str:
    """
    Membangun prompt agar model hanya mengembalikan JSON Array daftar soal.
    Format yang diminta sangat ketat untuk memudahkan parsing.
    """
    return f"""
Anda adalah generator bank soal untuk kuis pembelajaran adaptif.
Buat {n} soal pilihan ganda (4 opsi: A, B, C, D) yang menuntut HOTS (analyze/evaluate/create) Jangan terlalu panjang dan/atau menggunakan gambar.
Semua soal harus relevan dengan mata pelajaran "{subject}", jenjang "{level}", dan terhubung dengan cita-cita siswa: "{aspiration}".

Persyaratan:
- Minimal 1 dari {n} soal bertipe "image" (menggunakan gambar kontekstual). Jika sulit, boleh gunakan ilustrasi/infografik yang tersedia online.
- Sisa soal bertipe "text".
- Untuk soal bertipe "image", sertakan field "image_url" yang MERUJUK ke gambar publik (mis. Wikimedia, situs pendidikan, atau domain bebas pakai). Pastikan URL langsung ke file gambar (jpg/png/webp) jika memungkinkan.
- Gunakan konteks cita-cita untuk mempersonalisasi skenario soal (misal: dokter, guru, programmer, arsitek, polisi, wirausaha, desainer, atlet, dsb.).
- Hindari data pribadi atau konten sensitif. Hindari gambar berhak cipta non-bebas.
- Tingkat kognitif HOTS (misal: Analyze/Evaluate/Create) dapat dicantumkan di field "hots".

Format KELUARAN WAJIB berupa JSON Array SAJA (tanpa catatan tambahan, tanpa markdown, tanpa backticks),
dengan skema:
[
  {{
    "id": 1,
    "type": "text" | "image",
    "question": "Pertanyaan yang kontekstual dan menantang",
    "image_url": "https://..."  // WAJIB ADA jika type == "image", jika tidak ada, kosongkan string ""
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A",  // salah satu dari "A"/"B"/"C"/"D"
    "rationale": "Alasan kenapa jawabannya benar, jelaskan logikanya",
    "hots": "Analyze|Evaluate|Create"
  }},
  ... total {n} objek ...
]

Ingat: Keluarkan HANYA JSON array yang valid.
""".strip()


def call_gemini(api_key: str, prompt: str) -> str:
    """Panggil Gemini 2.5 Flash via REST. Kembalikan teks mentah respons model."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json", "X-goog-api-key:":api_key}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Struktur respons typical: { "candidates": [ { "content": { "parts": [ {"text": "..."} ] } } ] }
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # Jika format tidak sesuai, tampilkan seluruh json agar bisa didiagnosis
        return json.dumps(data)


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Ekstrak JSON array dari teks (menghapus code fences dan teks di luar JSON)."""
    # Hilangkan code fences ```json ... ``` jika ada
    cleaned = re.sub(r"^```[a-zA-Z]*", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned)

    # Cari blok array JSON pertama
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Tidak menemukan JSON array dalam respons model.")
    block = match.group(0)

    # Perbaiki trailing koma sederhana
    block = re.sub(r",\s*]", "]", block)

    try:
        data = json.loads(block)
        if not isinstance(data, list):
            raise ValueError("Root bukan array.")
        return data
    except json.JSONDecodeError as e:
        # Coba perbaikan kutip yang tidak konsisten (opsional)
        raise ValueError(f"Gagal parse JSON dari model: {e}\nPotongan: {block[:500]}")


def normalize_question(q: Dict[str, Any], idx: int) -> Dict[str, Any]:
    q = dict(q)
    q.setdefault("id", idx + 1)
    q.setdefault("type", "text")
    q.setdefault("question", "")
    q.setdefault("image_url", "")
    q.setdefault("options", [])
    q.setdefault("answer", "A")
    q.setdefault("rationale", "")
    q.setdefault("hots", "Analyze")

    # Pastikan opsi dalam format ["A. ...","B. ...","C. ...","D. ..."]
    opts = q.get("options", [])
    if isinstance(opts, dict):
        # Jika model memberi dict seperti {"A":"..."} ubah ke list
        mapped = [opts.get(k, "") for k in ["A", "B", "C", "D"]]
        q["options"] = [f"{ch}. {txt}" for ch, txt in zip(["A", "B", "C", "D"], mapped)]
    elif isinstance(opts, list):
        # Tambah label jika belum ada
        normalized = []
        letters = ["A", "B", "C", "D"]
        for i, item in enumerate(opts[:4]):
            s = str(item)
            if not s.strip().upper().startswith((letters[i] + ".")):
                s = f"{letters[i]}. {s}"
            normalized.append(s)
        # Pad jika kurang dari 4
        while len(normalized) < 4:
            i = len(normalized)
            normalized.append(f"{letters[i]}. ")
        q["options"] = normalized
    else:
        q["options"] = ["A. ", "B. ", "C. ", "D. "]

    # Jawaban hanya huruf
    ans = str(q.get("answer", "A")).strip().upper()
    if ans not in {"A", "B", "C", "D"}:
        # Coba deteksi dari teks opsi yang bertanda (BENAR)
        detected = None
        for letter, opt in zip(["A", "B", "C", "D"], q["options"]):
            if "(BENAR)" in opt.upper() or "(CORRECT)" in opt.upper():
                detected = letter
                break
        ans = detected or "A"
    q["answer"] = ans

    # Bersih-bersih image_url jika type != image
    if q["type"].lower() != "image":
        q["image_url"] = ""

    return q


def generate_questions(api_key: str, subject: str, level: str, aspiration: str, n: int = 5) -> List[Dict[str, Any]]:
    prompt = build_prompt(subject, level, aspiration, n)
    raw = call_gemini(api_key, prompt)
    data = extract_json_array(raw)
    questions = [normalize_question(q, i) for i, q in enumerate(data[:n])]
    # Jika kurang dari n, pad dengan dummy
    while len(questions) < n:
        i = len(questions)
        questions.append(
            normalize_question(
                {
                    "id": i + 1,
                    "type": "text",
                    "question": f"Soal cadangan #{i+1}: (dummy)",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                    "rationale": "",
                    "hots": "Analyze",
                },
                i,
            )
        )
    return questions


# ------------------------------------------------------------
# UI State
# ------------------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions: List[Dict[str, Any]] = []
if "answers" not in st.session_state:
    st.session_state.answers: List[str | None] = []
if "current_index" not in st.session_state:
    st.session_state.current_index: int = 0
if "submitted" not in st.session_state:
    st.session_state.submitted: bool = False


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Pengaturan Tes")
    subject = st.selectbox("Pilih Mata Pelajaran", SUBJECTS, index=0)
    level = st.selectbox("Pilih Jenjang", LEVELS, index=1)
    aspiration = st.text_input("Cita-cita Kamu", placeholder="mis. Dokter, Programmer, Arsitek…")

    st.markdown("---")
    st.caption("Kunci API Google Generative Language (Gemini)")
    # Ambil dari secrets/env bila ada
    default_api = st.secrets.get("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")) if hasattr(st, "secrets") else os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("API Key", value=default_api, type="password", help="Masukkan API key untuk gemini-2.5-flash")

    start = st.button("🚀 MULAI TES", type="primary", use_container_width=True)

    if start:
        if not api_key:
            st.error("Mohon isi API Key terlebih dahulu.")
        elif not aspiration.strip():
            st.error("Mohon isi cita-cita terlebih dahulu.")
        else:
            try:
                with st.spinner("Menghasilkan soal dari AI…"):
                    qs = generate_questions(api_key, subject, level, aspiration, n=5)
                st.session_state.questions = qs
                st.session_state.answers = [None] * len(qs)
                st.session_state.current_index = 0
                st.session_state.submitted = False
                st.success("Soal berhasil dibuat! Selamat mengerjakan ✨")
            except Exception as e:
                st.error(f"Gagal menghasilkan soal: {e}")


# ------------------------------------------------------------
# Main Area
# ------------------------------------------------------------
st.title("AI Kuis Cita-cita Siswa 🧠✨")
st.write(DEFAULT_INSTRUCTIONS)

if not st.session_state.questions:
    st.info("Silakan atur pengaturan di sidebar lalu klik **MULAI TES** untuk memulai.")
else:
    questions = st.session_state.questions
    idx = st.session_state.current_index
    total = len(questions)

    if not st.session_state.submitted:
        # Navigasi & Tampilan Soal
        q = questions[idx]
        st.subheader(f"Soal {idx+1} / {total}")
        st.markdown(f"**HOTS:** {q.get('hots', '')}")
        st.write(q.get("question", ""))

        if q.get("type", "text").lower() == "image" and q.get("image_url"):
            try:
                st.image(q["image_url"], caption="Perhatikan gambar berikut", use_container_width=True)
            except Exception:
                st.warning("Gambar tidak dapat dimuat. Lanjutkan dengan memahami deskripsi soal.")

        # Opsi Jawaban
        letters = ["A", "B", "C", "D"]
        options = q.get("options", ["A. ", "B. ", "C. ", "D. "])
        # Tampilkan sebagai radio
        current_answer = st.session_state.answers[idx]
        default_index = letters.index(current_answer) if current_answer in letters else None
        choice = st.radio(
            "Pilih jawaban:",
            options,
            index=default_index,
            key=f"q_{idx}_radio",
        )
        # Simpan hanya hurufnya (A/B/C/D)
        if choice:
            chosen_letter = choice.split(".")[0].strip()
            st.session_state.answers[idx] = chosen_letter

        # Tombol navigasi
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Sebelumnya", disabled=idx == 0, use_container_width=True):
                st.session_state.current_index = max(0, idx - 1)
                st.rerun()
        with col2:
            if st.button("➡️ Berikutnya", disabled=idx == total - 1, use_container_width=True):
                st.session_state.current_index = min(total - 1, idx + 1)
                st.rerun()
        with col3:
            all_answered = all(a in {"A", "B", "C", "D"} for a in st.session_state.answers)
            if st.button("✅ Kumpulkan Jawaban", type="primary", disabled=not all_answered, use_container_width=True):
                st.session_state.submitted = True
                st.rerun()

    else:
        # Hasil & Review
        st.success("Tes selesai! Berikut hasil kamu:")
        letters = ["A", "B", "C", "D"]
        correct_flags = []
        for i, (q, ans) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
            correct_flags.append(1 if ans == q.get("answer", "A").upper() else 0)
        score = round(sum(correct_flags) / len(correct_flags) * 100, 2)
        st.metric(label="Skor Akhir", value=f"{score}")

        st.markdown("---")
        st.subheader("Review Soal & Jawaban Kamu")
        for i, (q, ans, is_ok) in enumerate(zip(st.session_state.questions, st.session_state.answers, correct_flags)):
            st.markdown(f"### Soal {i+1} — {'✅ Benar' if is_ok else '❌ Salah'}")
            st.markdown(f"**HOTS:** {q.get('hots','')}")
            st.write(q.get("question", ""))
            if q.get("type","text").lower() == "image" and q.get("image_url"):
                try:
                    st.image(q["image_url"], caption="Gambar pada soal", use_container_width=True)
                except Exception:
                    pass

            # Tampilkan opsi lagi dengan penanda jawaban
            options = q.get("options", [])
            correct = q.get("answer", "A").upper()
            for opt in options:
                letter = opt.split(".")[0].strip().upper()
                if letter == correct:
                    st.write(f"- **{opt}** (Kunci)")
                elif letter == (ans or "").upper():
                    st.write(f"- ~~{opt}~~ (Jawaban Kamu)")
                else:
                    st.write(f"- {opt}")

            st.markdown(f"**Alasan/Kunci:** {q.get('rationale','')}")
            st.markdown("---")

        if st.button("🔁 Kerjakan Lagi", use_container_width=True):
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.current_index = 0
            st.session_state.submitted = False
            st.rerun()
