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
    "Jawablah setiap soal. Soal disusun HOTS, bisa berupa pilihan ganda atau mencocokkan, sesuai mata pelajaran, jenjang, "
    "dan dikaitkan dengan cita-cita yang kamu tulis."
)

# ------------------------------------------------------------
# Gemini Client
# ------------------------------------------------------------

def build_prompt(subject: str, level: str, aspiration: str, n: int = 5) -> str:
    return f"""
Anda adalah generator bank soal untuk kuis pembelajaran adaptif.
Buat {n} soal yang menuntut HOTS (analyze/evaluate/create). Semua soal harus relevan dengan mata pelajaran "{subject}", jenjang "{level}", dan terhubung dengan cita-cita siswa: "{aspiration}".
Jenis soal hanya ada 2:
1. multiple_choice (pilihan ganda, 4 opsi A-D).
2. matching (mencocokkan, ada daftar kiri dan kanan yang harus dipasangkan).

Persyaratan:
- field "type" hanya bisa "multiple_choice" atau "matching".
- Jika multiple_choice: gunakan field "options" dan "answer" (salah satu A-D).
- Jika matching: gunakan field "pairs" berupa array objek {{"left":"..","right":".."}}, dan "answer" berupa array indeks jawaban benar, contoh: [2,0,1].
- Sertakan penjelasan jawaban di field "rationale".
- Tingkat HOTS wajib di field "hots".

Format keluaran: JSON array SAJA tanpa tambahan:
[
  {{
    "id": 1,
    "type": "multiple_choice" | "matching",
    "question": "Pertanyaan...",
    "options": ["A. ...","B. ...","C. ...","D. ..."],  // hanya multiple_choice
    "pairs": [{{"left":"...","right":"..."}},...],      // hanya matching
    "answer": "A" | [0,2,1],
    "rationale": "...",
    "hots": "Analyze|Evaluate|Create"
  }},
  ...
]
""".strip()


def call_gemini(api_key: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data)


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    cleaned = re.sub(r"^```[a-zA-Z]*", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned)
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Tidak menemukan JSON array.")
    block = re.sub(r",\s*]", "]", match.group(0))
    return json.loads(block)


def normalize_question(q: Dict[str, Any], idx: int) -> Dict[str, Any]:
    q = dict(q)
    q.setdefault("id", idx + 1)
    q.setdefault("type", "multiple_choice")
    q.setdefault("question", "")
    q.setdefault("rationale", "")
    q.setdefault("hots", "Analyze")

    if q["type"] == "multiple_choice":
        q.setdefault("options", ["A. ", "B. ", "C. ", "D. "])
        ans = str(q.get("answer", "A")).strip().upper()
        if ans not in {"A","B","C","D"}:
            ans = "A"
        q["answer"] = ans
    elif q["type"] == "matching":
        q.setdefault("pairs", [])
        if not isinstance(q.get("answer", []), list):
            q["answer"] = []
    return q


def generate_questions(api_key: str, subject: str, level: str, aspiration: str, n: int=5):
    prompt = build_prompt(subject, level, aspiration, n)
    raw = call_gemini(api_key, prompt)
    data = extract_json_array(raw)
    return [normalize_question(q, i) for i, q in enumerate(data[:n])]

# ------------------------------------------------------------
# UI State
# ------------------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Pengaturan Tes")
    subject = st.selectbox("Pilih Mata Pelajaran", SUBJECTS)
    level = st.selectbox("Pilih Jenjang", LEVELS)
    aspiration = st.text_input("Cita-cita Kamu", placeholder="mis. Dokter, Programmer…")
    st.markdown("---")
    api_key = st.text_input("API Key", value=os.getenv("GOOGLE_API_KEY",""), type="password")
    start = st.button("🚀 MULAI TES", type="primary", use_container_width=True)
    if start:
        if not api_key or not aspiration.strip():
            st.error("Mohon isi semua field.")
        else:
            with st.spinner("Menghasilkan soal dari AI…"):
                qs = generate_questions(api_key, subject, level, aspiration)
            st.session_state.questions = qs
            st.session_state.answers = [None]*len(qs)
            st.session_state.current_index = 0
            st.session_state.submitted = False
            st.success("Soal berhasil dibuat!")

# ------------------------------------------------------------
# Main Area
# ------------------------------------------------------------
st.title("Kenan AI - Your Dream Quiz 🧠✨")
st.write(DEFAULT_INSTRUCTIONS)

if not st.session_state.questions:
    st.info("Silakan mulai tes di sidebar.")
else:
    questions = st.session_state.questions
    idx = st.session_state.current_index
    total = len(questions)
    q = questions[idx]

    if not st.session_state.submitted:
        st.subheader(f"Soal {idx+1} / {total}")
        st.markdown(f"**HOTS:** {q.get('hots','')}")
        st.write(q.get("question",""))

        if q["type"] == "multiple_choice":
            options = q.get("options", [])
            letters = ["A","B","C","D"]
            current_answer = st.session_state.answers[idx]
            default_index = letters.index(current_answer) if current_answer in letters else None
            choice = st.radio("Pilih jawaban:", options, index=default_index, key=f"q_{idx}_radio")
            if choice:
                st.session_state.answers[idx] = choice.split(".")[0].strip()

        elif q["type"] == "matching":
            pairs = q.get("pairs", [])
            left_items = [p["left"] for p in pairs]
            right_items = [p["right"] for p in pairs]
            selected = []
            for i, left in enumerate(left_items):
                ans = st.selectbox(f"{left}", ["-"]+right_items, key=f"match_{idx}_{i}")
                selected.append(right_items.index(ans) if ans in right_items else None)
            st.session_state.answers[idx] = selected

        # Navigasi
        col1,col2,col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Sebelumnya", disabled=idx==0):
                st.session_state.current_index -= 1
                st.rerun()
        with col2:
            if st.button("➡️ Berikutnya", disabled=idx==total-1):
                st.session_state.current_index += 1
                st.rerun()
        with col3:
            if st.button("✅ Kumpulkan Jawaban", type="primary"):
                st.session_state.submitted = True
                st.rerun()

    else:
        st.success("Tes selesai! Berikut hasil kamu:")
        correct_flags = []
        for q, ans in zip(st.session_state.questions, st.session_state.answers):
            if q["type"] == "multiple_choice":
                correct_flags.append(1 if ans == q.get("answer") else 0)
            elif q["type"] == "matching":
                correct_flags.append(1 if ans == q.get("answer") else 0)
        score = round(sum(correct_flags)/len(correct_flags)*100,2)
        st.metric("Skor Akhir", f"{score}")

        st.markdown("---")
        for i,(q,ans,is_ok) in enumerate(zip(st.session_state.questions, st.session_state.answers, correct_flags)):
            st.markdown(f"### Soal {i+1} — {'✅ Benar' if is_ok else '❌ Salah'}")
            st.write(q.get("question",""))
            if q["type"] == "multiple_choice":
                for opt in q.get("options",[]):
                    letter = opt.split(".")[0].strip().upper()
                    if letter == q.get("answer"): st.write(f"- **{opt}** (Kunci)")
                    elif letter == (ans or ""): st.write(f"- ~~{opt}~~ (Jawaban Kamu)")
                    else: st.write(f"- {opt}")
            elif q["type"] == "matching":
                st.write("Pasangan yang benar:")
                for l,r in zip([p["left"] for p in q["pairs"]], [p["right"] for p in q["pairs"]]):
                    st.write(f"- {l} ⇔ {r}")
                st.write(f"Jawaban kamu: {ans}")
            st.write(f"**Alasan:** {q.get('rationale','')}")
            st.markdown("---")

        if st.button("🔁 Kerjakan Lagi"):
            st.session_state.questions=[]
            st.session_state.answers=[]
            st.session_state.current_index=0
            st.session_state.submitted=False
            st.rerun()
