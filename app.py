"""
ClarifyMed – AI Medical Visit Summarizer
Workflow:
  PDF upload + Audio upload
    → extract_pdf()          (pdfplumber, local)
    → transcribe_audio()     (OpenAI Whisper, Hebrew)
    → generate_summary()     (Google Gemini 1.5 Flash)
    → create_avatar()        (D-ID Talks API)
    → display final video
"""

import base64
import os
import time

import pdfplumber
import requests
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
DID_API_KEY      = os.getenv("DID_API_KEY")
AVATAR_IMAGE_URL = os.getenv("AVATAR_IMAGE_URL")

# ── Strict Gemini system prompt (Hebrew, 4 mandatory sections) ─────────────────
_SYSTEM_PROMPT = """אתה מסכם רפואי מקצועי. תפקידך לקרוא תמלול שיחה בין רופא למטופל וסיכום ביקור,
ולהפיק סיכום ברור ופשוט **בעברית** למטופל.

הסיכום חייב לכלול בדיוק את ארבעת הסעיפים הבאים:
1. **אבחנה עיקרית** – מה קבע הרופא?
2. **תרופות שנרשמו** – רשימה מלאה.
3. **הוראות שימוש בתרופות** – מינון, תדירות ומשך הטיפול לכל תרופה.
4. **המשך טיפול** – בדיקות, הפניות, ביקורי מעקב.

כתוב בשפה פשוטה למטופל שאינו רופא. השתמש רק במידע שסופק. אל תוסיף הסתייגויות משפטיות.

הסיכום צריך להיות קצר ומתאים לנאום בן 30-60 שניות (כ-100-200 מילים)."""


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 – PDF Text Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_pdf(pdf_file) -> str:
    """
    Extract all text from an uploaded PDF (Streamlit BytesIO-like file object).

    Returns:
        Full text content of the PDF as a single string.

    Raises:
        RuntimeError: If the PDF is empty, scanned (no embedded text), or unreadable.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages_text).strip()
    except Exception as e:
        raise RuntimeError(f"שגיאה בפתיחת ה-PDF: {e}")

    if not text:
        raise RuntimeError(
            "לא נמצא טקסט ב-PDF. ייתכן שזהו קובץ סרוק – "
            "נסה להמיר אותו לטקסט לפני ההעלאה."
        )
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 – Hebrew Audio Transcription (OpenAI Whisper)
# ══════════════════════════════════════════════════════════════════════════════
def transcribe_audio(audio_file) -> str:
    """
    Transcribe a Hebrew audio recording using OpenAI Whisper API.

    Supported formats: mp3, wav, m4a, ogg, flac, webm, mpeg (max 25 MB).
    Passes language='he' for better accuracy and lower latency.

    Returns:
        Hebrew transcript as a plain string.

    Raises:
        RuntimeError: On API error, unsupported format, or file-size violation.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY חסר ב-.env")

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        # Streamlit UploadedFile exposes .name; Whisper needs a (name, bytes, type) tuple
        audio_bytes = audio_file.read()
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.name, audio_bytes, audio_file.type),
            language="he",
            response_format="text",
        )
        return transcript
    except Exception as e:
        raise RuntimeError(f"שגיאה בתמלול Whisper: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 – Patient-Friendly Summarization (Gemini 1.5 Flash)
# ══════════════════════════════════════════════════════════════════════════════
def generate_summary(pdf_text: str, transcript: str) -> str:
    """
    Send PDF text + audio transcript to Gemini 1.5 Flash with a strict Hebrew
    medical summarizer system prompt.

    Returns:
        4-section patient-friendly Hebrew summary.

    Raises:
        RuntimeError: On API error or empty response.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY חסר ב-.env")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )

    if transcript:
        prompt = (
            "תמלול השיחה הרפואית:\n"
            "---\n"
            f"{transcript}\n"
            "---\n\n"
            "טקסט מסיכום הביקור (PDF):\n"
            "---\n"
            f"{pdf_text}\n"
            "---\n\n"
            "אנא הפק סיכום ידידותי למטופל בארבעת הסעיפים הנדרשים."
        )
    else:
        prompt = (
            "טקסט מסיכום הביקור (PDF):\n"
            "---\n"
            f"{pdf_text}\n"
            "---\n\n"
            "אנא הפק סיכום ידידותי למטופל בארבעת הסעיפים הנדרשים."
        )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"שגיאת Gemini: {e}")

    if not text:
        raise RuntimeError("Gemini החזיר תגובה ריקה – נסה שוב.")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 – Talking Doctor Avatar (D-ID API)
# ══════════════════════════════════════════════════════════════════════════════
_DID_BASE = "https://api.d-id.com"
_POLL_INTERVAL_SEC = 5
_MAX_POLLS = 36  # 36 × 5 s = 3 minutes max


def _did_auth_headers() -> dict:
    """Build D-ID Basic-auth headers (empty username, API key as password)."""
    token = base64.b64encode(DID_API_KEY.encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def create_avatar(summary_text: str) -> str:
    """
    Submit the Hebrew summary to D-ID Talks API and poll until the video is ready.

    Uses:
        - source_url: AVATAR_IMAGE_URL (public HTTPS JPEG/PNG of a face)
        - voice:      he-IL-HilaNeural (female Hebrew) – change to he-IL-AvriNeural for male

    Returns:
        Direct MP4 URL (result_url) ready for st.video() / download.

    Raises:
        RuntimeError: On HTTP errors, D-ID processing errors, or timeout.
    """
    if not DID_API_KEY:
        raise RuntimeError("DID_API_KEY חסר ב-.env")
    if not AVATAR_IMAGE_URL or AVATAR_IMAGE_URL.startswith("https://your-public"):
        raise RuntimeError(
            "AVATAR_IMAGE_URL לא הוגדר ב-.env. "
            "הכנס URL פומבי לתמונת פנים (HTTPS JPEG/PNG)."
        )

    headers = _did_auth_headers()
    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "text",
            "input": summary_text,
            "provider": {
                "type": "microsoft",
                "voice_id": "he-IL-HilaNeural",  # 👈 swap to he-IL-AvriNeural for male doctor
            },
        },
        "config": {"stitch": True},
    }

    # ── Step 1: Create the talk ──────────────────────────────────────────────
    try:
        r = requests.post(
            f"{_DID_BASE}/talks",
            json=payload,
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("פסק זמן (timeout) בשליחת הבקשה ל-D-ID – נסה שוב.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"D-ID שגיאת HTTP {e.response.status_code}: {e.response.text}"
        )

    talk_id = r.json().get("id")
    if not talk_id:
        raise RuntimeError(f"D-ID לא החזיר מזהה וידאו. תגובה: {r.text}")

    # ── Step 2: Poll for completion ──────────────────────────────────────────
    poll_url = f"{_DID_BASE}/talks/{talk_id}"
    for attempt in range(_MAX_POLLS):
        time.sleep(_POLL_INTERVAL_SEC)
        try:
            status_r = requests.get(poll_url, headers=headers, timeout=15)
            status_r.raise_for_status()
        except requests.exceptions.Timeout:
            continue  # transient timeout, keep polling
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"D-ID שגיאת polling {e.response.status_code}: {e.response.text}"
            )

        data = status_r.json()
        status = data.get("status")

        if status == "done":
            result_url = data.get("result_url")
            if not result_url:
                raise RuntimeError("D-ID סיים אך לא החזיר result_url.")
            return result_url

        if status == "error":
            err_detail = data.get("error", {})
            raise RuntimeError(f"D-ID נכשל ביצירת הוידאו: {err_detail}")

        # statuses 'created' / 'started' → keep waiting
    raise RuntimeError(
        f"פסק זמן: הוידאו לא הושלם תוך {_MAX_POLLS * _POLL_INTERVAL_SEC // 60} דקות. נסה שוב."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="ClarifyMed",
        page_icon="🩺",
        layout="centered",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🩺 ClarifyMed")
    st.subheader("הרופא המסביר שלך – AI")
    st.markdown(
        "העלה את הקלטת הביקור ואת סיכום הרופא, "
        "וקבל הסבר ברור ופשוט **בוידאו** מרופא AI."
    )
    st.divider()

    # ── File uploaders ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader(
            "📄 סיכום ביקור (PDF)",
            type=["pdf"],
            help="קובץ PDF שהרופא הדפיס / שלח לך.",
        )
    with col2:
        audio_file = st.file_uploader(
            "🎙️ הקלטת הביקור",
            type=["mp3", "wav", "m4a", "ogg", "flac", "webm", "mpeg"],
            help="הקלטה של השיחה עם הרופא (עד 25 MB).",
        )

    st.divider()

    if pdf_file and audio_file:
        st.info(
            f"📂 קבצים נטענו: **{pdf_file.name}** ו-**{audio_file.name}**  \n"
            "לחץ על הכפתור להתחלת העיבוד."
        )
    elif pdf_file:
        st.info(
            f"📄 סיכום PDF נטען: **{pdf_file.name}**  \n"
            "ניתן להוסיף הקלטה, או ללחוץ להמשיך עם הסיכום בלבד."
        )

    run_btn = st.button(
        "🚀 צור סיכום וידאו",
        disabled=not pdf_file,
        use_container_width=True,
        type="primary",
    )

    if run_btn:
        # ── Stage 1: PDF ──────────────────────────────────────────────────────
        with st.status("📄 שלב 1/4 – חילוץ טקסט מה-PDF...", expanded=False) as s1:
            try:
                pdf_text = extract_pdf(pdf_file)
                s1.update(label="✅ שלב 1/4 – PDF חולץ", state="complete")
            except RuntimeError as e:
                s1.update(label="❌ שלב 1/4 – כשל ב-PDF", state="error")
                st.error(str(e))
                st.stop()

        # ── Stage 2: Whisper (optional) ───────────────────────────────────────
        transcript = ""
        if audio_file:
            with st.status("🎙️ שלב 2/4 – תמלול הקלטה (Whisper)...", expanded=False) as s2:
                try:
                    transcript = transcribe_audio(audio_file)
                    s2.update(label="✅ שלב 2/4 – תמלול הושלם", state="complete")
                except RuntimeError as e:
                    s2.update(label="❌ שלב 2/4 – כשל בתמלול", state="error")
                    st.error(str(e))
                    st.stop()

            with st.expander("📝 צפה בתמלול המלא"):
                st.write(transcript)
        else:
            st.info("🎙️ שלב 2/4 – לא הועלתה הקלטה, ממשיך עם סיכום PDF בלבד.")

        # ── Stage 3: Gemini ───────────────────────────────────────────────────
        with st.status("🤖 שלב 3/4 – יצירת סיכום (Gemini)...", expanded=False) as s3:
            try:
                summary = generate_summary(pdf_text, transcript)
                s3.update(label="✅ שלב 3/4 – סיכום נוצר", state="complete")
            except RuntimeError as e:
                s3.update(label="❌ שלב 3/4 – כשל ב-Gemini", state="error")
                st.error(str(e))
                st.stop()

        with st.expander("📋 צפה בסיכום הרפואי"):
            st.markdown(summary)

        # ── Stage 4: D-ID ─────────────────────────────────────────────────────
        with st.status(
            "🎬 שלב 4/4 – יצירת וידאו עם רופא AI (D-ID)... (1–3 דקות)",
            expanded=False,
        ) as s4:
            try:
                video_url = create_avatar(summary)
                s4.update(label="✅ שלב 4/4 – וידאו מוכן!", state="complete")
            except RuntimeError as e:
                s4.update(label="❌ שלב 4/4 – כשל ב-D-ID", state="error")
                st.error(str(e))
                st.stop()

        # ── Output ────────────────────────────────────────────────────────────
        st.divider()
        st.success("🎉 הוידאו שלך מוכן!")
        st.video(video_url)
        st.markdown(
            f"[⬇️ הורד את הוידאו (MP4)]({video_url})",
            unsafe_allow_html=False,
        )


if __name__ == "__main__":
    main()
