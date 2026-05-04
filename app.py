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

import asyncio
import base64
import io
import json
import os
import re
import shutil
import time
import edge_tts
import tempfile
import subprocess
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

# Windows fallback path for FFmpeg installed via WinGet
FFMPEG_BINARY_FALLBACK = (
    r"C:\Users\royme\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

# ── Strict Gemini system prompt (Hebrew, 4 mandatory sections) ─────────────────
_SYSTEM_PROMPT = """אתה רופא שמסביר למטופל את סיכום הביקור בצורה מדויקת וברורה.
חובה לפתוח כך:

שלום,

אני כאן כדי להסביר לך בצורה פשוטה וברורה את סיכום הביקור הרפואי שלך.

לאחר מכן המשך לסיכום.
חוקים קריטיים (אל תעבור עליהם):

1. נאמנות למקור:
- אסור להמציא מידע שלא מופיע בדוח
- אם משהו לא מצוין (למשל משך טיפול) → אל תנחש
- השתמש רק במה שכתוב

3. הסבר במקום העתקה:
- אל תעתיק משפטים מהדוח
- תסביר את המצב בצורה פשוטה
- לדוגמה:
  Cervicalgia → "כאבי צוואר"

4. טון:
- רגוע, ברור, מקצועי
- אם אין ממצא חמור → לציין שזה מצב שכיח/לא מסוכן

מבנה חובה:

שלום,

להלן סיכום הביקור שלך:

1. אבחנה עיקרית:
הסבר ברור מה יש למטופל ומה המשמעות.

2. תרופות שנרשמו:
רשימה בלבד.

3. הוראות שימוש:
רק מה שכתוב בפועל בדוח.

4. המשך טיפול:
רק מה שמופיע בדוח (למשל פיזיותרפיה).

חשוב מאוד:
- לא להוסיף פרטים שלא קיימים
- לא לנחש
- לא להמציא
- עדיף מידע חסר מאשר מידע שגוי
הסיכום צריך להתאים לנאום של 30–60 שניות (כ-100–200 מילים).
"אל תשתמש ב-Markdown. אל תשתמש בכוכביות **. אל תשתמש במספור 1,2,3,4. "
"הכותרות יהיו בטקסט רגיל בלבד."
"""


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 – PDF Text Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_pdf(pdf_file) -> str:
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

def extract_patient_name(text: str) -> str | None:
    match = re.search(r"שם משפחה ופרטי:\s*([^\n]+)", text)
    if match:
        full_name = match.group(1).strip()
        parts = full_name.split()
        if len(parts) >= 2:
            return parts[-1]  # מחזיר את השם הפרטי
    return None

def _is_rtl_text(text: str) -> bool:
    return bool(re.search(r'[֐-׿؀-ۿ]', text))


def clean_subtitles_text(text: str) -> str:
    # Remove markdown bold/italic markers
    text = re.sub(r"\*+", "", text)
    # Remove markdown headings (#, ##, …)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove lines that are purely section headers (≤50 chars, ends with colon)
    text = re.sub(r"^[^\n]{1,50}:\s*$", "", text, flags=re.MULTILINE)
    # Remove numbered list/section prefixes: 1. / 1) / 1.1 / 1.2.3 at line start
    text = re.sub(r"^\s*\d+(?:[.\)]\d*)+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.\)]\s+", "", text, flags=re.MULTILINE)
    # Remove bullet markers
    text = re.sub(r"^[-•*]\s+", "", text, flags=re.MULTILINE)
    # Remove trailing colons at end of lines
    text = re.sub(r":\s*$", "", text, flags=re.MULTILINE)
    # Collapse blank lines and strip
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines).strip()

# ══════════════════════════════════════════════════════════════════════════════
# Module 2 – Hebrew Audio Transcription (OpenAI Whisper)
# ══════════════════════════════════════════════════════════════════════════════
def transcribe_audio(audio_file) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY חסר ב-.env")

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
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
            "אנא הפק סיכום ידידותי למטופל. "
            "פתח בפתיח ונעים, ואז המשך לארבעת הסעיפים הנדרשים."
        )
    else:
        prompt = (
            "טקסט מסיכום הביקור (PDF):\n"
            "---\n"
            f"{pdf_text}\n"
            "---\n\n"
            "אנא הפק סיכום ידידותי למטופל. "
            "פתח בפתיח ונעים, ואז המשך לארבעת הסעיפים הנדרשים."
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
# Module 3b – Subtitle Translation (Gemini)
# ══════════════════════════════════════════════════════════════════════════════
_LANG_NAMES = {"he": "Hebrew", "en": "English", "ru": "Russian", "ar": "Arabic", "am": "Amharic"}

def translate_summary(text: str, target_lang: str) -> str:
    if target_lang == "he":
        return text
    if not GEMINI_API_KEY:
        return text
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"Translate the following Hebrew medical summary to {_LANG_NAMES[target_lang]}. "
        "Keep the numbered sections structure. Output only the translation, no extra commentary.\n\n"
        f"{text}"
    )
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 – Talking Doctor Avatar (D-ID API) + Audio Fallback
# ══════════════════════════════════════════════════════════════════════════════
_DID_BASE = "https://api.d-id.com"
_POLL_INTERVAL_SEC = 5
_MAX_POLLS = 36  # 36 × 5 s = 3 minutes max


def _did_auth_headers() -> dict:
    token = base64.b64encode(DID_API_KEY.encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _clean_for_tts(text: str) -> str:
    """Strip markdown so TTS doesn't read asterisks/hashes aloud."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_ffmpeg_binary() -> str:
    """Return the FFmpeg executable path, trying PATH first then the WinGet install."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    if os.path.isfile(FFMPEG_BINARY_FALLBACK):
        return FFMPEG_BINARY_FALLBACK
    raise RuntimeError(
        "FFmpeg לא נמצא. התקן FFmpeg (winget install Gyan.FFmpeg) "
        "או הוסף אותו ל-PATH."
    )


def run_ffmpeg(args: list) -> None:
    """Run FFmpeg with the given argument list. Raises RuntimeError with command + stderr on failure."""
    binary = get_ffmpeg_binary()
    cmd = [binary] + args
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            f"FFmpeg נכשל (exit {result.returncode})\n"
            f"פקודה: {cmd_str}\n"
            f"stderr:\n{result.stderr}"
        )


def get_media_duration(path: str) -> float | None:
    """Return media duration in seconds using ffprobe, or None on failure."""
    try:
        binary = get_ffmpeg_binary()
    except RuntimeError:
        return None
    probe = (
        binary.replace("ffmpeg.exe", "ffprobe.exe")
        if binary != "ffmpeg"
        else "ffprobe"
    )
    if not (shutil.which(probe) or os.path.isfile(probe)):
        return None
    try:
        result = subprocess.run(
            [probe, "-v", "quiet", "-print_format", "json", "-show_format", path],
            check=False, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        pass
    return None


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm (supports > 60 s correctly)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def ffmpeg_escape_subtitle_path(path: str) -> str:
    """Escape a filesystem path for use inside an FFmpeg filtergraph subtitles= value.

    FFmpeg's filter parser treats ':' as an option separator, so the drive-letter
    colon in Windows paths (C:/...) must be escaped as C\:/ .  The path is then
    wrapped in single quotes so the parser treats it as one token and does not
    mistake any remaining colon (e.g. in force_style) for another option.
    """
    path = path.replace("\\", "/")       # backslash → forward slash
    path = path.replace(":", "\\:", 1)   # C: → C\:  (first colon only)
    return path


def _make_subtitle_filter(srt_path: str) -> str:
    """Return a complete FFmpeg subtitles= filter string ready for -vf."""
    safe = ffmpeg_escape_subtitle_path(srt_path)
    style = (
        "FontSize=22,"
        "Alignment=2,"
        "Outline=2,"
        "Shadow=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000"
    )
    # Single quotes around the path prevent ':' from being parsed as option separator
    return f"subtitles='{safe}':force_style='{style}'"


# ══════════════════════════════════════════════════════════════════════════════
# Subtitle / video pipeline
# ══════════════════════════════════════════════════════════════════════════════

def create_srt_file(text: str, output_path: str, total_duration: float | None = None, early_sec: float = 0.4) -> None:
    """
    Write a valid .srt file from plain text.
    If total_duration is given, all timestamps are scaled so the last subtitle
    ends exactly at that duration — keeping subtitles in sync with the actual audio.
    early_sec shifts every subtitle earlier by this many seconds so captions
    appear just before the word is spoken rather than lagging behind.
    """
    raw_lines = re.split(r'\n+|(?<=[.!?])\s+', text.strip())
    raw_lines = [l.strip() for l in raw_lines if l.strip()]

    chunks: list[str] = []
    for line in raw_lines:
        while len(line) > 65:
            idx = line.rfind(' ', 0, 65)
            if idx == -1:
                idx = 65
            chunks.append(line[:idx].strip())
            line = line[idx:].strip()
        if line:
            chunks.append(line)

    if not chunks:
        return

    rtl = _is_rtl_text(text)
    base_durations = [max(2.5, len(chunk.split()) / 2.5) for chunk in chunks]
    gap = 0.1
    estimated_total = sum(base_durations) + gap * (len(chunks) - 1)
    scale = (total_duration / estimated_total) if total_duration and estimated_total > 0 else 1.0

    current = 0.0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (chunk, base_dur) in enumerate(zip(chunks, base_durations), start=1):
            duration = base_dur * scale
            start = max(0.0, current - early_sec)
            end = max(start + 0.1, current + duration - early_sec)
            current = current + duration + gap * scale
            display_chunk = ('\u200F' + chunk) if rtl else chunk
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(display_chunk + "\n\n")


def burn_subtitles_into_video(video_url: str, subtitle_text: str) -> str:
    """
    Download the D-ID MP4 from video_url, burn subtitles, and return a local path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        srt_path    = os.path.join(tmpdir, "subtitles.srt")
        output_video = os.path.join(tmpdir, "output.mp4")

        resp = requests.get(video_url, timeout=60)
        resp.raise_for_status()
        with open(input_video, "wb") as f:
            f.write(resp.content)

        video_duration = get_media_duration(input_video)
        create_srt_file(clean_subtitles_text(subtitle_text), srt_path, total_duration=video_duration)

        run_ffmpeg([
            "-y",
            "-i", input_video,
            "-vf", _make_subtitle_filter(srt_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_video,
        ])

        final_path = os.path.join(tempfile.gettempdir(), "clarifymed_video_with_subtitles.mp4")
        with open(output_video, "rb") as src, open(final_path, "wb") as dst:
            dst.write(src.read())

        return final_path


def create_static_video_with_audio_and_subtitles(audio_bytes: bytes, subtitle_text: str) -> str:
    """
    Create a fallback MP4: static doctor image + TTS audio + burned subtitles.
    No lip-sync, but a real playable video with sound and captions.
    """
    if not AVATAR_IMAGE_URL:
        raise RuntimeError("AVATAR_IMAGE_URL לא מוגדר — לא ניתן ליצור וידאו fallback.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download avatar image
        img_resp = requests.get(AVATAR_IMAGE_URL, timeout=30)
        img_resp.raise_for_status()
        img_path = os.path.join(tmpdir, "avatar.jpg")
        with open(img_path, "wb") as f:
            f.write(img_resp.content)

        # Save TTS audio
        audio_path = os.path.join(tmpdir, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Create SRT timed to the actual audio duration
        srt_path = os.path.join(tmpdir, "subtitles.srt")
        audio_duration = get_media_duration(audio_path)
        create_srt_file(clean_subtitles_text(subtitle_text), srt_path, total_duration=audio_duration)

        output_video = os.path.join(tmpdir, "output.mp4")

        # scale ensures even dimensions (H.264 requirement); subtitles filter follows
        vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,{_make_subtitle_filter(srt_path)}"

        run_ffmpeg([
            "-y",
            "-loop", "1", "-i", img_path,
            "-i", audio_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_video,
        ])

        final_path = os.path.join(tempfile.gettempdir(), "clarifymed_fallback_video.mp4")
        with open(output_video, "rb") as src, open(final_path, "wb") as dst:
            dst.write(src.read())

        return final_path


async def _edge_tts_bytes(text: str, voice: str = "he-IL-AvriNeural") -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate="-10%", pitch="-5Hz")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _audio_fallback(text: str, voice: str = "he-IL-AvriNeural") -> bytes:
    """Generate male audio with Microsoft Edge TTS (free fallback)."""
    return asyncio.run(_edge_tts_bytes(_clean_for_tts(text), voice))


def create_avatar(summary_text: str, lang: str = "he") -> str:
    """
    Submit the summary to D-ID Talks API and poll until the video is ready.

    Returns:
        Direct MP4 URL ready for downloading.

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

    voice_id = _VOICE_MAP.get(lang, "he-IL-AvriNeural")
    headers = _did_auth_headers()
    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "text",
            "input": _clean_for_tts(summary_text),
            "provider": {
                "type": "microsoft",
                "voice_id": voice_id,
                "voice_config": {"rate": "0.9"},
            },
        },
        "config": {"stitch": True, "subtitles": True},
    }

    try:
        r = requests.post(f"{_DID_BASE}/talks", json=payload, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("פסק זמן (timeout) בשליחת הבקשה ל-D-ID – נסה שוב.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"D-ID שגיאת HTTP {e.response.status_code}: {e.response.text}")

    talk_id = r.json().get("id")
    if not talk_id:
        raise RuntimeError(f"D-ID לא החזיר מזהה וידאו. תגובה: {r.text}")

    poll_url = f"{_DID_BASE}/talks/{talk_id}"
    for _ in range(_MAX_POLLS):
        time.sleep(_POLL_INTERVAL_SEC)
        try:
            status_r = requests.get(poll_url, headers=headers, timeout=15)
            status_r.raise_for_status()
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"D-ID שגיאת polling {e.response.status_code}: {e.response.text}")

        data   = status_r.json()
        status = data.get("status")

        if status == "done":
            result_url = data.get("result_url")
            if not result_url:
                raise RuntimeError("D-ID סיים אך לא החזיר result_url.")
            return result_url
        if status == "error":
            raise RuntimeError(f"D-ID נכשל ביצירת הוידאו: {data.get('error', {})}")

    raise RuntimeError(
        f"פסק זמן: הוידאו לא הושלם תוך {_MAX_POLLS * _POLL_INTERVAL_SEC // 60} דקות. נסה שוב."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800;900&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'Heebo', sans-serif !important;
    direction: rtl;
}

/* ── Background + dot grid ── */
.stApp {
    background: #060f22;
    background-image:
        radial-gradient(ellipse at 15% 30%, rgba(0,180,220,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 65%, rgba(0,80,190,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 5%,  rgba(10,50,130,0.10) 0%, transparent 40%);
}
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: radial-gradient(rgba(0,188,212,0.10) 1px, transparent 1px);
    background-size: 32px 32px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 800px !important; }

/* ── Floating icons ── */
.bg-icons { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.bg-icon  { position: absolute; opacity: 0.045; animation: floatIcon 20s ease-in-out infinite; }
.bg-icon:nth-child(1)  { top: 4%;  left: 6%;  font-size:2.0rem; animation-delay:0s;   }
.bg-icon:nth-child(2)  { top:10%;  left:78%;  font-size:3.2rem; animation-delay:2s;   }
.bg-icon:nth-child(3)  { top:25%;  left:18%;  font-size:1.8rem; animation-delay:4s;   }
.bg-icon:nth-child(4)  { top:33%;  left:90%;  font-size:2.6rem; animation-delay:1s;   }
.bg-icon:nth-child(5)  { top:52%;  left:4%;   font-size:2.8rem; animation-delay:3s;   }
.bg-icon:nth-child(6)  { top:60%;  left:62%;  font-size:2.1rem; animation-delay:5s;   }
.bg-icon:nth-child(7)  { top:75%;  left:28%;  font-size:2.4rem; animation-delay:2.5s; }
.bg-icon:nth-child(8)  { top:83%;  left:84%;  font-size:1.9rem; animation-delay:6s;   }
.bg-icon:nth-child(9)  { top:44%;  left:48%;  font-size:1.7rem; animation-delay:1.5s; }
.bg-icon:nth-child(10) { top:91%;  left:14%;  font-size:3.0rem; animation-delay:4.5s; }
@keyframes floatIcon {
    0%,100% { transform:translateY(0)   rotate(0deg);  }
    33%     { transform:translateY(-20px) rotate(9deg); }
    66%     { transform:translateY(12px)  rotate(-6deg);}
}

/* ── Hero ── */
.cm-hero { position:relative; text-align:center; padding:3.2rem 2rem 2rem; z-index:1; }
.cm-logo-ring {
    width:92px; height:92px;
    background: linear-gradient(145deg, #0055a5, #00bcd4);
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:2.5rem;
    margin:0 auto 1.2rem;
    box-shadow: 0 0 0 14px rgba(0,188,212,0.07), 0 0 0 28px rgba(0,188,212,0.03);
    animation: pulse 3.5s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { box-shadow:0 0 0 14px rgba(0,188,212,0.07),0 0 0 28px rgba(0,188,212,0.03); }
    50%     { box-shadow:0 0 0 20px rgba(0,188,212,0.11),0 0 0 40px rgba(0,188,212,0.05); }
}
.cm-hero h1 {
    font-size:3.4rem; font-weight:900; letter-spacing:-1.5px; margin:0 0 0.6rem;
    background: linear-gradient(135deg, #ffffff 0%, #b8e8ff 45%, #00d4f0 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.cm-hero p { color:rgba(170,210,240,0.75); font-size:1.1rem; font-weight:300; margin:0; }
.cm-pill {
    display:inline-block;
    background:rgba(0,188,212,0.10); border:1px solid rgba(0,188,212,0.28);
    color:#6dd6ea; border-radius:50px;
    padding:0.22rem 1.1rem; font-size:0.76rem; font-weight:600;
    letter-spacing:1.2px; text-transform:uppercase; margin-bottom:1.1rem;
}

/* ── Glow divider ── */
.cm-glow-line {
    height:1px; margin:1.8rem 0;
    background:linear-gradient(90deg, transparent 0%, rgba(0,188,212,0.45) 50%, transparent 100%);
    position:relative; overflow:hidden;
}
.cm-glow-line::after {
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg, transparent, rgba(0,220,255,0.9), transparent);
    animation:shimmer 3.5s ease-in-out infinite;
}
@keyframes shimmer { to { left:100%; } }

/* ── Section label ── */
.cm-section-label {
    color:rgba(100,180,220,0.7); font-size:0.78rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;
}

/* ── Upload ── */
.cm-upload-title {
    color:#90c8e8; font-size:0.88rem; font-weight:600;
    letter-spacing:0.4px; margin-bottom:0.45rem;
}
section[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,0.03) !important;
    border:1.5px dashed rgba(0,188,212,0.30) !important;
    border-radius:14px !important; transition:all 0.2s;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    background:rgba(0,188,212,0.05) !important;
    border-color:rgba(0,188,212,0.65) !important;
}
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] p { color:rgba(140,190,225,0.65) !important; }
button[data-testid="baseButton-secondary"] {
    background:rgba(0,188,212,0.10) !important; border:1px solid rgba(0,188,212,0.35) !important;
    color:#7ec8e3 !important; border-radius:8px !important;
}

/* ── Status bar ── */
.cm-status-bar {
    background:rgba(0,188,212,0.07); border:1px solid rgba(0,188,212,0.18);
    border-radius:12px; padding:0.75rem 1.2rem;
    color:#90d8f0; font-size:0.88rem; text-align:center; margin:0.6rem 0;
}

/* ── Language selector ── */
.cm-lang-label {
    color:#90c8e8; font-size:0.82rem; font-weight:700;
    letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.6rem;
    display:flex; align-items:center; gap:0.5rem;
}
div[data-testid="stRadio"] > label { color:#90c8e8 !important; font-size:0.9rem !important; }
div[data-testid="stRadio"] > div   { gap:0.5rem !important; }
div[data-testid="stRadio"] > div > label {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(0,188,212,0.22) !important;
    border-radius:10px !important; padding:0.4rem 0.9rem !important;
    color:#a8d8f0 !important; font-size:0.9rem !important;
    transition:all 0.18s !important; cursor:pointer !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background:rgba(0,188,212,0.18) !important;
    border-color:rgba(0,188,212,0.6) !important;
    color:#ffffff !important; font-weight:600 !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg, #005fa3, #00bcd4) !important;
    border:none !important; border-radius:14px !important;
    font-size:1.12rem !important; font-weight:800 !important;
    padding:0.85rem 2rem !important; color:white !important;
    letter-spacing:0.4px; box-shadow:0 6px 30px rgba(0,188,212,0.28);
    transition:transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 12px 40px rgba(0,188,212,0.42) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:disabled {
    background:rgba(255,255,255,0.05) !important;
    box-shadow:none !important; color:rgba(255,255,255,0.25) !important;
}

/* ── Status / expanders ── */
div[data-testid="stStatus"] {
    background:rgba(255,255,255,0.03) !important;
    border:1px solid rgba(0,188,212,0.18) !important;
    border-radius:12px !important; color:#90d0ef !important;
}
details {
    background:rgba(255,255,255,0.025) !important;
    border:1px solid rgba(0,188,212,0.13) !important;
    border-radius:12px !important; color:#c0e0f5 !important;
}
summary { color:#7ec8e3 !important; }

/* ── Result card ── */
.cm-result {
    background:rgba(255,255,255,0.035);
    backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
    border:1px solid rgba(0,188,212,0.18);
    border-radius:22px; padding:2.5rem 2rem;
    text-align:center;
    box-shadow:0 10px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-top:1.5rem;
}
.cm-result-title {
    font-size:1.65rem; font-weight:900; margin-bottom:0.35rem;
    background:linear-gradient(135deg, #eaf5ff, #7ec8e3);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.cm-result-sub { color:rgba(150,195,230,0.6); font-size:0.92rem; margin-bottom:1.4rem; }
.cm-avatar-ring {
    width:195px; height:195px; border-radius:50%; object-fit:cover;
    border:3px solid rgba(0,188,212,0.55);
    box-shadow:0 0 0 10px rgba(0,188,212,0.07), 0 0 50px rgba(0,188,212,0.18);
    margin:0 auto 1.4rem; display:block;
    animation:pulse 3.5s ease-in-out infinite;
}
.cm-download-btn {
    display:inline-block;
    background:rgba(0,188,212,0.10); border:1px solid rgba(0,188,212,0.32);
    color:#7ec8e3 !important; border-radius:10px;
    padding:0.5rem 1.6rem; font-size:0.88rem;
    text-decoration:none; margin-top:1rem; transition:background 0.2s;
}
.cm-download-btn:hover { background:rgba(0,188,212,0.20) !important; }

/* ── Subtitle card ── */
.cm-subtitle {
    background:rgba(0,188,212,0.04);
    border:1px solid rgba(0,188,212,0.18);
    border-radius:16px; padding:1.4rem 1.6rem; margin-top:1.4rem; text-align:right;
}
.cm-subtitle-header {
    display:flex; align-items:center; gap:0.6rem;
    font-size:0.78rem; font-weight:700; letter-spacing:1.4px;
    text-transform:uppercase; color:#6dd6ea; margin-bottom:0.9rem;
}
.cm-subtitle-body {
    color:#b8ddf2; font-size:0.95rem; line-height:1.8; white-space:pre-wrap;
}
.cm-subtitle-body[dir="ltr"] { direction:ltr; text-align:left; }

/* ── Alerts ── */
div[data-testid="stInfo"]    { background:rgba(0,140,200,0.08)!important; border:1px solid rgba(0,188,212,0.22)!important; border-radius:12px!important; color:#90d0ef!important; }
div[data-testid="stWarning"] { background:rgba(255,170,0,0.07)!important; border:1px solid rgba(255,170,0,0.22)!important; border-radius:12px!important; color:#ffd890!important; }
div[data-testid="stSuccess"] { background:rgba(0,200,110,0.07)!important; border:1px solid rgba(0,200,110,0.22)!important; border-radius:12px!important; color:#90f0c0!important; }
div[data-testid="stError"]   { background:rgba(220,50,50,0.09)!important; border:1px solid rgba(220,50,50,0.28)!important; border-radius:12px!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#060f22; }
::-webkit-scrollbar-thumb { background:rgba(0,188,212,0.28); border-radius:10px; }

/* ── Gradient blobs ── */
.cm-blobs { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.cm-blob  { position:absolute; border-radius:50%; filter:blur(80px); pointer-events:none; animation:blobPulse 12s ease-in-out infinite; }
.cm-blob-tl {
    width:560px; height:560px; top:-220px; left:-180px;
    background:radial-gradient(circle at 40% 40%, rgba(0,188,212,0.55), rgba(0,85,165,0.40), transparent 70%);
    opacity:0.22;
}
.cm-blob-br {
    width:500px; height:500px; bottom:-180px; right:-150px;
    background:radial-gradient(circle at 60% 60%, rgba(124,58,237,0.50), rgba(30,64,175,0.45), transparent 70%);
    opacity:0.20; animation-delay:5s;
}
.cm-blob-ml {
    width:280px; height:280px; top:35%; left:5%;
    background:radial-gradient(circle, rgba(6,182,212,0.45), transparent 65%);
    opacity:0.13; animation-delay:2.5s;
}
.cm-blob-mr {
    width:320px; height:320px; top:20%; right:5%;
    background:radial-gradient(circle, rgba(99,102,241,0.40), transparent 65%);
    opacity:0.14; animation-delay:7s;
}
@keyframes blobPulse {
    0%,100% { transform:scale(1)    translate(  0px,  0px); }
    25%     { transform:scale(1.06) translate(-15px, 20px); }
    50%     { transform:scale(1.02) translate( 10px,-10px); }
    75%     { transform:scale(0.97) translate( -5px, 15px); }
}

/* ── Side illustration panels ── */
.cm-side-panel {
    position:fixed; top:50%; transform:translateY(-50%);
    z-index:1; pointer-events:none; opacity:0.32;
}
.cm-side-panel-left  { left:26px;  animation:sideFloat 7s ease-in-out infinite; }
.cm-side-panel-right { right:26px; animation:sideFloat 7s ease-in-out infinite reverse; }
@keyframes sideFloat {
    0%,100% { transform:translateY(-50%); }
    50%     { transform:translateY(calc(-50% - 18px)); }
}
@media (max-width:1060px) { .cm-side-panel { display:none; } }
</style>
"""

_BG_ICONS_HTML = """
<div class="bg-icons">
  <span class="bg-icon">🩺</span><span class="bg-icon">❤️</span>
  <span class="bg-icon">💊</span><span class="bg-icon">🧬</span>
  <span class="bg-icon">🏥</span><span class="bg-icon">🩻</span>
  <span class="bg-icon">💉</span><span class="bg-icon">🔬</span>
  <span class="bg-icon">🩺</span><span class="bg-icon">🧪</span>
</div>
"""

_SIDE_DECOR_HTML = """
<div class="cm-blobs" aria-hidden="true">
  <div class="cm-blob cm-blob-tl"></div>
  <div class="cm-blob cm-blob-br"></div>
  <div class="cm-blob cm-blob-ml"></div>
  <div class="cm-blob cm-blob-mr"></div>
</div>

<div class="cm-side-panel cm-side-panel-left" aria-hidden="true">
  <svg width="90" height="350" viewBox="0 0 90 350" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M45,0 C75,17 75,53 45,70 C15,87 15,123 45,140 C75,157 75,193 45,210 C15,227 15,263 45,280 C75,297 75,333 45,350"
          stroke="#00d4f0" stroke-width="2.2"/>
    <path d="M45,0 C15,17 15,53 45,70 C75,87 75,123 45,140 C15,157 15,193 45,210 C75,227 75,263 45,280 C15,297 15,333 45,350"
          stroke="#7c3aed" stroke-width="2.2"/>
    <line x1="24" y1="18"  x2="66" y2="18"  stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="52"  x2="66" y2="52"  stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="88"  x2="66" y2="88"  stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="122" x2="66" y2="122" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="158" x2="66" y2="158" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="192" x2="66" y2="192" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="228" x2="66" y2="228" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="262" x2="66" y2="262" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="298" x2="66" y2="298" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <line x1="24" y1="332" x2="66" y2="332" stroke="rgba(180,220,255,0.45)" stroke-width="1.4"/>
    <circle cx="45" cy="70"  r="3.5" fill="#00d4f0"/>
    <circle cx="45" cy="140" r="3.5" fill="#7c3aed"/>
    <circle cx="45" cy="210" r="3.5" fill="#00d4f0"/>
    <circle cx="45" cy="280" r="3.5" fill="#7c3aed"/>
  </svg>
</div>

<div class="cm-side-panel cm-side-panel-right" aria-hidden="true">
  <svg width="80" height="350" viewBox="0 0 80 350" fill="none" xmlns="http://www.w3.org/2000/svg">
    <line x1="20" y1="0"   x2="20" y2="350" stroke="rgba(0,188,212,0.10)" stroke-width="1"/>
    <line x1="60" y1="0"   x2="60" y2="350" stroke="rgba(0,188,212,0.10)" stroke-width="1"/>
    <line x1="0"  y1="87"  x2="80" y2="87"  stroke="rgba(0,188,212,0.10)" stroke-width="1"/>
    <line x1="0"  y1="175" x2="80" y2="175" stroke="rgba(0,188,212,0.10)" stroke-width="1"/>
    <line x1="0"  y1="263" x2="80" y2="263" stroke="rgba(0,188,212,0.10)" stroke-width="1"/>
    <path d="M40,0 L40,35 Q54,46 40,57 L40,66 L28,73 L72,83 L28,93 L40,99
             Q57,114 40,129 L40,175
             Q54,186 40,197 L40,206 L28,213 L72,223 L28,233 L40,239
             Q57,254 40,269 L40,350"
          stroke="#00d4f0" stroke-width="2.2"/>
    <circle cx="72" cy="83"  r="3.5" fill="#00d4f0"/>
    <circle cx="72" cy="223" r="3.5" fill="#00d4f0"/>
  </svg>
</div>
"""

_LANG_OPTIONS = {
    "he": ("🇮🇱", "עברית"),
    "en": ("🇺🇸", "English"),
    "ru": ("🇷🇺", "Русский"),
    "ar": ("🇸🇦", "العربية"),
    "am": ("🇪🇹", "አማርኛ"),
}

# Microsoft Neural TTS voice IDs (male) per language — used for both D-ID and edge_tts fallback
_VOICE_MAP = {
    "he": "he-IL-AvriNeural",
    "en": "en-US-GuyNeural",
    "ru": "ru-RU-DmitryNeural",
    "ar": "ar-SA-HamedNeural",
    "am": "am-ET-AmehaNeural",
}


def main():
    st.set_page_config(page_title="ClarifyMed", page_icon="🩺", layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_BG_ICONS_HTML, unsafe_allow_html=True)
    st.markdown(_SIDE_DECOR_HTML, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cm-hero">
      <div class="cm-pill">✦ &nbsp; פלטפורמה רפואית מבוססת AI</div>
      <div class="cm-logo-ring">🩺</div>
      <h1>ClarifyMed</h1>
      <p>מעלים סיכום ביקור והקלטה — מקבלים הסבר קולי ברור עם רופא AI</p>
    </div>
    <div class="cm-glow-line"></div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="cm-section-label">📂 &nbsp;העלאת קבצים</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="cm-upload-title">📄 &nbsp;סיכום ביקור (PDF)</div>', unsafe_allow_html=True)
        pdf_file = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")
    with col2:
        st.markdown('<div class="cm-upload-title">🎙️ &nbsp;הקלטת הביקור <span style="opacity:0.45;font-weight:400">(אופציונלי)</span></div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("audio", type=["mp3","wav","m4a","ogg","flac","webm","mpeg"], label_visibility="collapsed")

    if pdf_file and audio_file:
        st.markdown(f'<div class="cm-status-bar">✅ &nbsp;מוכן: <strong>{pdf_file.name}</strong> + <strong>{audio_file.name}</strong></div>', unsafe_allow_html=True)
    elif pdf_file:
        st.markdown(f'<div class="cm-status-bar">📄 &nbsp;<strong>{pdf_file.name}</strong> — ניתן להמשיך בלי הקלטה</div>', unsafe_allow_html=True)

    st.markdown('<div class="cm-glow-line"></div>', unsafe_allow_html=True)

    # ── Subtitle language picker ──────────────────────────────────────────────
    st.markdown('<div class="cm-section-label">🌐 &nbsp;שפת כתוביות</div>', unsafe_allow_html=True)
    lang_labels = [f"{flag} {name}" for flag, name in _LANG_OPTIONS.values()]
    lang_keys   = list(_LANG_OPTIONS.keys())
    chosen_idx  = st.radio("lang", lang_labels, index=0, horizontal=True, label_visibility="collapsed")
    subtitle_lang = lang_keys[lang_labels.index(chosen_idx)]

    st.markdown('<div class="cm-glow-line"></div>', unsafe_allow_html=True)

    run_btn = st.button("✨  צור סיכום קולי עם רופא AI", disabled=not pdf_file,
                        use_container_width=True, type="primary")
    if not run_btn:
        return

    st.write("")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    with st.status("📄  שלב 1 / 4 — חילוץ טקסט מה-PDF...", expanded=False) as s1:
        try:
            pdf_text = extract_pdf(pdf_file)
            s1.update(label="✅  שלב 1 / 4 — PDF חולץ", state="complete")
        except RuntimeError as e:
            s1.update(label="❌  שלב 1 / 4 — כשל ב-PDF", state="error")
            st.error(str(e)); st.stop()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    transcript = ""
    if audio_file:
        with st.status("🎙️  שלב 2 / 4 — תמלול הקלטה (Whisper AI)...", expanded=False) as s2:
            try:
                transcript = transcribe_audio(audio_file)
                s2.update(label="✅  שלב 2 / 4 — תמלול הושלם", state="complete")
            except RuntimeError as e:
                s2.update(label="❌  שלב 2 / 4 — כשל בתמלול", state="error")
                st.error(str(e)); st.stop()
        with st.expander("📝  צפה בתמלול המלא"):
            st.write(transcript)
    else:
        st.info("🎙️  שלב 2 / 4 — ממשיך עם סיכום PDF בלבד.")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    with st.status("🤖  שלב 3 / 4 — יצירת סיכום רפואי (Gemini AI)...", expanded=False) as s3:
        try:
            summary = generate_summary(pdf_text, transcript)
            s3.update(label="✅  שלב 3 / 4 — סיכום נוצר", state="complete")
        except RuntimeError as e:
            s3.update(label="❌  שלב 3 / 4 — כשל ב-Gemini", state="error")
            st.error(str(e)); st.stop()

    with st.expander("📋  צפה בסיכום הרפואי המלא"):
        st.markdown(summary)

    # ── Translate before avatar so D-ID speaks + subtitles in selected language ──
    flag, lang_name = _LANG_OPTIONS[subtitle_lang]
    with st.spinner(f"מתרגם ל-{lang_name}...") if subtitle_lang != "he" else st.empty():
        avatar_text = translate_summary(summary, subtitle_lang)
    voice = _VOICE_MAP.get(subtitle_lang, "he-IL-AvriNeural")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    video_url        = None
    audio_bytes      = None
    final_video_path = None

    with st.status("🎬  שלב 4 / 4 — יצירת וידאו עם רופא AI (D-ID)...", expanded=False) as s4:
        try:
            video_url = create_avatar(avatar_text, subtitle_lang)
            s4.update(label="✅  שלב 4 / 4 — וידאו מוכן!", state="complete")
        except RuntimeError as e:
            if "402" in str(e) or "credits" in str(e).lower():
                s4.update(label="✅  שלב 4 / 4 — סיכום קולי מוכן (מצב חינמי)", state="complete")
                audio_bytes = _audio_fallback(avatar_text, voice)
            else:
                s4.update(label="❌  שלב 4 / 4 — כשל ב-D-ID", state="error")
                st.error(str(e)); st.stop()

    # ── Burn subtitles into D-ID video ────────────────────────────────────────
    if video_url:
        with st.status("🎞️  מטמיע כתוביות בסרטון...", expanded=False) as sv:
            try:
                final_video_path = burn_subtitles_into_video(video_url, avatar_text)
                sv.update(label="✅  כתוביות הוטמעו בסרטון", state="complete")
            except RuntimeError as e:
                sv.update(label="⚠️  כתוביות נכשלו — מציג סרטון ללא כתוביות", state="error")
                st.error(f"שגיאת FFmpeg (כתוביות): {e}")
                # serve the raw D-ID URL as a fallback for the download path
                final_video_path = None

    # ── Build fallback static video (image + audio + subtitles) ───────────────
    if audio_bytes and not video_url:
        with st.status("🎞️  יוצר וידאו עם תמונת רופא + כתוביות...", expanded=False) as sfv:
            try:
                final_video_path = create_static_video_with_audio_and_subtitles(audio_bytes, avatar_text)
                sfv.update(label="✅  וידאו fallback מוכן", state="complete")
            except RuntimeError as e:
                sfv.update(label="⚠️  לא ניתן ליצור וידאו — מציג אודיו בלבד", state="error")
                st.error(f"שגיאת FFmpeg (fallback): {e}")

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown('<div class="cm-result">', unsafe_allow_html=True)

    if final_video_path and os.path.isfile(final_video_path):
        if video_url:
            st.markdown('<div class="cm-result-title">הוידאו שלך מוכן</div>', unsafe_allow_html=True)
            st.markdown('<div class="cm-result-sub">הרופא המדבר מסביר את הסיכום הרפואי שלך — כתוביות מוטמעות בסרטון</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cm-result-title">הסיכום הקולי שלך מוכן</div>', unsafe_allow_html=True)
            st.markdown('<div class="cm-result-sub">תמונת רופא + הקלטה קולית + כתוביות מוטמעות</div>', unsafe_allow_html=True)

        st.video(final_video_path)
        with open(final_video_path, "rb") as f:
            file_name = "clarifymed_video_with_subtitles.mp4" if video_url else "clarifymed_fallback_video.mp4"
            st.download_button(
                "⬇️ הורד וידאו עם כתוביות",
                data=f,
                file_name=file_name,
                mime="video/mp4",
            )

    elif audio_bytes:
        # Last-resort: FFmpeg unavailable, show audio + static image only
        if AVATAR_IMAGE_URL:
            st.markdown(f'<img class="cm-avatar-ring" src="{AVATAR_IMAGE_URL}" alt="רופא AI"/>', unsafe_allow_html=True)
        st.markdown('<div class="cm-result-title">הסיכום הקולי שלך מוכן</div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-result-sub">לחץ על ▶ להאזנה לסיכום הרפואי</div>', unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/mp3")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
