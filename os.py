# app.py
import os
import io
import re
import datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st
import speech_recognition as sr
from together import Together

# Optional TTS toolings
ELEVEN_OK = True
try:
    from elevenlabs.client import ElevenLabs
    from pydub import AudioSegment, effects
except Exception:
    ELEVEN_OK = False

GTTS_OK = True
try:
    from gtts import gTTS
except Exception:
    GTTS_OK = False

# ---------------------------
# Config / Keys
# ---------------------------
st.set_page_config(page_title="AI Restaurant Assistant", layout="wide")
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY") or os.getenv("TOGETHER_API_KEY")
ELEVEN_API_KEY = st.secrets.get("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")

if not TOGETHER_API_KEY:
    st.warning("Set TOGETHER_API_KEY in st.secrets or env var to use the LLM.")
if not (ELEVEN_API_KEY or GTTS_OK):
    st.info("ElevenLabs key not found. gTTS fallback will be used if installed.")

# Initialize clients
llm = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY) if (ELEVEN_OK and ELEVEN_API_KEY) else None

ORDERS_FILE = "orders.xlsx"

# ---------------------------
# Helpers: order parsing & Excel
# ---------------------------
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def parse_order(text: str, menu_items: Optional[list] = None):
    """Extract simple order fields from text. Returns dict or None."""
    t = " ".join(text.lower().strip().split())
    # avoid parsing confirmations / small talk
    if any(k in t for k in ("confirm", "cancel", "thanks", "thank", "no")):
        return None

    # qty
    m = re.search(r"\b(\d{1,3})\b", t)
    qty = int(m.group(1)) if m else None
    if qty is None:
        for w,v in NUMBER_WORDS.items():
            if re.search(rf"\b{w}\b", t):
                qty = v
                break
    qty = qty or 1

    # price (look for ₹ or numbers near "each" or "price")
    price = None
    m_price = re.search(r"(?:₹|rs\.?|rupees\s*)\s*(\d{1,5}(?:\.\d{1,2})?)", t)
    if m_price:
        price = float(m_price.group(1))
    else:
        # fallback: any standalone number beyond qty
        all_nums = re.findall(r"\b(\d{1,5}(?:\.\d{1,2})?)\b", t)
        if len(all_nums) >= 2:
            price = float(all_nums[1])
        elif len(all_nums) == 1 and qty != int(all_nums[0]):
            price = float(all_nums[0])

    # customer: 'for <name>' or 'under <name>'
    customer = None
    m_for = re.search(r"\bfor\s+([a-z][a-z\s]{0,40})$", t)
    if m_for:
        customer = m_for.group(1).strip().title()
    else:
        m_under = re.search(r"under\s+(?:the\s+)?name\s+([a-z][a-z\s]{0,40})", t)
        if m_under:
            customer = m_under.group(1).strip().title()

    # size
    size = None
    m_size = re.search(r"\b(xl|extra large|large|medium|small|regular)\b", t)
    if m_size:
        size = m_size.group(1).title()

    # item detection: use menu.json if available
    item = None
    if os.path.exists("menu.json"):
        try:
            import json
            with open("menu.json","r",encoding="utf-8") as f:
                menu = json.load(f)
            names = []
            if isinstance(menu, list):
                if menu and isinstance(menu[0], dict) and "name" in menu[0]:
                    names = [it["name"].lower() for it in menu]
                else:
                    names = [str(x).lower() for x in menu]
            for n in names:
                if n in t:
                    item = n.title()
                    break
        except Exception:
            item = None

    # basic fallback: take words between qty and 'for' or price
    if not item:
        # index after qty
        start = 0
        mqty = re.search(r"\b(\d{1,3})\b", t)
        if mqty:
            start = mqty.end()
        # end before 'for' or price marker
        end = len(t)
        mfor = re.search(r"\bfor\b", t)
        if mfor:
            end = mfor.start()
        mprice = re.search(r"(?:₹|rs|rupees|\bprice\b)", t)
        if mprice:
            end = min(end, mprice.start())
        chunk = t[start:end].strip(", .")
        chunk = re.sub(r"\b(add|order|please|i'd|i will|i want|get|give|me|to|the|a|an)\b","", chunk).strip()
        if chunk:
            # keep first few words
            item = " ".join(chunk.split()[:5]).strip().title()

    if not item:
        return None

    price = float(price) if price else 0.0
    return {
        "Customer": customer or "",
        "Item": item,
        "Size": size or "",
        "Qty": int(qty),
        "PriceEach": float(price),
        "Total": round(float(price) * int(qty),2),
        "PlacedAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": "voice/chat"
    }

def append_order_excel(order: dict):
    """Append order row into ORDERS_FILE"""
    df_new = pd.DataFrame([order])
    if os.path.exists(ORDERS_FILE):
        df_old = pd.read_excel(ORDERS_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_excel(ORDERS_FILE, index=False)

# ---------------------------
# Helpers: LLM + TTS
# ---------------------------
def llm_reply(messages):
    """messages: list of {"role":..,"content":..} (system/user/assistant)"""
    if not llm:
        return "LLM not configured."
    try:
        resp = llm.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"

def tts_generate_bytes_eleven(text: str, voice_id: str, speed: float = 1.0) -> bytes:
    """Use ElevenLabs client to generate mp3 bytes, then optionally adjust speed with pydub"""
    if not tts_client:
        raise RuntimeError("ElevenLabs not configured")
    gen = tts_client.text_to_speech.convert(voice_id=voice_id, model_id="eleven_multilingual_v2", text=text)
    raw = b"".join(gen)
    if abs(speed - 1.0) > 1e-3:
        seg = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
        seg2 = effects.speedup(seg, playback_speed=max(0.5, min(2.0, speed)))
        out = io.BytesIO()
        seg2.export(out, format="mp3")
        return out.getvalue()
    return raw

def tts_generate_bytes_gtts(text: str, speed: float = 1.0) -> bytes:
    """Fallback using gTTS (then pydub speed if requested)"""
    if not GTTS_OK:
        raise RuntimeError("gTTS not available")
    buf = io.BytesIO()
    g = gTTS(text=text, lang="en")
    g.write_to_fp(buf)
    raw = buf.getvalue()
    if abs(speed - 1.0) > 1e-3:
        seg = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
        seg2 = effects.speedup(seg, playback_speed=max(0.6, min(1.6, speed)))
        out = io.BytesIO()
        seg2.export(out, format="mp3")
        return out.getvalue()
    return raw

def tts_bytes_with_fallback(text: str, voice_id: str, speed: float = 1.0) -> bytes:
    """Try ElevenLabs first, then gTTS fallback."""
    if tts_client:
        try:
            return tts_generate_bytes_eleven(text, voice_id, speed)
        except Exception as e:
            st.warning(f"ElevenLabs TTS failed: {e}. Falling back to gTTS (if available).")
    if GTTS_OK:
        try:
            return tts_generate_bytes_gtts(text, speed)
        except Exception as e:
            st.error(f"gTTS failed: {e}")
    return b""

# ---------------------------
# Speech-to-text (microphone)
# ---------------------------
def stt_from_mic(timeout=6, phrase_time_limit=10):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# ---------------------------
# UI: Sidebar & Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":"You are a helpful restaurant assistant. If the user provides an order, parse and ask for confirmation."}]
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending_order" not in st.session_state:
    st.session_state.pending_order = None

st.sidebar.header("Settings")
speak_replies = st.sidebar.checkbox("🔊 Speak assistant replies", value=True)
voice_id = st.sidebar.text_input("ElevenLabs Voice ID (or name)", value="SZfY4K69FwXus87eayHK")
speed = st.sidebar.slider("Voice speed", 0.7, 1.4, 1.05, 0.05)
st.sidebar.markdown("---")
if os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "rb") as f:
        st.sidebar.download_button("📥 Download orders.xlsx", f, file_name="orders.xlsx")

# ---------------------------
# Main layout: Mode selector
# ---------------------------
st.title("🍽️ AI Restaurant Assistant (Chat + Voice)")
mode = st.radio("Mode", ["Chat Mode", "Voice Mode"], horizontal=True)

# Chat input area (shared logic)
def handle_user_message(user_text: str):
    if not user_text:
        return
    # append user message to chat
    st.session_state.chat.append(("user", user_text))

    # attempt parsing order first (fast)
    parsed = parse_order(user_text)
    # check pending order confirmations
    low = user_text.lower()
    if st.session_state.pending_order:
        if "confirm" in low or low.strip() in ("yes","y","confirm"):
            append_order_excel(st.session_state.pending_order)
            st.success("✅ Order saved to orders.xlsx")
            st.session_state.pending_order = None
            st.session_state.chat.append(("assistant","Order confirmed and saved."))
            return
        if "cancel" in low or low.strip() in ("no","n","cancel"):
            st.session_state.pending_order = None
            st.info("Pending order cancelled.")
            st.session_state.chat.append(("assistant","Order cancelled."))
            return

    # if a parsed order exists, hold for confirmation
    if parsed:
        st.session_state.pending_order = parsed
        summary = (f"I detected an order: {parsed['Qty']} × {parsed['Item']} "
                   f"@ ₹{parsed['PriceEach']} each for {parsed['Customer'] or '—'}. "
                   "Please confirm (say/type 'confirm') or cancel.")
        st.session_state.chat.append(("assistant", summary))
        if speak_replies:
            audio = tts_bytes_with_fallback(summary, voice_id, speed)
            if audio:
                st.audio(audio, format="audio/mp3")
        return

    # otherwise use LLM for general reply
    # build messages for LLM using stored chat as list of dicts
    messages = [{"role": r, "content": c} for (r,c) in st.session_state.chat]
    try:
        reply = llm_reply(messages)
    except Exception as e:
        reply = f"LLM error: {e}"
    st.session_state.chat.append(("assistant", reply))
    if speak_replies:
        audio = tts_bytes_with_fallback(reply, voice_id, speed)
        if audio:
            st.audio(audio, format="audio/mp3")

# Render UI depending on mode
if mode == "Chat Mode":
    col1, col2 = st.columns([4,1])
    with col1:
        user_text = st.text_input("Type your message or order here", key="chat_input")
        if st.button("Send"):
            handle_user_message(user_text)
    with col2:
        if st.session_state.pending_order:
            st.markdown("### Pending Order")
            po = st.session_state.pending_order
            st.write(f"**{po['Qty']} × {po['Item']}**")
            st.write(f"Customer: {po['Customer'] or '—'}")
            st.write(f"Size: {po['Size'] or '—'}")
            st.write(f"Each: ₹{po['PriceEach']:.2f}")
            st.write(f"Total: ₹{po['Total']:.2f}")
            if st.button("✅ Confirm pending order"):
                append_order_excel(po)
                st.success("Order saved to orders.xlsx")
                st.session_state.pending_order = None
            if st.button("❌ Cancel pending order"):
                st.session_state.pending_order = None
                st.info("Order cancelled.")
else:
    # Voice Mode
    st.write("Press record and speak an order or question. The assistant will respond and (optionally) speak back.")
    if st.button("🎙️ Record (5s)"):
        spoken = stt_from_mic(timeout=6, phrase_time_limit=5)
        if not spoken:
            st.warning("No speech detected or STT failed.")
        else:
            st.write("You said:", spoken)
            handle_user_message(spoken)

# Chat history display
st.markdown("## Chat History")
for role, text in st.session_state.chat:
    if role == "user":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Assistant:** {text}")

# Manual Order Form tab below
st.markdown("---")
st.header("Manual Order Form")
with st.form("manual_order", clear_on_submit=True):
    name = st.text_input("Customer name")
    item = st.text_input("Item")
    size = st.selectbox("Size", ["", "Small", "Medium", "Large", "XL"])
    qty = st.number_input("Quantity", 1, 100, 1)
    price = st.number_input("Price each (₹)", 0.0, 99999.0, 0.0, 1.0)
    submitted = st.form_submit_button("Add order")
if submitted:
    order = {
        "Customer": name.title() if name else "",
        "Item": item.title(),
        "Size": size,
        "Qty": int(qty),
        "PriceEach": float(price),
        "Total": round(float(price) * int(qty),2),
        "PlacedAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": "form"
    }
    append_order_excel(order)
    st.success("Order saved to orders.xlsx")

# Orders viewer
st.header("Orders")
if os.path.exists(ORDERS_FILE):
    df = pd.read_excel(ORDERS_FILE)
    st.dataframe(df, use_container_width=True)
    with open(ORDERS_FILE, "rb") as f:
        st.download_button("📥 Download orders.xlsx", f, file_name="orders.xlsx")
else:
    st.info("No orders saved yet.")
