# ═══════════════════════════════════════════════════════════════
#  MedAI  —  Smart Health Assistant
#  Groq Vision + PDF + Blood Report + First Aid + Medicine
# ═══════════════════════════════════════════════════════════════
import streamlit as st
from groq import Groq
from PIL import Image
import base64, re, fitz
from io import BytesIO

# ───────────────────────────────────────────────────────────────
#  CONFIG  ← your Groq API key
# ───────────────────────────────────────────────────────────────
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"
client       = Groq(api_key=GROQ_API_KEY)

# ───────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
#  MINIMAL CSS  (no uploader hacks — let Streamlit render it)
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

/* ── App background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* ── Sidebar radio labels ── */
div[data-testid="stRadio"] label { color: #8b949e !important; font-size: 13px !important; }
div[data-testid="stRadio"] label[aria-checked="true"] {
    color: #58a6ff !important; font-weight: 600 !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    text-align: left !important;
}
.stButton > button:hover {
    border-color: #58a6ff !important;
    color: #ffffff !important;
}

/* ── File uploader  ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] button {
    background: #1f6feb !important;
    border-radius: 9px !important;
}

/* ── User chat bubble ── */
.user-bubble {
    background: #1f3a6e;
    border: 1px solid #2a4a8a;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 6px 0 6px auto;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
    color: #e6edf3;
}

/* ── AI chat bubble ── */
.ai-bubble {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    margin: 6px auto 6px 0;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.75;
    color: #e6edf3;
}
.ai-bubble h3 {
    font-size: 13.5px;
    font-weight: 700;
    color: #58a6ff;
    margin: 14px 0 6px;
    padding-bottom: 5px;
    border-bottom: 1px solid #21262d;
}
.ai-bubble h3:first-child { margin-top: 0; }
.ai-bubble ul { padding-left: 20px; margin: 6px 0; }
.ai-bubble li { margin-bottom: 5px; }
.ai-bubble strong { color: #79c0ff; }
.ai-bubble em     { color: #8b949e; font-style: italic; }

/* Status badges */
.b-ok  { background:#0d4429; color:#3fb950; border:1px solid #238636;
          padding:1px 8px; border-radius:5px; font-size:11px; font-weight:700; }
.b-hi  { background:#3d1f1f; color:#f85149; border:1px solid #da3633;
          padding:1px 8px; border-radius:5px; font-size:11px; font-weight:700; }
.b-lo  { background:#3d2e00; color:#f0883e; border:1px solid #d29922;
          padding:1px 8px; border-radius:5px; font-size:11px; font-weight:700; }
.b-brd { background:#2d2000; color:#e3b341; border:1px solid #bb8009;
          padding:1px 8px; border-radius:5px; font-size:11px; font-weight:700; }

/* Severity label */
.sev-minor    { color:#3fb950; font-weight:700; }
.sev-moderate { color:#e3b341; font-weight:700; }
.sev-severe   { color:#f85149; font-weight:700; }

/* Scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:4px; }

/* Divider */
hr { border-color: #21262d !important; margin: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
#  SESSION STATE
# ───────────────────────────────────────────────────────────────
defaults = {
    "messages":      [],
    "mode":          "General",
    "pend_b64":      None,
    "pend_name":     None,
    "pend_type":     None,   # "image" | "pdf"
    "pend_pdf_text": None,
    "prev_upname":   "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def extract_pdf(raw: bytes) -> str:
    doc  = fitz.open(stream=raw, filetype="pdf")
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    return text.strip()

def pdf_preview_b64(raw: bytes) -> str:
    doc = fitz.open(stream=raw, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return pil_to_b64(img)

def md2html(text: str) -> str:
    """Convert markdown-like AI response to HTML for the bubble."""
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'^## (.+)$',  r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'^# (.+)$',   r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', text)
    text = re.sub(r'\bNORMAL\b',     '<span class="b-ok">✓ NORMAL</span>',      text)
    text = re.sub(r'\bHIGH\b',       '<span class="b-hi">↑ HIGH</span>',        text)
    text = re.sub(r'\bLOW\b',        '<span class="b-lo">↓ LOW</span>',         text)
    text = re.sub(r'\bBORDERLINE\b', '<span class="b-brd">⚠ BORDERLINE</span>', text)
    text = re.sub(r'^[-•] (.+)$',    r'<li>\1</li>', text, flags=re.M)
    text = re.sub(r'^(\d+)\. (.+)$', r'<li><b>\1.</b> \2</li>', text, flags=re.M)
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text

# ───────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ───────────────────────────────────────────────────────────────
PROMPTS = {

"General": """You are MedAI, a friendly and knowledgeable medical AI assistant.
Help users with medicines, blood reports, injuries, skin conditions, and general health.
Be clear and empathetic. Always remind users to consult a real doctor for diagnosis.
Use ### headings and bullet points. Keep responses focused and practical.
IMPORTANT: If an image is provided, ALWAYS look at it first and describe what you see before answering.""",

"Medicine Info": """You are a medicine information expert AI.
When given a medicine name OR image of medicine packaging:
### 💊 Medicine Name & Type
### 🎯 What It Treats (conditions/diseases)
### ⚙️ How It Works (simple explanation)
### 📋 Dosage & How to Take
### ⚠️ Common Side Effects
### 🚫 Who Should Avoid It / Precautions
### 🔗 Drug Interactions (important ones)
### 🏷️ Common Brand Names in India
End with: *Always consult your doctor or pharmacist before taking any medicine.*""",

"Body Checkup": """You are a blood test and medical report analysis AI.
When given a report via image, PDF, or typed text:
1. Extract EVERY test parameter visible
2. For each: value · normal reference range · NORMAL / HIGH / LOW / BORDERLINE
3. Explain what each abnormal value means in plain simple language
### 📊 Complete Test-by-Test Analysis  (list every single parameter)
### 🔴 Abnormal Values Summary  (only flagged ones with explanation)
### 🥗 Diet & Lifestyle Changes  (specific to the findings)
### 👨‍⚕️ Doctor / Specialist Recommendation
Be thorough. Flag every abnormal value clearly.
End with: *Please share this report with your doctor for a proper diagnosis.*""",

"First Aid": """You are a certified first aid expert AI.

When the user uploads an IMAGE — look at it carefully and:
1. Identify exactly what injury or condition is shown (burn, cut, rash, bruise, wound, swelling, fracture, skin infection, insect bite, etc.)
2. Assess the visible severity from the image
3. Give targeted, specific first aid for what you actually see

ALWAYS respond with ALL these sections:

### 🔍 What I Can See
(Describe the injury/condition visible in the image clearly — burn area, wound size, redness, swelling, etc.)

### 🚨 Severity Assessment
(Minor 🟢 / Moderate 🟡 / Severe 🔴 — explain WHY based on what you see)

### ✅ Immediate First Aid Steps
(Numbered steps, specific to this injury — do NOT give generic advice)

### ❌ What You Must NOT Do
(Common mistakes for this specific injury type)

### 💊 What to Apply / Use
(Specific medicines, creams, bandages relevant to this injury)

### 🏥 Go to Hospital If...
(Specific warning signs for this injury)

For burns: identify burn degree (1st/2nd/3rd degree) from the image.
For cuts/wounds: assess if stitches may be needed.
For severe cases: always say CALL 108 IMMEDIATELY.

India emergency numbers: 108 Ambulance · 112 Police · 1066 Poison Control"""
}

# ───────────────────────────────────────────────────────────────
#  API CALLS
# ───────────────────────────────────────────────────────────────
def call_text(system: str, history: list, user_msg: str) -> str:
    msgs = [{"role": "system", "content": system}]
    for m in history[-8:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(
        model=TEXT_MODEL, messages=msgs, temperature=0.45, max_tokens=2000
    )
    return r.choices[0].message.content


def call_vision(system: str, user_msg: str, b64: str) -> str:
    r = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text",      "text": user_msg},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }}
            ]}
        ],
        temperature=0.3, max_tokens=2200
    )
    return r.choices[0].message.content


def call_pdf_analysis(system: str, pdf_text: str, user_msg: str) -> str:
    combined = f"""{system}

The user uploaded a PDF medical document. Here is the full extracted text:
---
{pdf_text[:7000]}
---
User question: {user_msg}

Analyze the above document thoroughly. If it is a blood report, analyze every single value."""
    r = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": combined},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.35, max_tokens=2200
    )
    return r.choices[0].message.content

# ───────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS PROMPT BUILDER
# ───────────────────────────────────────────────────────────────
def build_image_prompt(user_msg: str, fname: str, mode: str) -> str:
    base = PROMPTS[mode]
    return f"""{base}

The user uploaded an image file: "{fname}"

Your job:
- FIRST: Look at the image carefully.
- IDENTIFY what it shows:
  * BURN → describe burn degree and area, give complete burn first aid
  * CUT / WOUND → assess depth and bleeding, give wound care steps
  * SKIN RASH / REDNESS / INFECTION → identify and advise treatment
  * BRUISE / SWELLING → advise treatment
  * BLOOD TEST REPORT → read every value, compare to normal ranges, flag HIGH/LOW
  * MEDICINE PACKAGING / STRIP → identify and explain the medicine
  * X-RAY / PRESCRIPTION → analyze and explain

- THEN: Give the full appropriate response with all sections.

User's message: {user_msg}

Remember: Be specific about what you SEE in the image. Do not give generic advice."""

# ───────────────────────────────────────────────────────────────
#  SIDEBAR
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MedAI")
    st.caption("AI Health Assistant")
    st.divider()

    # Mode selector
    st.markdown("**Select Mode**")
    mode_options = {
        "General":      "💬  General Assistant",
        "Medicine Info":"💊  Medicine Info",
        "Body Checkup": "🧬  Body Checkup",
        "First Aid":    "🩹  First Aid",
    }
    chosen = st.radio(
        "mode_radio",
        list(mode_options.values()),
        label_visibility="collapsed",
        index=list(mode_options.keys()).index(st.session_state.mode)
    )
    st.session_state.mode = [k for k, v in mode_options.items() if v == chosen][0]

    st.divider()

    # Quick ask
    st.markdown("**⚡ Quick Ask**")
    quick_list = [
        ("💊 Paracetamol",      "What is Paracetamol used for? Give dosage and side effects."),
        ("💊 Metformin",        "Tell me everything about Metformin medicine."),
        ("💊 Azithromycin",     "What is Azithromycin used for?"),
        ("🧬 Blood report",     "Hemoglobin: 10.2 g/dL, Blood Sugar: 148 mg/dL, Cholesterol: 238 mg/dL, TSH: 6.8 mIU/L, Creatinine: 1.5 mg/dL, WBC: 11500. Analyze this blood report fully."),
        ("🩹 Cut finger",       "I cut my finger with a knife, it is bleeding. Step by step first aid?"),
        ("🩹 Burn hand",        "I burned my hand with hot water. What is the correct first aid?"),
        ("🩹 Someone fainted",  "A person near me fainted and is unconscious. What do I do?"),
    ]
    for label, prompt in quick_list:
        if st.button(label, use_container_width=True, key=f"q_{label}"):
            st.session_state["_quick"] = prompt
            st.rerun()

    st.divider()
    if st.button("🗑️  Clear Chat", use_container_width=True, key="clear_chat"):
        st.session_state.messages      = []
        st.session_state.pend_b64      = None
        st.session_state.pend_name     = None
        st.session_state.pend_type     = None
        st.session_state.pend_pdf_text = None
        st.session_state.prev_upname   = ""
        st.rerun()

    st.markdown("""
    <div style="font-size:11px; color:#484f58; line-height:1.7; margin-top:8px;">
    ⚕️ For educational purposes only.<br>
    Always consult a qualified doctor.<br>
    Emergency: <b style="color:#8b949e">108</b> (Ambulance)
    </div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
#  MAIN AREA
# ───────────────────────────────────────────────────────────────
mode_icon = {"General":"💬","Medicine Info":"💊","Body Checkup":"🧬","First Aid":"🩹"}

# Top bar
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown(
        f"### {mode_icon[st.session_state.mode]}  MedAI — {st.session_state.mode}"
    )
with col_t2:
    st.caption("Groq · llama-4-scout · llama-3.3-70b")

st.divider()

# ── Chat history ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 50px 0 30px; color:#8b949e;">
        <div style="font-size:52px; margin-bottom:12px;">🩺</div>
        <div style="font-size:22px; font-weight:700; color:#e6edf3; margin-bottom:8px;">
            How can I help you?
        </div>
        <div style="font-size:14px; max-width:420px; margin:0 auto; line-height:1.65;">
            Ask about any medicine · Upload a blood report image or PDF ·
            Upload a burn or wound photo for first aid · Describe your symptoms
        </div>
        <div style="margin-top:18px; display:flex; flex-wrap:wrap; gap:8px; justify-content:center;">
            <span style="padding:7px 14px;background:#161b22;border:1px solid #21262d;border-radius:20px;font-size:12px;">💊 Medicine lookup</span>
            <span style="padding:7px 14px;background:#161b22;border:1px solid #21262d;border-radius:20px;font-size:12px;">🧬 Blood report</span>
            <span style="padding:7px 14px;background:#161b22;border:1px solid #21262d;border-radius:20px;font-size:12px;">📄 PDF report</span>
            <span style="padding:7px 14px;background:#161b22;border:1px solid #21262d;border-radius:20px;font-size:12px;">🔥 Burn / wound photo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User bubble
            st.markdown(
                f'<div class="user-bubble">👤 &nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            # Show attached image thumbnail
            if msg.get("att_type") == "image" and msg.get("att_b64"):
                col_img, _ = st.columns([2, 3])
                with col_img:
                    st.image(
                        Image.open(BytesIO(base64.b64decode(msg["att_b64"]))),
                        caption=msg.get("att_name", "uploaded image"),
                        use_container_width=True
                    )
            # Show PDF indicator
            elif msg.get("att_type") == "pdf":
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                    f'padding:8px 12px;display:inline-flex;align-items:center;gap:8px;'
                    f'font-size:12px;color:#8b949e;margin:4px 0;">📄 {msg.get("att_name","report.pdf")}</div>',
                    unsafe_allow_html=True
                )
        else:
            # AI bubble
            html_content = md2html(msg["content"])
            st.markdown(
                f'<div class="ai-bubble">🩺 &nbsp;<strong>MedAI</strong><br><br>{html_content}</div>',
                unsafe_allow_html=True
            )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
#  INPUT SECTION  — file uploader (VISIBLE) + chat input
# ───────────────────────────────────────────────────────────────
st.divider()

# Show pending attachment preview
if st.session_state.pend_name:
    prev_col, del_col = st.columns([9, 1])
    with prev_col:
        if st.session_state.pend_type == "image" and st.session_state.pend_b64:
            img_data = base64.b64decode(st.session_state.pend_b64)
            pil_prev = Image.open(BytesIO(img_data))
            thumb_col, info_col = st.columns([1, 4])
            with thumb_col:
                st.image(pil_prev, width=80)
            with info_col:
                st.markdown(
                    f"**📎 {st.session_state.pend_name}**  \n"
                    f"<span style='font-size:12px;color:#8b949e;'>"
                    f"Image ready — type your message below and press Enter</span>",
                    unsafe_allow_html=True
                )
        elif st.session_state.pend_type == "pdf":
            st.markdown(
                f"📄 **{st.session_state.pend_name}**  \n"
                f"<span style='font-size:12px;color:#8b949e;'>"
                f"PDF extracted — type your question below and press Enter</span>",
                unsafe_allow_html=True
            )
    with del_col:
        if st.button("✕  Remove", key="rm"):
            st.session_state.pend_b64      = None
            st.session_state.pend_name     = None
            st.session_state.pend_type     = None
            st.session_state.pend_pdf_text = None
            st.session_state.prev_upname   = ""
            st.rerun()
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── File uploader — fully visible ────────────────────────────
st.markdown(
    "<span style='font-size:13px;color:#8b949e;'>📎 Attach a file (blood report image, PDF, burn/wound photo, medicine photo):</span>",
    unsafe_allow_html=True
)
uploaded = st.file_uploader(
    label="Upload image or PDF",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
    label_visibility="collapsed",
    key="file_upload",
)

# Process new upload
if uploaded and uploaded.name != st.session_state.prev_upname:
    raw = uploaded.read()
    if uploaded.type == "application/pdf":
        st.session_state.pend_pdf_text = extract_pdf(raw)
        try:
            st.session_state.pend_b64  = pdf_preview_b64(raw)
        except Exception:
            st.session_state.pend_b64  = None
        st.session_state.pend_type = "pdf"
    else:
        pil_img = Image.open(BytesIO(raw)).convert("RGB")
        st.session_state.pend_b64      = pil_to_b64(pil_img)
        st.session_state.pend_type     = "image"
        st.session_state.pend_pdf_text = None
    st.session_state.pend_name   = uploaded.name
    st.session_state.prev_upname = uploaded.name
    st.rerun()

# ── Chat input ───────────────────────────────────────────────
placeholder_map = {
    "General":      "Ask anything about health, medicines, symptoms…",
    "Medicine Info":"Type a medicine name (e.g. Paracetamol, Metformin)…",
    "Body Checkup": "Paste blood values, or upload a report above then ask…",
    "First Aid":    "Describe the emergency, or upload an injury/burn photo above…",
}
user_input = st.chat_input(
    placeholder_map[st.session_state.mode],
    key="chat_in"
)

# Inject quick prompt
if "_quick" in st.session_state:
    user_input = st.session_state.pop("_quick")

# ───────────────────────────────────────────────────────────────
#  PROCESS MESSAGE
# ───────────────────────────────────────────────────────────────
if user_input:
    b64      = st.session_state.pend_b64
    pdf_text = st.session_state.pend_pdf_text
    att_type = st.session_state.pend_type
    att_name = st.session_state.pend_name
    system   = PROMPTS[st.session_state.mode]

    # Save user turn
    user_entry = {"role": "user", "content": user_input}
    if att_type:
        user_entry["att_type"] = att_type
        user_entry["att_name"] = att_name
        if b64:
            user_entry["att_b64"] = b64
    st.session_state.messages.append(user_entry)

    # Generate AI response
    with st.spinner("🩺 MedAI is analyzing…"):
        try:
            if att_type == "image" and b64:
                prompt = build_image_prompt(user_input, att_name or "image", st.session_state.mode)
                reply  = call_vision(system, prompt, b64)

            elif att_type == "pdf" and pdf_text:
                reply = call_pdf_analysis(system, pdf_text, user_input)

            else:
                reply = call_text(system, st.session_state.messages[:-1], user_input)

        except Exception as e:
            err = str(e)
            if "401" in err or "api_key" in err.lower():
                reply = "❌ **Invalid API key.** Update `GROQ_API_KEY` in the script. Get a free key at console.groq.com"
            elif "429" in err or "rate" in err.lower():
                reply = "⏳ **Rate limit reached.** Please wait a moment and try again."
            elif "model" in err.lower():
                reply = f"⚠️ **Model error:** {err}"
            else:
                reply = f"❌ **Error:** {err}"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Clear pending attachment
    st.session_state.pend_b64      = None
    st.session_state.pend_name     = None
    st.session_state.pend_type     = None
    st.session_state.pend_pdf_text = None
    st.session_state.prev_upname   = ""

    st.rerun()
