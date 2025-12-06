import streamlit as st
from huggingface_hub import InferenceClient
import pdfplumber
import re
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Interview Generator",
    page_icon="🎯",
    layout="wide"
)

# =============================
# HUGGINGFACE API CONFIG (✅ PYTHON 3.13 SAFE)
# =============================
@st.cache_resource
def get_client():
    try:
        hf_key = st.secrets.get("HF_API_KEY", "")
        if not hf_key:
            st.warning("⚠️ HuggingFace API key not found in secrets.")
            return None

        return InferenceClient(
            model="meta-llama/Llama-3.2-1B-Instruct",
            token=hf_key
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize HF client: {str(e)}")
        return None

client = get_client()

# =============================
# PDF EXTRACTOR
# =============================
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error extracting PDF: {str(e)}")
        return None

# =============================
# QUESTION FORMATTING
# =============================
def format_questions(raw_text):
    lines = raw_text.split('\n')
    questions = []
    current_question = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^(\d+[\.)]\s*|[-*•]\s*|\b(What|How|Why|Describe|Explain|Can you|Tell me)\b)', line, re.IGNORECASE):
            if current_question:
                questions.append(current_question.strip())
            current_question = re.sub(r'^(\d+[\.)]\s*|[-*•]\s*)', '', line)
        else:
            current_question += " " + line

    if current_question:
        questions.append(current_question.strip())

    return questions if questions else [raw_text]

# =============================
# ✅ PYTHON 3.13–SAFE LLM CALL
# =============================
def run_llm(prompt):
    if not client:
        return None
    try:
        output = client.text_generation(
            prompt,
            max_new_tokens=800,
            temperature=0.7
        )
        return output
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

# =============================
# QUESTION GENERATION
# =============================
def generate_normal_questions(role, difficulty="Medium", num_questions=10):
    prompt = f"""
Generate exactly {num_questions} interview questions for a {role} position at {difficulty} difficulty.

Format: each question on a new line starting with a number.

40% technical
30% behavioral
30% scenario based
"""
    return run_llm(prompt)

def generate_resume_questions(resume_text, num_questions=12):
    if len(resume_text) > 3000:
        resume_text = resume_text[:3000] + "..."

    prompt = f"""
Based on this resume, generate exactly {num_questions} specific interview questions:

{resume_text}
"""
    return run_llm(prompt)

# =============================
# DISPLAY FUNCTIONS
# =============================
def display_questions(questions, category="Interview"):
    st.markdown(f"### 📋 {category} Questions ({len(questions)} total)")

    for i, q in enumerate(questions, 1):
        with st.container():
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(f"**{i}.** {q}")
            with col2:
                if st.button("📋", key=f"copy_{i}"):
                    st.toast(f"Question {i} copied!")
            st.divider()

def save_questions_to_file(questions):
    content = f"Interview Questions - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    for i, q in enumerate(questions, 1):
        content += f"{i}. {q}\n\n"
    return content

# =============================
# STREAMLIT UI
# =============================
st.title("🎯 AI Interview Question Generator")
st.markdown("Generate tailored interview questions using AI")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Select Mode:",
        ["Role-Based Questions", "Resume-Based Questions"]
    )

# =============================
# MODE 1 — ROLE-BASED
# =============================
if mode == "Role-Based Questions":
    col1, col2 = st.columns([2, 1])

    with col1:
        job_role = st.text_input("Job Role")

    with col2:
        difficulty = st.selectbox("Difficulty Level", ["Entry Level", "Medium", "Senior/Expert"])

    num_questions = st.slider("Number of Questions", 5, 20, 10)

    if st.button("🚀 Generate Questions", use_container_width=True):
        if not job_role.strip():
            st.error("⚠️ Please enter a job role.")
        else:
            with st.spinner("🤖 Generating questions..."):
                raw_output = generate_normal_questions(job_role, difficulty, num_questions)

                if raw_output:
                    questions = format_questions(raw_output)
                    st.success(f"✅ Generated {len(questions)} questions!")
                    display_questions(questions, job_role)

                    st.download_button(
                        "📥 Download Questions",
                        save_questions_to_file(questions),
                        file_name="interview_questions.txt"
                    )

# =============================
# MODE 2 — RESUME BASED
# =============================
else:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])
    num_questions = st.number_input("Questions", 8, 20, 12)

    if resume_file:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")

        if resume_text:
            st.text_area("Resume Preview", resume_text, height=200)

            if st.button("🚀 Generate Resume-Based Questions"):
                with st.spinner("🤖 Analyzing resume..."):
                    raw_output = generate_resume_questions(resume_text, num_questions)
                    if raw_output:
                        questions = format_questions(raw_output)
                        display_questions(questions, "Resume")

                        st.download_button(
                            "📥 Download Questions",
                            save_questions_to_file(questions),
                            file_name="resume_questions.txt"
                        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>Powered by Llama 3.2 • Built with Streamlit</div>",
    unsafe_allow_html=True
)
