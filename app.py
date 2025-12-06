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
# HUGGINGFACE API CONFIG
# =============================
@st.cache_resource
def get_client():
    """Initialize and cache the HF client"""
    try:
        return InferenceClient(
            "meta-llama/Llama-3.2-1B-Instruct",
            token=st.secrets.get("HF_API_KEY", "")
        )
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return None

client = get_client()

# =============================
# PDF EXTRACTOR
# =============================
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

# =============================
# QUESTION FORMATTING
# =============================
def format_questions(raw_text):
    """Parse and format questions into a clean list"""
    # Split by common question patterns
    lines = raw_text.split('\n')
    questions = []
    current_question = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with number, bullet, or question word
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
# QUESTION GENERATION
# =============================
def generate_normal_questions(role, difficulty="Medium", num_questions=10):
    """Generate role-based interview questions"""
    if not client:
        return None
        
    prompt = f"""Generate exactly {num_questions} interview questions for a {role} position at {difficulty} difficulty level.

Format each question on a new line starting with a number.

Include:
- 40% Technical/skill-based questions
- 30% Behavioral/situational questions  
- 30% Problem-solving/scenario questions

Make questions specific to {role} responsibilities and realistic for actual interviews.

Questions:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def generate_resume_questions(resume_text, num_questions=12):
    """Generate resume-specific interview questions"""
    if not client:
        return None
    
    # Truncate resume if too long
    if len(resume_text) > 3000:
        resume_text = resume_text[:3000] + "..."
    
    prompt = f"""Based on this resume, generate exactly {num_questions} targeted interview questions:

{resume_text}

Format each question on a new line starting with a number.

Focus on:
- Specific projects mentioned (ask for details, challenges, outcomes)
- Technical skills listed (assess depth of knowledge)
- Work experience (responsibilities, achievements, lessons learned)
- Technologies and tools mentioned

Make questions specific to what's actually in the resume. Avoid generic questions.

Questions:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# =============================
# DISPLAY FUNCTIONS
# =============================
def display_questions(questions, category="Interview"):
    """Display questions in an organized format"""
    st.markdown(f"### 📋 {category} Questions ({len(questions)} total)")
    
    for i, q in enumerate(questions, 1):
        with st.container():
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(f"**{i}.** {q}")
            with col2:
                if st.button("📋", key=f"copy_{i}", help="Copy question"):
                    st.toast(f"Question {i} copied!")
            st.divider()

def save_questions_to_file(questions, filename):
    """Generate downloadable text file"""
    content = f"Interview Questions - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    for i, q in enumerate(questions, 1):
        content += f"{i}. {q}\n\n"
    return content

# =============================
# STREAMLIT UI
# =============================
st.title("🎯 AI Interview Question Generator")
st.markdown("Generate tailored interview questions using AI")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Select Mode:",
        ["Role-Based Questions", "Resume-Based Questions"]
    )
    
    st.divider()
    st.markdown("### About")
    st.info("This tool uses Llama 3.2 to generate context-specific interview questions.")

# =============================
# MODE 1 — ROLE-BASED
# =============================
if mode == "Role-Based Questions":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_role = st.text_input(
            "Job Role",
            placeholder="e.g., Senior Data Analyst, Full Stack Developer",
            help="Enter the specific job title or role"
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Entry Level", "Medium", "Senior/Expert"]
        )
    
    num_questions = st.slider("Number of Questions", 5, 20, 10)
    
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        if not job_role.strip():
            st.error("⚠️ Please enter a job role.")
        else:
            with st.spinner("🤖 Generating questions..."):
                raw_output = generate_normal_questions(job_role, difficulty, num_questions)
                
                if raw_output:
                    questions = format_questions(raw_output)
                    
                    # Store in session state
                    st.session_state['questions'] = questions
                    st.session_state['job_role'] = job_role
                    
                    st.success(f"✅ Generated {len(questions)} questions!")
                    
                    # Display questions
                    display_questions(questions, f"{job_role} Interview")
                    
                    # Download button
                    file_content = save_questions_to_file(
                        questions, 
                        f"{job_role.replace(' ', '_')}_questions.txt"
                    )
                    st.download_button(
                        label="📥 Download Questions",
                        data=file_content,
                        file_name=f"{job_role.replace(' ', '_')}_questions.txt",
                        mime="text/plain"
                    )

# =============================
# MODE 2 — RESUME BASED
# =============================
else:
    st.markdown("### 📄 Upload Resume")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        resume_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt"],
            help="Upload resume in PDF or TXT format"
        )
    
    with col2:
        num_questions = st.number_input("Questions", 8, 20, 12)
    
    if resume_file:
        # Extract text
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")
        
        if resume_text:
            st.success(f"✅ Resume uploaded ({len(resume_text)} characters)")
            
            # Show preview
            with st.expander("📄 View Resume Text"):
                st.text_area("Resume Content", resume_text, height=200)
            
            if st.button("🚀 Generate Resume-Based Questions", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing resume and generating questions..."):
                    raw_output = generate_resume_questions(resume_text, num_questions)
                    
                    if raw_output:
                        questions = format_questions(raw_output)
                        
                        # Store in session state
                        st.session_state['questions'] = questions
                        
                        st.success(f"✅ Generated {len(questions)} resume-specific questions!")
                        
                        # Display questions
                        display_questions(questions, "Resume-Based Interview")
                        
                        # Download button
                        file_content = save_questions_to_file(
                            questions,
                            "resume_based_questions.txt"
                        )
                        st.download_button(
                            label="📥 Download Questions",
                            data=file_content,
                            file_name="resume_based_questions.txt",
                            mime="text/plain"
                        )
        else:
            st.error("❌ Could not extract text from resume.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Llama 3.2 • Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
