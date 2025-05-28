import streamlit as st
import re
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# Set OpenAI API key (ensure this is securely stored in production)
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# ----------------------------- CONFIG ----------------------------- #
st.set_page_config(page_title="AI Job Hunter", layout="wide")
st.markdown("""
    <style>
        .stApp {background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif;}
        .big-title {font-size: 40px; color: #29465B; font-weight: bold;}
        .section {font-size: 24px; font-weight: 600; color: #2E3B4E; margin-top: 30px;}
        .metric-box {padding: 1rem; border-radius: 10px; background-color: #e8eff7;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------- HELPERS ----------------------------- #
def clean_text(text):
    text = text.replace('•', ' ').replace('–', ' ').replace('-', ' ')
    text = re.sub(r'[^A-Za-z0-9,. ]+', ' ', text)
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

def extract_keywords(text):
    stopwords = set([
        'and','or','the','with','for','from','using','this','that','to','of','on','at','as','by','an','in','be','is','are','a',
        'your','their','you','we','us','our','it','they','etc','such','upon','about','have','has','had','been','being','do',
        'does','did','can','could','would','should','may','might','will','shall','must','if','while','during','into','out','up',
        'down','over','under','again','further','then','once','only','because','so','than','too','very','both','each','few',
        'more','most','other','some','no','nor','not','just','also','even','still','ever','always','never','where','when','why',
        'how','whose','whom','which','who','what','yet','though','although','until','before','after','within','without','among',
        'between','beside','around','across','through','per','versus','via']
    )
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]

def keyword_weight(word):
    return 10 if word.lower() in important_keywords else 1

def generate_tailored_bullets(missing):
    bullets = []
    random.shuffle(smart_phrases)
    for idx, skill in enumerate(missing):
        template = smart_phrases[idx % len(smart_phrases)]
        bullets.append(template.format(skill=skill))
    return bullets

def llm_generate_cover_letter(resume_text, jd_text):
    llm = OpenAI(model="gpt-3.5-turbo")
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
    
    docs = ["Resume:\n" + resume_text, "Job Description:\n" + jd_text]
    index = VectorStoreIndex.from_documents(SimpleDirectoryReader(input_files=None, documents=docs).load_data())
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query("Write a tailored professional cover letter based on the resume and job description")
    return str(response)

# ---------------------- DATA ---------------------- #
important_keywords = ["sql","python","tableau","etl","dashboard","cloud","analytics","bi","ai","machine","data","governance"]
smart_phrases = [
    "Built scalable solutions utilizing {skill} to drive business efficiency.",
    "Implemented {skill} best practices to streamline operations.",
    "Enhanced data analysis processes by applying {skill} techniques.",
    "Developed dashboards and KPIs using {skill} for strategic decision-making."
]

# ----------------------- UI ----------------------- #
st.markdown("<div class='big-title'>🧠 AI Job Hunter — Tailor Your Resume & Cover Letter</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx", "txt"], key="resume")
with col2:
    jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"], key="jd")

if resume_file and jd_file:
    from PyPDF2 import PdfReader
    import docx

    def read_content(file):
        ext = file.name.split('.')[-1]
        if ext == 'pdf':
            pdf = PdfReader(file)
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext == 'docx':
            d = docx.Document(file)
            return "\n".join([para.text for para in d.paragraphs])
        elif ext == 'txt':
            return file.read().decode("utf-8")
        return ""

    resume_text = read_content(resume_file)
    jd_text = read_content(jd_file)

    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)

    # Match score
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([clean_resume, clean_jd])
    score = round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

    # Skill gap
    resume_skills = set(extract_keywords(resume_text))
    jd_skills = set(extract_keywords(jd_text))
    missing_skills = sorted(list(jd_skills - resume_skills), key=lambda x: -keyword_weight(x))[:10]
    matched_skills = sorted(list(jd_skills & resume_skills))

    st.markdown("<div class='section'>📊 Resume - JD Match Score</div>", unsafe_allow_html=True)
    st.metric("Match Score", f"{score}%")

    st.markdown("<div class='section'>🚨 Missing Critical Skills</div>", unsafe_allow_html=True)
    st.write(missing_skills if missing_skills else "None! Resume looks well-aligned. 🎯")

    st.markdown("<div class='section'>✅ Keywords Already Matched</div>", unsafe_allow_html=True)
    st.write(matched_skills if matched_skills else "No major overlaps found.")

    st.markdown("<div class='section'>📌 Tailored Resume Suggestions</div>", unsafe_allow_html=True)
    bullets = generate_tailored_bullets(missing_skills)
    for b in bullets:
        st.write(f"- {b}")

    st.markdown("<div class='section'>✉️ LLM-Powered Tailored Cover Letter</div>", unsafe_allow_html=True)
    with st.spinner("Generating Cover Letter..."):
        letter = llm_generate_cover_letter(resume_text, jd_text)
    st.code(letter, language="markdown")
