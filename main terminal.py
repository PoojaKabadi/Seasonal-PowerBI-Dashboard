import os
import sys
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Helper Functions ---

def clean_text(text):
    text = text.replace('•', ' ').replace('–', ' ').replace('-', ' ')
    text = re.sub(r'[^A-Za-z0-9,. ]+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_file(filepath):
    ext = filepath.split('.')[-1].lower()
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == 'pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == 'docx':
        import docx
        doc = docx.Document(filepath)
        return '\n'.join(para.text for para in doc.paragraphs)
    else:
        print("❌ Unsupported file type. Please upload .txt, .pdf, or .docx.")
        exit()

def get_large_text_input(prompt_message):
    print(prompt_message)
    print("\n👉 Paste your entire text. Press Ctrl+D (Mac/Linux) or Ctrl+Z + Enter (Windows) when done.")
    return sys.stdin.read()

def extract_local_keywords(text):
    stopwords = set([
        'and', 'or', 'the', 'with', 'for', 'from', 'using', 'this', 'that',
        'to', 'of', 'on', 'at', 'as', 'by', 'an', 'in', 'be', 'is', 'are', 'a',
        'your', 'their', 'you', 'we', 'us', 'our', 'it', 'they', 'etc', 'such', 'upon', 'about',
        'have', 'has', 'had', 'been', 'being', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'may',
        'might', 'will', 'shall', 'must', 'if', 'while', 'during', 'into', 'out', 'up', 'down', 'over', 'under',
        'again', 'further', 'then', 'once', 'only', 'because', 'so', 'than', 'too', 'very', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'just', 'also', 'even', 'still', 'ever', 'always', 'never', 'where',
        'when', 'why', 'how', 'whose', 'whom', 'which', 'who', 'what', 'yet', 'though', 'although', 'until', 'before', 'after', 'within',
        'without', 'among', 'between', 'beside', 'around', 'across', 'through', 'per', 'versus', 'via'
    ])
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [word for word in words if word not in stopwords and len(word) > 2]

def extract_impact_achievements(text):
    impact_keywords = ["improved", "increased", "reduced", "boosted", "streamlined", "optimized", "cut", "saved", "grew", "enhanced", "led", "achieved"]
    return [line.strip() for line in text.split('\n') if any(word in line.lower() for word in impact_keywords)]

important_keywords_list = [
    "sql", "python", "tableau", "snowflake", "aws", "azure", "gcp", "machine", "learning", "hadoop", "powerbi", "kpi", "etl", "dashboard",
    "visualization", "regression", "classification", "data", "analytics", "modeling", "finance", "risk", "governance", "consulting", "management",
    "developer", "analyst", "scientist", "engineer", "consultant", "strategy", "operations", "scrum", "agile", "jira", "git", "cloud", "api", "product",
    "leadership", "customer", "marketing", "Business Strategy Alignment", "Digital Transformation", "Stakeholder Management", "Cross-functional Collaboration",
    "Business Intelligence (BI)", "Data-Driven Decision Making", "ROI Optimization", "Cost Reduction Initiatives", "Risk Management", "Value Proposition", "Product Innovation",
    "Operational Efficiency", "Revenue Growth Support", "Process Automation", "Agile Methodology", "Business Process Improvement", "Change Management", "Client Relationship Management",
    "User-Centered Design", "Customer Experience (CX) Enhancement", "Spearheaded", "Optimized", "Streamlined", "Delivered", "Orchestrated", "Drove", "Modernized", "Implemented", "Partnered",
]

def keyword_weight(word):
    if word.lower() in important_keywords_list:
        return 10
    elif len(word) >= 8:
        return 5
    elif len(word) >= 6:
        return 3
    else:
        return 1

def is_professional(word):
    return not any(char.isdigit() for char in word) and len(word) > 3

smart_phrases = [
    "Built scalable solutions utilizing {skill} to drive business efficiency.",
    "Implemented {skill} best practices to streamline operations.",
    "Designed and developed systems using {skill} to improve service delivery.",
    "Automated reporting and dashboards with {skill}, saving critical time for leadership.",
    "Enhanced data analysis processes by applying {skill} techniques.",
    "Collaborated cross-functionally to integrate {skill} solutions into business workflows.",
    "Led initiatives applying {skill} to optimize project outcomes and reduce risks.",
    "Created KPIs and dashboards using {skill} for executive decision support.",
    "Developed ETL pipelines and data models leveraging {skill} for faster insights.",
    "Introduced process automation leveraging {skill} resulting in significant cost savings."
]

def generate_tailored_bullets(missing_keywords):
    random.shuffle(smart_phrases)
    return [smart_phrases[i % len(smart_phrases)].format(skill=skill.capitalize()) for i, skill in enumerate(missing_keywords)]

def generate_tailored_cover_letter(job_title, company_name, impacts):
    intro = f"I am excited to apply for the {job_title} position at {company_name}. With a proven track record of delivering measurable results and technical excellence, I am confident in my ability to contribute meaningfully to your organization."
    body = "Here are some highlights of my impact:\n" + '\n'.join(f"- {line}" for line in impacts)
    closing = "\nThank you for considering my application. I look forward to discussing how my experience aligns with your team's goals."
    return intro + "\n\n" + body + "\n" + closing

# --- Input Section ---
print("\n🛠️ Choose Input Method:")
print("1. Upload Resume file & Upload Job Description file")
print("2. Upload Resume file & Paste Job Description text")
print("3. Paste Resume text & Upload Job Description file")
print("4. Paste both Resume and Job Description manually")

choice = input("\nEnter your choice (1/2/3/4): ").strip()

if choice == '1':
    resume_text = read_file(input("\n📄 Resume file path: ").strip())
    job_description_text = read_file(input("📄 JD file path: ").strip())
elif choice == '2':
    resume_text = read_file(input("\n📄 Resume file path: ").strip())
    job_description_text = get_large_text_input("\n🖊️ Paste Job Description:")
elif choice == '3':
    resume_text = get_large_text_input("\n🖊️ Paste Resume:")
    job_description_text = read_file(input("\n📄 JD file path: ").strip())
elif choice == '4':
    resume_text = get_large_text_input("\n🖊️ Paste Resume:")
    job_description_text = get_large_text_input("\n🖊️ Paste Job Description:")
else:
    print("❌ Invalid choice. Exiting.")
    exit()

# --- Cleaned Matching Logic ---
cleaned_resume = clean_text(resume_text)
cleaned_jd = clean_text(job_description_text)

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

resume_skills = set(extract_local_keywords(resume_text))
jd_skills = set(extract_local_keywords(job_description_text))
missing = jd_skills - resume_skills
matched = jd_skills & resume_skills

weighted = [(word, keyword_weight(word)) for word in missing if is_professional(word)]
top_missing = [word for word, _ in sorted(weighted, key=lambda x: x[1], reverse=True)[:10]]

print(f"\n✅ Resume-JD Match Score: {round(similarity_score * 100, 2)}%")
print("\n🔍 Top Missing Business/Technical Keywords:")
print("- None! 🎯" if not top_missing else '\n'.join(f"- {word}" for word in top_missing))

print("\n✅ Matched Keywords:")
print("- None matched." if not matched else '\n'.join(f"- {word}" for word in sorted(matched)))

# --- Resume Tailoring ---
print("\n📄 Suggested Resume Enhancements:")
bullets = generate_tailored_bullets(top_missing)
print("- No updates needed." if not bullets else '\n'.join(f"• {b}" for b in bullets))

# --- LLM-Based Professional Summary ---
print("\n🤖 LLM-Powered Summary:")
reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
docs = reader.load_data()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.4)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(docs, service_context=service_context)
response = index.as_query_engine().query("Generate a 5-sentence resume summary highlighting strengths for a business analyst role.")
print(response)

# --- Tailored Cover Letter ---
job_title = input("\n🖊️ Enter Job Title: ").strip()
company_name = input("🖊️ Enter Company Name: ").strip()
impactful = extract_impact_achievements(resume_text)
cover_letter = generate_tailored_cover_letter(job_title, company_name, impactful)

print("\n✉️ Tailored Cover Letter:\n")
print(cover_letter)

print("\n✅ Tailoring Completed Successfully!")
