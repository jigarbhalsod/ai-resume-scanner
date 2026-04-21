# 📄 AI Resume Scanner

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?style=flat&logo=streamlit)
![spaCy](https://img.shields.io/badge/spaCy-3.7+-09A3D5?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An AI-powered resume analysis tool that parses resumes, extracts skills using NLP, scores ATS compatibility, detects AI-generated content, and matches resumes to job roles — all through a clean Streamlit web interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Resume Parsing** | Supports PDF and DOCX file formats |
| 🧠 **Skill Extraction** | NLP-powered extraction across 7 skill categories |
| 📊 **ATS Scoring** | Weighted scoring across 6 compatibility factors |
| 🤖 **AI Detection** | Detects AI-generated or AI-assisted resume content |
| 🎯 **Job Matching** | TF-IDF cosine similarity matching against 6 job roles |
| 📈 **Visual Reports** | Plotly charts, radar graphs, gauge indicators, and word clouds |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/jigarbhalsod/ai-resume-scanner.git
cd ai-resume-scanner

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm

# 5. Run the app
streamlit run app.py
```

---

## 📖 Usage

1. Open the app in your browser (usually at `http://localhost:8501`)
2. Upload your resume using the sidebar file uploader (PDF or DOCX)
3. Optionally select a target job role from the dropdown
4. Click **Analyse**
5. Explore the four result tabs — Skills, ATS Score, AI Detection, and Job Match

---

## 🔍 How It Works

**Skill Extraction** — The NLP engine uses spaCy for tokenisation and named entity recognition, combined with regex word-boundary matching against a curated database of 150+ skills across 7 categories including programming languages, frameworks, cloud/DevOps, and ML/AI concepts.

**ATS Scoring** — The ATS scorer evaluates resumes across six weighted dimensions: section completeness (20%), formatting quality (15%), keyword density (25%), resume length (10%), readability (15%), and contact info completeness (15%). Resumes scoring above 60 are flagged as ATS-passing.

**AI Detection** — The detector analyses vocabulary diversity using Type-Token Ratio, scans for known AI phrase patterns and overused buzzwords, and checks for sentence-level repetition. A weighted probability score is computed and mapped to a human-readable verdict.

**Job Matching** — TF-IDF vectors are computed from scratch for both the resume and six job description profiles. Cosine similarity between these vectors produces a ranked list of role matches, helping candidates understand where their profile fits best.

---

## 🗂️ Project Structure

```
ai-resume-scanner/
├── app.py                   # Main Streamlit application
├── requirements.txt
├── resume_scanner/
│   ├── __init__.py
│   ├── parser.py            # PDF/DOCX text extraction
│   ├── nlp_engine.py        # Skill extraction via NLP
│   ├── ats_scorer.py        # ATS compatibility scoring
│   ├── ai_detector.py       # AI-generated content detection
│   └── job_matcher.py       # TF-IDF job role matching
└── data/
    ├── skills_database.json
    └── job_keywords.json
```

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit
- **NLP** — spaCy, NLTK
- **ML / Similarity** — scikit-learn (TF-IDF), custom cosine similarity
- **File Parsing** — PyMuPDF (fitz), python-docx
- **Visualisation** — Plotly, Matplotlib, WordCloud
- **Data** — Pandas, NumPy

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, new features, or improvements.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Jigar Bhalsod**  
[GitHub](https://github.com/jigarbhalsod) • [LinkedIn](www.linkedin.com/in/jigar-bhalsod)
