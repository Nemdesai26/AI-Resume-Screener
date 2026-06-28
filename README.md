# 🤖 ATS Resume Screener

An AI-powered resume screening tool that automatically evaluates resumes against job descriptions using NLP techniques — TF-IDF keyword matching, Named Entity Recognition (NER), and semantic similarity scoring.

---

## 📌 Overview

Hiring teams spend a significant chunk of time manually filtering resumes. This project automates that process by scoring resumes based on how well they match a given job description, surfacing the most relevant candidates quickly and objectively.

---

## ✨ Features

- **TF-IDF Keyword Matching** — Extracts and weights important terms from job descriptions and resumes to compute relevance scores
- **Named Entity Recognition (NER)** — Identifies key entities like skills, tools, degrees, and institutions using spaCy
- **Semantic Similarity** — Goes beyond keyword overlap by comparing contextual meaning using sentence embeddings
- **Composite Scoring** — Combines all three signals into a single ranked score per resume
- **Batch Processing** — Screen multiple resumes against a single JD in one run

---

## 🗂️ Project Structure

```
ats-resume-screener/
│
├── data/
│   ├── resumes/              # Input resumes (PDF or plain text)
│   └── job_descriptions/     # JD files for screening
│
├── src/
│   ├── preprocessor.py       # Text cleaning and extraction
│   ├── tfidf_scorer.py       # TF-IDF vectorization and scoring
│   ├── ner_extractor.py      # Entity extraction via spaCy
│   ├── semantic_scorer.py    # Sentence embedding similarity
│   └── ranker.py             # Final composite scoring and ranking
│
├── outputs/
│   └── results.csv           # Ranked results with individual scores
│
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|---|---|
| Text Preprocessing | `NLTK`, `re` |
| TF-IDF Scoring | `scikit-learn` |
| Named Entity Recognition | `spaCy` (`en_core_web_sm`) |
| Semantic Similarity | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| PDF Parsing | `PyMuPDF` / `pdfminer` |
| Data Handling | `pandas` |

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/ats-resume-screener.git
cd ats-resume-screener
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

Place your resumes in `data/resumes/` and your job description in `data/job_descriptions/`.

```bash
python main.py --jd data/job_descriptions/ml_engineer.txt --resumes data/resumes/
```

**Output** — a ranked CSV at `outputs/results.csv`:

| Resume | TF-IDF Score | NER Score | Semantic Score | Final Score |
|---|---|---|---|---|
| candidate_a.pdf | 0.82 | 0.75 | 0.88 | 0.83 |
| candidate_b.pdf | 0.61 | 0.70 | 0.65 | 0.65 |

---

## 📊 Scoring Methodology

The final score is a weighted combination of three signals:

```
Final Score = 0.4 × TF-IDF + 0.3 × NER Match + 0.3 × Semantic Similarity
```

- **TF-IDF** captures keyword relevance between the JD and resume
- **NER Match** measures overlap in entities like skills, tools, and qualifications
- **Semantic Similarity** captures contextual alignment using sentence embeddings, handling synonyms and paraphrasing that keyword matching misses

---

## 📈 Example Results

Tested on a set of 20 resumes for a Data Engineer JD:

- Top-3 candidates surfaced had an average manual relevance rating of **4.6 / 5**
- Screening time reduced from ~40 minutes to under **30 seconds**

---

## 🔮 Future Improvements

- [ ] Add a Streamlit web interface for non-technical users
- [ ] Support DOCX resume formats
- [ ] Fine-tune NER model on a domain-specific resume dataset
- [ ] Add bias detection module to flag potentially discriminatory filtering
- [ ] REST API endpoint for integration with HR tools

---

## 👤 Author

**Nem Desai**  
B.Tech ECE + Minor in Data Science | Nirma University, Ahmedabad  

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
