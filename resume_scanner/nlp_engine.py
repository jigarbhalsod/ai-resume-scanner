"""
nlp_engine.py — NLP Skill Extraction Engine
=============================================
Concept: Think of this as a highly trained recruiter who has memorised
every tech skill in existence. You hand them a resume, they instantly
highlight every skill they recognise.

How skill matching works here:
  1. We maintain curated sets of known skills (programming languages, frameworks, etc.)
  2. For each skill, we search the resume text using regex with WORD BOUNDARIES
  3. Word boundaries (\b) are critical — without them, "R" matches inside "React",
     "Go" matches inside "Google", "C" matches everywhere

What is \b (word boundary)?
  - \b matches the position between a word character and a non-word character
  - "python" with \b: matches " python " but NOT "cpython" or "pythonic"
  - Example: r'\bgo\b' matches "Go" as a language but not "Google" or "going"

Type-Token Ratio (TTR) explained:
  - Counts unique words / total words
  - High TTR (0.7+) = rich, varied vocabulary = human writing
  - Low TTR (0.3-) = repetitive vocabulary = AI-generated writing
  - Example: "I love coding. I love Python. I love ML."
    Total words: 9, Unique words: 6 → TTR = 6/9 = 0.67

spaCy pipeline:
  - spaCy loads a pre-trained neural network model (en_core_web_sm)
  - The model was trained on millions of English sentences
  - It performs: tokenisation → POS tagging → dependency parsing → NER
  - NER (Named Entity Recognition) identifies: ORG, PERSON, GPE (location), DATE, etc.

Common error: spaCy model not downloaded
  Fix: python -m spacy download en_core_web_sm
"""

import re
import math
from collections import Counter
from datetime import datetime


# ── spaCy — graceful fallback if not installed ────────────────────────────────
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Using pattern matching only.")
    print("Fix: pip install spacy && python -m spacy download en_core_web_sm")


class NLPEngine:
    """
    Extracts skills, entities, experience years, and text quality metrics
    from resume plain text.

    Two modes:
    - With spaCy (default): Uses neural NER + regex pattern matching
    - Without spaCy (fallback): Uses regex pattern matching only
    """

    # ──────────────────────────────────────────────────────────────────────────
    # SKILL KNOWLEDGE BASE — curated sets of known skills per category
    # Using sets (not lists) for O(1) lookup performance
    # ──────────────────────────────────────────────────────────────────────────

    PROGRAMMING_LANGUAGES = {
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "ruby", "go", "golang", "rust", "kotlin", "swift", "scala", "php",
        "perl", "r", "matlab", "julia", "dart", "bash", "shell", "sql",
        "html", "css", "sass", "less", "groovy", "haskell", "elixir",
        "clojure", "f#", "cobol", "fortran", "assembly", "vba", "powershell"
    }

    FRAMEWORKS_LIBRARIES = {
        # Python
        "django", "flask", "fastapi", "streamlit", "pandas", "numpy",
        "scipy", "matplotlib", "seaborn", "plotly", "scikit-learn",
        "tensorflow", "keras", "pytorch", "xgboost", "lightgbm",
        "nltk", "spacy", "transformers", "hugging face", "celery",
        "sqlalchemy", "pydantic", "pytest", "asyncio",
        # JavaScript
        "react", "angular", "vue", "next.js", "nuxt", "svelte",
        "node.js", "express", "nestjs", "jquery", "d3.js", "three.js",
        # Java
        "spring", "spring boot", "hibernate", "maven", "gradle", "junit",
        # Other
        "rails", ".net", "asp.net", "flutter", "react native", "graphql",
        "grpc", "fastify", "gin", "echo", "actix"
    }

    DATA_SCIENCE_TOOLS = {
        "jupyter", "jupyter notebook", "anaconda", "google colab", "kaggle",
        "databricks", "mlflow", "wandb", "weights and biases", "tensorboard",
        "tableau", "power bi", "looker", "grafana", "metabase",
        "excel", "google sheets", "stata", "spss", "sas",
        "dbt", "great expectations", "apache beam", "dataiku"
    }

    DATABASES = {
        "mysql", "postgresql", "postgres", "mongodb", "redis",
        "elasticsearch", "cassandra", "sqlite", "oracle", "dynamodb",
        "snowflake", "redshift", "bigquery", "firebase", "neo4j",
        "couchdb", "influxdb", "clickhouse", "cockroachdb", "supabase",
        "planetscale", "fauna", "arangodb"
    }

    CLOUD_DEVOPS = {
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
        "circleci", "travis ci", "prometheus", "grafana", "nginx",
        "heroku", "vercel", "netlify", "cloudflare", "helm",
        "istio", "consul", "vault", "packer", "pulumi",
        "argocd", "flux", "tekton", "spinnaker"
    }

    SOFT_SKILLS = {
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "agile", "scrum", "kanban", "project management",
        "mentoring", "coaching", "stakeholder management", "time management",
        "collaboration", "adaptability", "presentation", "negotiation",
        "conflict resolution", "strategic thinking", "decision making"
    }

    ML_AI_CONCEPTS = {
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "transfer learning",
        "feature engineering", "feature selection", "hyperparameter tuning",
        "clustering", "classification", "regression", "dimensionality reduction",
        "cnn", "rnn", "lstm", "gru", "transformer", "bert", "gpt",
        "generative ai", "llm", "large language model", "rag",
        "retrieval augmented generation", "langchain", "llamaindex",
        "sentiment analysis", "anomaly detection", "time series forecasting",
        "object detection", "image segmentation", "gan",
        "gradient boosting", "random forest", "svm", "neural network",
        "attention mechanism", "embedding", "vector database", "fine-tuning",
        "prompt engineering", "a/b testing", "mlops", "model deployment",
        "model monitoring", "data drift", "explainable ai", "xai"
    }

    # Action verbs that signal strong, human-written resume bullet points
    ACTION_VERBS = {
        "achieved", "built", "created", "delivered", "designed", "developed",
        "engineered", "established", "executed", "generated", "implemented",
        "improved", "increased", "launched", "led", "managed", "optimized",
        "reduced", "resolved", "scaled", "shipped", "solved", "trained",
        "transformed", "wrote", "analyzed", "automated", "collaborated",
        "contributed", "coordinated", "deployed", "integrated", "migrated",
        "refactored", "reviewed", "tested", "maintained", "documented"
    }

    def __init__(self, use_spacy=True):
        """
        Initialises the NLP engine.

        Args:
            use_spacy : bool — whether to load spaCy model
                        Set False to skip spaCy (faster, less accurate NER)
        """
        self.nlp = None

        if use_spacy and SPACY_AVAILABLE:
            try:
                # Load the small English model (~12MB)
                # en_core_web_sm is fast but less accurate than en_core_web_lg
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded yet — give user a clear fix
                print(
                    "spaCy model 'en_core_web_sm' not found.\n"
                    "Fix: python -m spacy download en_core_web_sm"
                )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIMARY SKILL EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_skills(self, text):
        """
        Extracts skills from resume text organised by category.

        Args:
            text : str — plain text resume content

        Returns:
            dict with keys: programming_languages, frameworks_libraries,
                            data_science_tools, databases, cloud_devops,
                            soft_skills, ml_ai_concepts
            Each value is a sorted list of found skills.
        """
        text_lower = text.lower()

        return {
            "programming_languages": self._find_skills(text_lower, self.PROGRAMMING_LANGUAGES),
            "frameworks_libraries":  self._find_skills(text_lower, self.FRAMEWORKS_LIBRARIES),
            "data_science_tools":    self._find_skills(text_lower, self.DATA_SCIENCE_TOOLS),
            "databases":             self._find_skills(text_lower, self.DATABASES),
            "cloud_devops":          self._find_skills(text_lower, self.CLOUD_DEVOPS),
            "soft_skills":           self._find_skills(text_lower, self.SOFT_SKILLS),
            "ml_ai_concepts":        self._find_skills(text_lower, self.ML_AI_CONCEPTS),
        }

    def _find_skills(self, text, skill_set):
        """
        Searches text for each skill using word boundary regex.

        Why re.escape(skill)?
        - Some skills contain regex special chars like "C++" or ".NET"
        - re.escape converts "C++" → "C\+\+" so + is treated as literal, not regex

        Why \b word boundaries?
        - Without \b: "r" would match inside "react", "docker", "server"
        - With \b: r'\br\b' only matches standalone "r" (the language)

        Args:
            text      : str — lowercase resume text
            skill_set : set — skills to search for

        Returns:
            list — sorted list of found skills (capitalised)
        """
        found = []
        for skill in skill_set:
            # Build pattern: \b + escaped_skill + \b
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Capitalise first letter of each word for display
                found.append(skill.title())
        return sorted(found)

    def get_all_skills_flat(self, text):
        """
        Returns all found skills as a single flat sorted list.
        Useful for word cloud generation and total skill count.

        Returns:
            list — all skills across all categories, deduplicated and sorted
        """
        skills_by_category = self.extract_skills(text)
        all_skills = []
        for skills in skills_by_category.values():
            all_skills.extend(skills)
        return sorted(set(all_skills))  # set() removes any cross-category duplicates

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY EXTRACTION (spaCy NER)
    # ──────────────────────────────────────────────────────────────────────────

    def extract_entities(self, text):
        """
        Extracts named entities: companies, locations, dates, degrees.

        spaCy NER labels used:
        - ORG  : organisations (Google, MIT, startup names)
        - GPE  : geopolitical entities (cities, countries)
        - LOC  : non-GPE locations (regions, rivers)
        - DATE : date expressions ("January 2022", "2 years ago")
        - PERSON: person names

        Returns:
            dict with keys: organizations, locations, dates, degrees, institutions
        """
        entities = {
            "organizations": [],
            "locations": [],
            "dates": [],
            "degrees": [],
            "institutions": []
        }

        # spaCy NER path
        if self.nlp:
            doc = self.nlp(text[:100000])  # spaCy has token limits — cap at 100k chars
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["organizations"].append(ent.text.strip())
                elif ent.label_ in ("GPE", "LOC"):
                    entities["locations"].append(ent.text.strip())
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text.strip())

        # Regex for education degrees (works even without spaCy)
        degree_pattern = r"\b(B\.?S\.?|M\.?S\.?|Ph\.?D\.?|B\.?E\.?|M\.?E\.?|MBA|BCA|MCA|B\.?Tech|M\.?Tech|Bachelor|Master|Doctorate)\b"
        degrees = re.findall(degree_pattern, text, re.IGNORECASE)
        entities["degrees"] = list(set(degrees))  # deduplicate

        # Regex for common institution keywords
        institution_pattern = r"\b\w+\s*(University|College|Institute|School|Academy)\b"
        institutions = re.findall(institution_pattern, text, re.IGNORECASE)
        entities["institutions"] = list(set(institutions))

        # Deduplicate all lists
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))  # preserves order

        return entities

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIENCE YEARS CALCULATION
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_experience_years(self, text):
        """
        Estimates total years of professional experience by finding year ranges
        in the resume text.

        Handles patterns like:
        - "2020 - 2022"
        - "Jan 2019 – Present"
        - "2018 to 2020"
        - "2021 – present"

        Args:
            text : str — resume plain text

        Returns:
            float — estimated total years of experience (rounded to 1 decimal)
        """
        CURRENT_YEAR = 2026  # Update this each year

        # Pattern captures: start_year ... separator ... end_year_or_present
        # Group 1: start year (4 digits starting with 19 or 20)
        # Group 2: end year or "present/current/now"
        year_range_pattern = r'(19|20)\d{2}\s*[-–—to]+\s*((19|20)\d{2}|present|current|now|ongoing)'

        matches = re.finditer(year_range_pattern, text, re.IGNORECASE)

        total_years = 0.0
        found_ranges = []

        for match in matches:
            full_match = match.group()

            # Extract start year
            start_match = re.search(r'(19|20)\d{2}', full_match)
            if not start_match:
                continue
            start_year = int(start_match.group())

            # Extract end year — check if "present/current/now"
            end_part = full_match.split(start_match.group())[1]  # everything after start
            if re.search(r'present|current|now|ongoing', end_part, re.IGNORECASE):
                end_year = CURRENT_YEAR
            else:
                end_match = re.search(r'(19|20)\d{2}', end_part)
                end_year = int(end_match.group()) if end_match else CURRENT_YEAR

            # Sanity check: start must be before end, both must be reasonable years
            if 1990 <= start_year < end_year <= CURRENT_YEAR + 1:
                duration = end_year - start_year
                found_ranges.append((start_year, end_year, duration))
                total_years += duration

        # If no year ranges found, look for "X years of experience" mentions
        if total_years == 0:
            explicit_pattern = r'(\d+)\+?\s*years?\s*(of\s+)?(experience|exp)'
            explicit_match = re.search(explicit_pattern, text, re.IGNORECASE)
            if explicit_match:
                total_years = float(explicit_match.group(1))

        return round(total_years, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # TEXT QUALITY ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_text_quality(self, text):
        """
        Computes writing quality metrics for the resume text.

        Metrics:
        - word_count              : total words
        - sentence_count          : total sentences
        - avg_word_length         : average characters per word
        - avg_sentence_length     : average words per sentence
        - vocabulary_richness     : Type-Token Ratio (unique/total words)
        - action_verb_count       : number of strong action verbs found
        - action_verb_percentage  : action verbs as % of total words

        Returns:
            dict of metric_name → value
        """
        if not text:
            return {}

        # Tokenise into words (only alphabetic words, no numbers/punctuation)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count = len(words)

        if word_count == 0:
            return {"word_count": 0}

        # Sentence detection: split on . ! ? followed by space or newline
        sentences = re.split(r'[.!?]+\s+|\n', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_count = max(len(sentences), 1)  # avoid division by zero

        # Average word length (character count)
        avg_word_length = sum(len(w) for w in words) / word_count

        # Average sentence length (word count per sentence)
        avg_sentence_length = word_count / sentence_count

        # Type-Token Ratio: unique words / total words
        # Lower case everything so "Python" and "python" count as the same token
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count

        # Count action verbs — lower case for comparison
        words_lower = {w.lower() for w in words}
        action_verb_count = len(words_lower.intersection(self.ACTION_VERBS))
        action_verb_percentage = (action_verb_count / word_count) * 100

        return {
            "word_count":             word_count,
            "sentence_count":         sentence_count,
            "avg_word_length":        round(avg_word_length, 2),
            "avg_sentence_length":    round(avg_sentence_length, 2),
            "vocabulary_richness":    round(ttr, 3),
            "action_verb_count":      action_verb_count,
            "action_verb_percentage": round(action_verb_percentage, 2),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SKILL SUMMARY
    # ──────────────────────────────────────────────────────────────────────────

    def get_skill_summary(self, text):
        """
        Returns a compact skills summary — total count, counts per category,
        and top 5 skills per category.

        Returns:
            dict with:
            - total_skills       : int
            - category_counts    : dict { category: count }
            - skills_by_category : dict { category: [skill, ...] }
            - top_skills         : dict { category: top_5_list }
        """
        skills_by_category = self.extract_skills(text)

        category_counts = {cat: len(skills) for cat, skills in skills_by_category.items()}
        total_skills = sum(category_counts.values())

        # Top 5 per category (already sorted alphabetically — no frequency data here)
        top_skills = {cat: skills[:5] for cat, skills in skills_by_category.items()}

        return {
            "total_skills":       total_skills,
            "category_counts":    category_counts,
            "skills_by_category": skills_by_category,
            "top_skills":         top_skills,
        }
