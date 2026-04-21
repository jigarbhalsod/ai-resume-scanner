"""
job_matcher.py — TF-IDF Job Role Matcher
==========================================

TWO KEY CONCEPTS explained before the code:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT 1: TF-IDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TF-IDF = Term Frequency × Inverse Document Frequency

Problem it solves:
  The word "the" appears in EVERY document. It tells us nothing about content.
  The word "pytorch" only appears in ML documents. It's very informative.
  TF-IDF gives HIGH scores to words that are frequent in THIS document
  but RARE across all documents.

TF (Term Frequency):
  How often does word X appear in THIS document?
  TF("python", resume) = count("python" in resume) / total_words_in_resume

IDF (Inverse Document Frequency):
  How rare is word X across ALL documents?
  IDF("python") = log(total_docs / docs_containing_python)
  Common words like "the" → low IDF (appears in all docs)
  Rare words like "pytorch" → high IDF (appears in few docs)

TF-IDF score = TF × IDF

Worked example:
  5 job descriptions total.
  "python" appears in all 5 → IDF = log(5/5) = log(1) = 0 (useless word)
  Wait — that's wrong! Python IS useful.
  Fix: use smoothed IDF = log(5 / (1 + 2)) = log(1.67) = 0.51

  In the resume, "python" appears 3 times out of 200 words:
  TF = 3/200 = 0.015
  TF-IDF = 0.015 × 0.51 = 0.0077

  "kubernetes" appears in 2 of 5 docs:
  IDF = log(5 / (1 + 2)) = 0.51 (similar)
  But in THIS resume, it appears 5 times:
  TF = 5/200 = 0.025
  TF-IDF = 0.025 × 0.51 = 0.013 — higher score — more distinctive!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT 2: COSINE SIMILARITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After TF-IDF, each document becomes a VECTOR of numbers.
  resume_vector      = {"python": 0.02, "docker": 0.015, "sql": 0.01, ...}
  data_scientist_vec = {"python": 0.03, "sql": 0.025, "tensorflow": 0.02, ...}

Cosine similarity measures the ANGLE between two vectors.
  - Angle of 0°  → cos(0)   = 1.0 → identical direction → perfect match
  - Angle of 90° → cos(90°) = 0.0 → perpendicular → no overlap at all

Formula:
  cosine_similarity = (A · B) / (|A| × |B|)

  Where:
  A · B  = dot product = sum of (a_i × b_i) for shared words
  |A|    = magnitude of A = sqrt(sum of a_i²)
  |B|    = magnitude of B = sqrt(sum of b_i²)

Why cosine over Euclidean distance?
  Cosine ignores document length — a short resume and long resume with
  the same topics score equally. Euclidean would penalise the shorter one.

Why build TF-IDF from scratch instead of sklearn?
  sklearn does this in 1 line. But building it yourself means you understand:
  - What a vocabulary is (set of all unique words across all docs)
  - What IDF is (how we penalise common words)
  - Why cosine similarity works geometrically
  Once you understand it, you can tune it, debug it, and extend it.
"""

import re
import math
from collections import Counter


class JobMatcher:
    """
    Matches a resume to job roles using TF-IDF vectors + cosine similarity.

    Workflow:
    1. __init__: build vocabulary + IDF from all job descriptions
    2. match(resume_text): compute resume TF-IDF, compare to each job, rank results
    """

    # ── Job descriptions — keyword-rich text for each role ────────────────────
    # These act as "ideal candidate" documents for each role.
    # The richer and more keyword-dense these are, the better the matching.
    JOB_DESCRIPTIONS = {
        "Data Scientist": """
            machine learning deep learning python r sql statistics tensorflow pytorch
            pandas numpy scikit-learn visualization nlp computer vision feature engineering
            model deployment a/b testing jupyter kaggle algorithms data analysis
            hypothesis testing regression classification clustering neural networks
            gradient boosting random forest xgboost time series forecasting
            experimental design statistical modelling business intelligence
        """,
        "ML Engineer": """
            machine learning python tensorflow pytorch docker kubernetes mlops
            model deployment aws gcp azure ci/cd pipeline api microservices
            feature store model monitoring production scalability distributed systems
            data pipeline spark airflow kafka streaming infrastructure automation
            gpu training inference optimization latency throughput batch processing
            model versioning experiment tracking mlflow wandb argo workflows
        """,
        "Data Analyst": """
            sql excel tableau power bi python r visualization reporting dashboards
            analytics business intelligence kpi metrics etl data cleaning
            statistical analysis stakeholders data storytelling pivot tables
            google analytics looker metabase data driven decision making
            ad hoc analysis trend analysis cohort analysis funnel analysis
        """,
        "Software Engineer": """
            python java javascript typescript c++ git agile scrum api rest
            microservices docker kubernetes ci/cd testing debugging sql nosql
            system design algorithms data structures object oriented programming
            code review pull requests continuous integration deployment
            backend frontend full stack web development cloud aws azure gcp
        """,
        "Data Engineer": """
            python sql spark hadoop airflow etl pipeline data warehouse aws gcp
            bigquery snowflake kafka streaming data modeling schema design
            redshift azure databricks dbt data lake batch processing
            data quality data governance orchestration transformation ingestion
            postgresql mongodb distributed computing cloud infrastructure
        """,
        "AI/ML Research": """
            research publications neural networks deep learning transformers nlp
            computer vision reinforcement learning pytorch tensorflow mathematics
            statistics algorithms optimisation paper writing arxiv experiments
            ablation study benchmark evaluation fine-tuning pre-training
            attention mechanism self-supervised learning contrastive learning
            generative models diffusion models language models bert gpt llm
        """,
    }

    def __init__(self):
        """
        Initialises the matcher by building vocabulary and IDF scores
        from all job descriptions.

        This runs once at startup — not per resume. Good for performance.
        """
        self.vocabulary = set()
        self.idf_scores = {}
        self._build_vocabulary()

    # ──────────────────────────────────────────────────────────────────────────
    # VOCABULARY AND IDF BUILDING
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vocabulary(self):
        """
        Step 1: Collect all unique words across all job descriptions.
        Step 2: Calculate IDF for each word.

        IDF formula: log(total_docs / (1 + docs_containing_word))
        The +1 prevents division by zero (Laplace smoothing).
        """
        total_docs = len(self.JOB_DESCRIPTIONS)
        # Track how many documents each word appears in
        word_doc_freq = Counter()

        tokenised_docs = {}
        for role, description in self.JOB_DESCRIPTIONS.items():
            tokens = self._tokenize(description)
            tokenised_docs[role] = tokens
            self.vocabulary.update(tokens)
            # Count each word only ONCE per document (set removes duplicates)
            for word in set(tokens):
                word_doc_freq[word] += 1

        # Calculate IDF for each word in vocabulary
        for word in self.vocabulary:
            doc_freq = word_doc_freq.get(word, 0)
            # log(N / (1 + df)) — higher IDF = rarer word = more informative
            self.idf_scores[word] = math.log(total_docs / (1 + doc_freq))

    # ──────────────────────────────────────────────────────────────────────────
    # TEXT PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _tokenize(self, text):
        """
        Converts text to a list of lowercase tokens.
        Filters out words shorter than 3 characters (removes "is", "to", "a", etc.)

        Args:
            text : str

        Returns:
            list of str tokens
        """
        # \b[a-z]+\b matches only alphabetic words (no numbers, symbols)
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter out very short words that carry little meaning
        return [t for t in tokens if len(t) >= 3]

    def _calculate_tf(self, words):
        """
        Calculates Term Frequency for a list of words.
        TF = count(word) / total_words

        Args:
            words : list of str

        Returns:
            dict { word: tf_score }
        """
        total = len(words)
        if total == 0:
            return {}
        counts = Counter(words)
        return {word: count / total for word, count in counts.items()}

    def _calculate_tfidf(self, text):
        """
        Computes TF-IDF vector for a given text.

        Steps:
        1. Tokenise text
        2. Calculate TF for each word
        3. Multiply TF by pre-computed IDF
        4. Only keep words that exist in our vocabulary

        Args:
            text : str

        Returns:
            dict { word: tfidf_score }
        """
        tokens = self._tokenize(text)
        tf = self._calculate_tf(tokens)

        tfidf = {}
        for word, tf_score in tf.items():
            # Only score words we've seen in job descriptions (in vocabulary)
            if word in self.idf_scores:
                tfidf[word] = tf_score * self.idf_scores[word]

        return tfidf

    # ──────────────────────────────────────────────────────────────────────────
    # COSINE SIMILARITY
    # ──────────────────────────────────────────────────────────────────────────

    def _cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two TF-IDF vectors.

        Formula: (A · B) / (|A| × |B|)

        Only words present in BOTH vectors contribute to dot product.
        Words in only one vector → zero contribution (like perpendicular dimensions).

        Args:
            vec1 : dict { word: score }
            vec2 : dict { word: score }

        Returns:
            float 0.0 to 1.0 (1.0 = identical direction = perfect match)
        """
        if not vec1 or not vec2:
            return 0.0

        # Dot product: sum of (score1 × score2) for words in BOTH vectors
        common_words = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[w] * vec2[w] for w in common_words)

        # Magnitudes: sqrt of sum of squares for each vector
        magnitude1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN MATCHING METHOD
    # ──────────────────────────────────────────────────────────────────────────

    def match(self, resume_text):
        """
        Matches resume text against all job descriptions.

        Steps:
        1. Compute TF-IDF for the resume
        2. Compute TF-IDF for each job description
        3. Calculate cosine similarity between resume and each job
        4. Sort by similarity score descending
        5. Return ranked matches + recommendations

        Args:
            resume_text : str — plain text resume content

        Returns:
            dict with:
            - best_match     : str — top matching role name
            - best_score     : float — similarity as percentage (0-100)
            - all_matches    : list of { role, score, percentage } sorted by score
            - recommendations: list of suggestion strings
            - keyword_analysis: dict showing present/missing keywords for best match
        """
        # Compute TF-IDF for the resume
        resume_tfidf = self._calculate_tfidf(resume_text)

        matches = []
        for role, description in self.JOB_DESCRIPTIONS.items():
            # Compute TF-IDF for this job description
            job_tfidf = self._calculate_tfidf(description)

            # Cosine similarity between resume and job description
            similarity = self._cosine_similarity(resume_tfidf, job_tfidf)

            # Convert to percentage and round
            percentage = round(similarity * 100, 1)

            matches.append({
                "role":       role,
                "score":      similarity,
                "percentage": percentage,
            })

        # Sort by score descending — best match first
        matches.sort(key=lambda x: x["score"], reverse=True)

        best_match = matches[0] if matches else None
        best_role = best_match["role"] if best_match else "Unknown"
        best_score = best_match["percentage"] if best_match else 0

        # Keyword gap analysis for best match
        keyword_analysis = self._analyze_keywords(resume_text, best_role)

        return {
            "best_match":       best_role,
            "best_score":       best_score,
            "all_matches":      matches,
            "recommendations":  self._get_recommendations(matches, resume_text),
            "keyword_analysis": keyword_analysis,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _get_recommendations(self, matches, resume_text):
        """
        Generates actionable recommendations based on match quality.

        Args:
            matches     : list of match dicts (sorted by score)
            resume_text : str

        Returns:
            list of recommendation strings
        """
        recommendations = []

        if not matches:
            return ["Unable to generate recommendations — no matches found"]

        top_match = matches[0]
        top_score = top_match["percentage"]
        top_role = top_match["role"]

        if top_score >= 70:
            recommendations.append(
                f"✅ Strong fit for {top_role} ({top_score}% match) — tailor your resume title to this role"
            )
        elif top_score >= 50:
            recommendations.append(
                f"🟡 Good potential for {top_role} ({top_score}% match) — add more role-specific keywords"
            )
        else:
            recommendations.append(
                f"🔴 Low match scores — consider expanding your skills section with relevant technologies"
            )

        # Check if second match is close — might be worth targeting both
        if len(matches) > 1:
            second = matches[1]
            if second["percentage"] >= top_score * 0.85:  # within 15% of top
                recommendations.append(
                    f"💡 Also consider {second['role']} ({second['percentage']}% match) — scores are close"
                )

        return recommendations

    def _analyze_keywords(self, resume_text, role):
        """
        For the best matching role, identifies which keywords from the
        job description are present vs missing in the resume.

        Args:
            resume_text : str
            role        : str — job role name

        Returns:
            dict with:
            - present : list of keywords found in resume
            - missing : list of keywords NOT found in resume
        """
        if role not in self.JOB_DESCRIPTIONS:
            return {"present": [], "missing": []}

        # Tokenise job description to get its key terms
        job_tokens = set(self._tokenize(self.JOB_DESCRIPTIONS[role]))
        resume_lower = resume_text.lower()

        present = []
        missing = []

        for token in sorted(job_tokens):
            if len(token) < 4:  # skip very short words
                continue
            pattern = r'\b' + re.escape(token) + r'\b'
            if re.search(pattern, resume_lower):
                present.append(token)
            else:
                missing.append(token)

        return {
            "present": present[:20],  # top 20 to keep UI clean
            "missing": missing[:20],
        }
