"""
ats_scorer.py — ATS Compatibility Scorer
==========================================
Concept: ATS (Applicant Tracking System) is software companies use to
automatically filter resumes BEFORE a human ever sees them.

Think of it like airport security:
- Your resume is the passenger
- The ATS is the scanner
- If it flags you, you never reach the gate (the hiring manager)

Studies show 75% of resumes are rejected by ATS before human review.
Common reasons:
- Missing keywords for the target role
- Tables/graphics that ATS can't parse (it reads left-to-right, top-to-bottom)
- Odd formatting: excessive symbols, inconsistent bullets
- Missing standard sections (no "Experience" section = instant reject)
- Contact info buried or missing

How weighted scoring works:
- Not all factors matter equally
- Keywords matter more than formatting
- We assign weights (must sum to 100%) and multiply each score by its weight
- Final score = sum of (sub_score × weight) for all factors

Example:
  keywords score = 80, weight = 0.25 → contributes 20 points
  contact score  = 100, weight = 0.15 → contributes 15 points
  ...etc
"""

import re
from collections import Counter


class ATSScorer:
    """
    Scores a resume for ATS compatibility on a 0-100 scale.

    Sub-scores and weights:
    - Section completeness : 20% (do required sections exist?)
    - Formatting quality   : 15% (clean text, no tables/symbols)
    - Keyword match        : 25% (role-relevant keywords present?)
    - Resume length        : 10% (not too short, not too long)
    - Readability          : 15% (sentence length, passive voice)
    - Contact info         : 15% (email, phone, LinkedIn, GitHub)
    """

    # ── Required sections — ATS WILL reject without these ─────────────────────
    # Each section name maps to keyword variants that might appear as headers
    REQUIRED_SECTIONS = {
        "contact":    ["contact", "email", "phone", "address", "linkedin", "gmail", "outlook"],
        "experience": ["experience", "work history", "employment", "career", "positions", "work experience"],
        "education":  ["education", "academic", "degree", "university", "college", "qualification"],
        "skills":     ["skills", "technical skills", "core competencies", "expertise", "technologies"],
    }

    # Optional but beneficial sections
    OPTIONAL_SECTIONS = {
        "summary":        ["summary", "objective", "profile", "about", "overview"],
        "projects":       ["projects", "portfolio", "personal projects"],
        "certifications": ["certifications", "certificates", "credentials", "licenses"],
        "achievements":   ["achievements", "accomplishments", "awards", "honors"],
        "publications":   ["publications", "research", "papers"],
    }

    # Role-specific keywords — used for keyword scoring when role is known
    ROLE_KEYWORDS = {
        "data_scientist": [
            "python", "r", "sql", "machine learning", "deep learning", "statistics",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "nlp",
            "feature engineering", "model deployment", "jupyter", "visualization",
            "a/b testing", "algorithms", "kaggle", "data analysis"
        ],
        "data_analyst": [
            "sql", "excel", "tableau", "power bi", "python", "statistics",
            "reporting", "dashboards", "analytics", "kpi", "etl",
            "data cleaning", "business intelligence", "r", "looker"
        ],
        "ml_engineer": [
            "python", "tensorflow", "pytorch", "docker", "kubernetes", "mlops",
            "aws", "model deployment", "ci/cd", "pipeline", "api",
            "microservices", "feature store", "model monitoring", "production"
        ],
        "software_engineer": [
            "python", "java", "javascript", "git", "agile", "api", "microservices",
            "docker", "kubernetes", "ci/cd", "testing", "sql", "rest",
            "system design", "scrum", "typescript"
        ],
        "frontend_developer": [
            "javascript", "typescript", "react", "vue", "angular", "html", "css",
            "responsive design", "webpack", "rest api", "git", "testing",
            "next.js", "performance", "accessibility"
        ],
        "backend_developer": [
            "python", "java", "node.js", "sql", "rest api", "docker",
            "kubernetes", "microservices", "postgresql", "redis", "aws",
            "authentication", "system design", "git", "testing"
        ],
    }

    # Score weights — must sum to 1.0
    WEIGHTS = {
        "sections":    0.20,
        "formatting":  0.15,
        "keywords":    0.25,
        "length":      0.10,
        "readability": 0.15,
        "contact":     0.15,
    }

    def __init__(self):
        # Store last results for get_improvement_suggestions()
        self._last_scores = {}
        self._last_feedback = []

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN SCORING METHOD
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_score(self, text, target_role=None):
        """
        Calculates overall ATS compatibility score.

        Args:
            text        : str — plain text resume content
            target_role : str or None — e.g. "data_scientist"
                          If None, auto-detected from content

        Returns:
            dict with:
            - overall_score  : float 0-100
            - sub_scores     : dict of individual scores
            - feedback       : list of feedback strings with emoji
            - grade          : letter grade (A+ to F)
            - pass_ats       : bool (True if score >= 60)
            - detected_role  : str — which role was used for keyword scoring
        """
        text_lower = text.lower()
        feedback = []

        # Auto-detect role if not provided
        detected_role = target_role or self._detect_role(text_lower)

        # Run all six sub-scorers
        section_score,    section_fb    = self._score_sections(text_lower)
        formatting_score, formatting_fb = self._score_formatting(text)
        keyword_score,    keyword_fb    = self._score_keywords(text_lower, detected_role)
        length_score,     length_fb     = self._score_length(text)
        readability_score,readability_fb= self._score_readability(text)
        contact_score,    contact_fb    = self._score_contact_info(text)

        # Collect all feedback
        feedback.extend(section_fb)
        feedback.extend(formatting_fb)
        feedback.extend(keyword_fb)
        feedback.extend(length_fb)
        feedback.extend(readability_fb)
        feedback.extend(contact_fb)

        # Build sub-scores dict
        sub_scores = {
            "sections":    round(section_score),
            "formatting":  round(formatting_score),
            "keywords":    round(keyword_score),
            "length":      round(length_score),
            "readability": round(readability_score),
            "contact":     round(contact_score),
        }

        # Weighted overall score
        overall = sum(
            sub_scores[key] * self.WEIGHTS[key]
            for key in sub_scores
        )
        overall = round(min(100, max(0, overall)), 1)

        # Store for improvement suggestions
        self._last_scores = sub_scores
        self._last_feedback = feedback

        return {
            "overall_score": overall,
            "sub_scores":    sub_scores,
            "feedback":      feedback,
            "grade":         self._get_grade(overall),
            "pass_ats":      overall >= 60,
            "detected_role": detected_role,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-SCORERS
    # ──────────────────────────────────────────────────────────────────────────

    def _score_sections(self, text):
        """
        Checks presence of required and optional resume sections.

        Scoring:
        - Required sections: 60 points total (15 pts each × 4 sections)
        - Optional sections: 40 points total (8 pts each × 5 sections)

        Returns:
            (float score 0-100, list of feedback strings)
        """
        score = 0
        feedback = []

        # Check required sections (15 pts each = 60 pts max)
        for section, keywords in self.REQUIRED_SECTIONS.items():
            pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            if re.search(pattern, text, re.IGNORECASE):
                score += 15
                feedback.append(f"✅ {section.title()} section found")
            else:
                feedback.append(f"⚠️  Missing {section.title()} section — ATS may reject your resume")

        # Check optional sections (8 pts each = 40 pts max)
        for section, keywords in self.OPTIONAL_SECTIONS.items():
            pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            if re.search(pattern, text, re.IGNORECASE):
                score += 8
                feedback.append(f"✅ {section.title()} section found (bonus)")

        return min(score, 100), feedback

    def _score_formatting(self, text):
        """
        Penalises formatting patterns that confuse ATS parsers.

        ATS reads text linearly — tables, graphics, and special symbols
        break the parser and cause skills/experience to be misread or skipped.

        Returns:
            (float score 0-100, list of feedback strings)
        """
        score = 100.0
        feedback = []

        # Penalty: too many special characters (symbols, brackets, etc.)
        # Normal resumes have ~2-5% special chars. More = formatting-heavy
        special_char_count = len(re.findall(r'[^\w\s\n\.,;:\-\(\)@/]', text))
        word_count = max(len(text.split()), 1)
        special_ratio = special_char_count / word_count

        if special_ratio > 0.15:
            score -= 15
            feedback.append("⚠️  Too many special characters — may confuse ATS parser")

        # Penalty: inconsistent bullet point styles
        # Mixing •, -, *, ▶ in same document = poor formatting consistency
        bullet_types = set(re.findall(r'^[\s]*([•\-\*▶►▸➤➢◆◇■□])', text, re.MULTILINE))
        if len(bullet_types) > 2:
            score -= 10
            feedback.append("⚠️  Inconsistent bullet styles — standardise to one style")

        # Penalty: excessive ALL CAPS (shouting, looks unprofessional to ATS)
        words = text.split()
        if len(words) > 0:
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
            if caps_ratio > 0.15:
                score -= 10
                feedback.append("⚠️  Excessive ALL CAPS text — ATS may misparse capitalized content")

        # Penalty: table-like pipe formatting (| col1 | col2 |)
        # Tables render visually but ATS reads them as garbled text
        pipe_lines = len(re.findall(r'\|.+\|', text))
        if pipe_lines > 3:
            score -= 15
            feedback.append("⚠️  Table formatting detected — ATS cannot parse tables correctly")

        if score >= 90:
            feedback.append("✅ Formatting looks clean and ATS-friendly")

        return max(score, 0), feedback

    def _score_keywords(self, text, target_role):
        """
        Checks how many role-relevant keywords appear in the resume.

        Why this matters: ATS systems score resumes by counting keyword matches
        against the job description. No keywords = no callback.

        Returns:
            (float score 0-100, list of feedback strings)
        """
        feedback = []

        if not target_role or target_role not in self.ROLE_KEYWORDS:
            feedback.append("💡 No target role detected — add role-specific keywords")
            return 50, feedback  # neutral score when role unknown

        keywords = self.ROLE_KEYWORDS[target_role]
        found = []
        missing = []

        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found.append(kw)
            else:
                missing.append(kw)

        total = len(keywords)
        found_count = len(found)
        score = (found_count / total) * 100 if total > 0 else 50

        feedback.append(f"✅ {found_count}/{total} keywords found for {target_role.replace('_', ' ').title()}")

        # Show top 5 missing keywords with suggestion emoji
        if missing:
            top_missing = missing[:5]
            for kw in top_missing:
                feedback.append(f"💡 Consider adding keyword: '{kw}'")

        return score, feedback

    def _score_length(self, text):
        """
        Scores based on word count. ATS and humans both prefer concise resumes.

        Ideal: 400-800 words (roughly 1-2 pages of dense text)
        Too short = not enough detail
        Too long  = padding, hard to parse

        Returns:
            (float score 0-100, list of feedback strings)
        """
        word_count = len(text.split())
        feedback = []

        if 400 <= word_count <= 800:
            score = 100
            feedback.append(f"✅ Good resume length: {word_count} words")
        elif 300 <= word_count < 400 or 800 < word_count <= 1000:
            score = 80
            feedback.append(f"⚠️  Resume length ({word_count} words) is slightly off — aim for 400-800 words")
        elif 200 <= word_count < 300:
            score = 60
            feedback.append(f"⚠️  Resume too short ({word_count} words) — add more detail to experience/projects")
        elif word_count > 1000:
            score = 60
            feedback.append(f"⚠️  Resume too long ({word_count} words) — trim to 1-2 pages for best ATS results")
        else:
            score = 40
            feedback.append(f"⚠️  Resume very short ({word_count} words) — needs significantly more content")

        return score, feedback

    def _score_readability(self, text):
        """
        Checks sentence structure — ATS and humans both struggle with very long
        or very short sentences.

        Also checks passive voice ratio — active voice ("Built X") is preferred
        over passive voice ("X was built by me").

        Returns:
            (float score 0-100, list of feedback strings)
        """
        score = 100.0
        feedback = []

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+|\n', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 50, ["⚠️  Could not analyse readability"]

        # Average sentence length in words
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_len > 30:
            score -= 20
            feedback.append(f"⚠️  Sentences too long (avg {avg_len:.0f} words) — break them up for clarity")
        elif avg_len < 10:
            score -= 10
            feedback.append(f"⚠️  Sentences very short (avg {avg_len:.0f} words) — add more detail")
        else:
            feedback.append(f"✅ Sentence length looks good (avg {avg_len:.0f} words)")

        # Passive voice detection — common passive patterns
        passive_patterns = [
            r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
            r'\bbeen\s+\w+ed\b', r'\bis\s+\w+ed\b',
            r'\bare\s+\w+ed\b',
        ]
        passive_count = sum(
            len(re.findall(p, text, re.IGNORECASE))
            for p in passive_patterns
        )
        passive_ratio = passive_count / max(len(sentences), 1)

        if passive_ratio > 0.05:
            score -= 10
            feedback.append("⚠️  High passive voice usage — prefer active voice ('Built X' over 'X was built')")

        return max(score, 0), feedback

    def _score_contact_info(self, text):
        """
        Checks for presence of contact information.
        ATS needs this to route your resume correctly.

        Points: email=30, phone=25, LinkedIn=25, GitHub/portfolio=20

        Returns:
            (float score 0-100, list of feedback strings)
        """
        score = 0
        feedback = []

        # Email
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            score += 30
            feedback.append("✅ Email address found")
        else:
            feedback.append("⚠️  No email found — essential for ATS and recruiters")

        # Phone number
        if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text):
            score += 25
            feedback.append("✅ Phone number found")
        else:
            feedback.append("⚠️  No phone number found")

        # LinkedIn
        if re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE):
            score += 25
            feedback.append("✅ LinkedIn profile found")
        else:
            feedback.append("💡 Add LinkedIn profile URL to boost recruiter trust")

        # GitHub or portfolio
        if re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE) or \
           re.search(r'\b(portfolio|website|blog)\b', text, re.IGNORECASE):
            score += 20
            feedback.append("✅ GitHub/portfolio found")
        else:
            feedback.append("💡 Add GitHub profile or portfolio link")

        return score, feedback

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_role(self, text):
        """
        Auto-detects the most likely target role by counting keyword matches.

        Returns:
            str — role key with highest keyword match count
        """
        role_scores = {}
        for role, keywords in self.ROLE_KEYWORDS.items():
            count = sum(
                1 for kw in keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)
            )
            role_scores[role] = count

        # Return role with highest match count, default to software_engineer
        return max(role_scores, key=role_scores.get) if role_scores else "software_engineer"

    def _get_grade(self, score):
        """Converts numeric score to letter grade."""
        if score >= 90: return "A+"
        if score >= 85: return "A"
        if score >= 80: return "A-"
        if score >= 75: return "B+"
        if score >= 70: return "B"
        if score >= 65: return "B-"
        if score >= 60: return "C+"
        if score >= 55: return "C"
        if score >= 50: return "C-"
        if score >= 45: return "D"
        return "F"

    def get_improvement_suggestions(self):
        """
        Returns prioritised improvement suggestions based on lowest sub-scores.
        Call this after calculate_score().

        Returns:
            list of suggestion strings, ordered by priority (worst scores first)
        """
        if not self._last_scores:
            return ["Run calculate_score() first"]

        suggestions = []
        sorted_scores = sorted(self._last_scores.items(), key=lambda x: x[1])

        for section, score in sorted_scores:
            if score < 60:
                if section == "keywords":
                    suggestions.append("🔴 HIGH PRIORITY: Add more role-specific keywords to match job descriptions")
                elif section == "sections":
                    suggestions.append("🔴 HIGH PRIORITY: Add missing resume sections (especially Experience and Skills)")
                elif section == "contact":
                    suggestions.append("🔴 HIGH PRIORITY: Complete your contact information")
                elif section == "formatting":
                    suggestions.append("🟡 Fix formatting — remove tables, standardise bullets, reduce special characters")
                elif section == "readability":
                    suggestions.append("🟡 Improve readability — use shorter sentences and active voice")
                elif section == "length":
                    suggestions.append("🟡 Adjust resume length — aim for 400-800 words")

        if not suggestions:
            suggestions.append("✅ Resume is well-optimised! Focus on tailoring keywords to each job application.")

        return suggestions
