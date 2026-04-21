"""
ai_detector.py — AI-Generated Content Detector
================================================
Concept: AI language models (ChatGPT, Claude, Gemini) generate text that
sounds polished but has statistical fingerprints humans don't naturally produce.

Think of it like handwriting analysis — a forgery might look right to the eye,
but microscopic patterns reveal it's not genuine.

Four signals we measure:

1. AI PHRASE DETECTION
   AI models are trained on internet text which is full of corporate buzzwords.
   They overuse phrases like "leveraging cutting-edge" and "results-driven"
   because these appear frequently in their training data.
   Humans writing authentically rarely use these exact phrases.

2. OVERUSED VERB DETECTION
   AI loves specific "impressive-sounding" verbs: spearheaded, orchestrated,
   catalyzed, pioneered. Real people say "built", "wrote", "fixed", "shipped".
   Seeing 3+ of these in one resume is a strong AI signal.

3. TYPE-TOKEN RATIO (TTR)
   As explained in nlp_engine.py:
   TTR = unique words / total words
   AI generates smooth, varied-sounding text but actually repeats sentence
   structures and vocabulary more than humans do.
   Human resume: TTR ~0.55-0.75
   AI resume:    TTR ~0.35-0.50

4. REPETITION PATTERN
   AI tends to start consecutive sentences the same way:
   "Spearheaded the launch... Developed the pipeline... Implemented the system..."
   Each bullet starts with a past-tense verb in the same structure.
   Humans vary their sentence openings more naturally.

What is "perplexity" (conceptually)?
   In language models, perplexity measures how "surprised" the model is by text.
   AI-generated text has LOW perplexity — the model predicted those exact words.
   Human text has HIGHER perplexity — more unexpected word choices.
   We don't implement this directly (needs a full LM), but TTR approximates it.
"""

import re
from collections import Counter


class AIDetector:
    """
    Detects likelihood of AI-generated content in resume text.
    Returns probability 0-100 where 100 = definitely AI-generated.
    """

    # ── Phrases that AI models overuse ────────────────────────────────────────
    # These come from the same patterns identified in humanizer tools —
    # AI writes like a corporate brochure, not like a real person
    AI_PHRASES = [
        "leveraging cutting-edge",
        "spearheaded initiatives",
        "drove strategic",
        "fostered collaborative",
        "orchestrated seamless",
        "catalyzed growth",
        "synergized efforts",
        "pioneered innovative",
        "championed digital",
        "cultivated relationships",
        "streamlined operations",
        "optimized workflows",
        "passionate about",
        "dedicated professional",
        "results-driven",
        "detail-oriented",
        "highly motivated",
        "proven track record",
        "dynamic professional",
        "thought leader",
        "value-added",
        "go-to person",
        "exceptional communication skills",
        "team player",
        "fast-paced environment",
        "cross-functional",
        "end-to-end",
        "best-in-class",
        "paradigm shift",
        "robust solution",
    ]

    # ── Verbs that AI loves but real people rarely write ──────────────────────
    OVERUSED_VERBS = [
        "leveraged",
        "spearheaded",
        "orchestrated",
        "synergized",
        "catalyzed",
        "pioneered",
        "championed",
        "cultivated",
        "revolutionized",
        "transformed",
        "ideated",
        "evangelized",
        "socialized",      # in the corporate "shared with stakeholders" sense
        "operationalized",
        "incentivized",
    ]

    def analyze(self, text):
        """
        Runs all four AI detection checks and combines into final probability.

        Weighted combination:
        - AI phrases:     30% (strong signal — humans don't write these)
        - Overused verbs: 20% (moderate signal)
        - Low TTR:        25% (statistical signal — reversed: low TTR = high AI)
        - Repetition:     25% (structural signal — AI has repetitive patterns)

        Args:
            text : str — plain text resume content

        Returns:
            dict with:
            - ai_probability   : float 0-100
            - verdict          : str description
            - confidence       : str ("High", "Medium", "Low")
            - phrase_score     : float 0-100
            - verb_score       : float 0-100
            - ttr_score        : float 0-100 (higher = more human)
            - repetition_score : float 0-100 (higher = more repetitive = more AI)
            - flags            : list of human-readable warning strings
        """
        if not text or len(text.strip()) < 50:
            return {
                "ai_probability": 0,
                "verdict": "Insufficient text to analyze",
                "confidence": "Low",
                "phrase_score": 0,
                "verb_score": 0,
                "ttr_score": 100,
                "repetition_score": 0,
                "flags": [],
            }

        # Run four independent checks
        phrase_score     = self._check_ai_phrases(text)
        verb_score       = self._check_overused_verbs(text)
        ttr_score        = self._calculate_ttr(text)        # HIGH = human, LOW = AI
        repetition_score = self._check_repetition(text)

        # Combine scores — note: ttr_score is inverted (100 - ttr_score)
        # because high TTR means human, but we want high score = more AI
        ai_probability = (
            phrase_score     * 0.30 +
            verb_score       * 0.20 +
            (100 - ttr_score) * 0.25 +
            repetition_score * 0.25
        )

        # Clamp to 0-100 range
        ai_probability = round(min(100, max(0, ai_probability)), 1)

        return {
            "ai_probability":   ai_probability,
            "verdict":          self._get_verdict(ai_probability),
            "confidence":       self._get_confidence(ai_probability),
            "phrase_score":     round(phrase_score, 1),
            "verb_score":       round(verb_score, 1),
            "ttr_score":        round(ttr_score, 1),
            "repetition_score": round(repetition_score, 1),
            "flags":            self._get_flags(text),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # FOUR DETECTION CHECKS
    # ──────────────────────────────────────────────────────────────────────────

    def _check_ai_phrases(self, text):
        """
        Counts how many known AI phrases appear in the text.

        Scoring thresholds:
        ≥6 phrases → 90 (very likely AI)
        ≥4 phrases → 70
        ≥2 phrases → 45
        ≥1 phrase  → 25
        0 phrases  → 10

        Returns:
            float — AI likelihood score 0-100
        """
        text_lower = text.lower()
        found_count = 0
        for phrase in self.AI_PHRASES:
            if phrase.lower() in text_lower:
                found_count += 1

        if found_count >= 6: return 90
        if found_count >= 4: return 70
        if found_count >= 2: return 45
        if found_count >= 1: return 25
        return 10

    def _check_overused_verbs(self, text):
        """
        Counts overused AI-favoured verbs in the text.

        Uses word boundary matching to avoid false matches
        (e.g. "pioneered" shouldn't match "pioneering" differently — both are caught)

        Scoring thresholds:
        ≥5 verbs → 85
        ≥3 verbs → 60
        ≥1 verb  → 30
        0 verbs  → 10

        Returns:
            float — AI likelihood score 0-100
        """
        text_lower = text.lower()
        found_count = 0
        for verb in self.OVERUSED_VERBS:
            # Match verb and its variations (leveraged, leveraging, leverage)
            pattern = r'\b' + re.escape(verb.rstrip('ed')) + r'\w*\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_count += 1

        if found_count >= 5: return 85
        if found_count >= 3: return 60
        if found_count >= 1: return 30
        return 10

    def _calculate_ttr(self, text):
        """
        Calculates Type-Token Ratio (TTR) and converts to human-likeness score.

        TTR = unique_words / total_words

        Normalisation:
        - Raw TTR of 0.3 = very repetitive = 0 (AI)
        - Raw TTR of 0.7 = very diverse    = 100 (human)
        - Formula: (ttr - 0.3) / 0.4 × 100, clamped 0-100

        Returns:
            float — human-likeness score 0-100 (higher = more human)
        """
        # Extract only alphabetic words (no numbers, symbols)
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Not enough words for reliable TTR
        if len(words) < 50:
            return 50  # neutral score

        total_words = len(words)
        unique_words = len(set(words))
        ttr = unique_words / total_words

        # Normalise to 0-100 scale
        # 0.3 maps to 0, 0.7 maps to 100
        normalised = (ttr - 0.3) / 0.4 * 100
        return round(min(100, max(0, normalised)), 1)

    def _check_repetition(self, text):
        """
        Detects repetitive sentence-start patterns.

        AI tends to structure bullet points identically:
        "Developed X... Developed Y... Implemented Z... Implemented W..."

        Method:
        1. Split into sentences longer than 20 chars
        2. Extract first word of each sentence
        3. Find the most repeated first word
        4. Score = (max_same_starts / sentence_count) × 100

        Example:
        10 sentences, 4 start with "Developed" → score = 40

        Returns:
            float — repetition score 0-100 (higher = more repetitive = more AI)
        """
        # Split on sentence-ending punctuation or newlines
        sentences = re.split(r'[.!?\n]+', text)
        # Keep only sentences with real content (>20 chars)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) < 3:
            return 0  # too few sentences to detect a pattern

        # Get first word of each sentence (lowercase for comparison)
        first_words = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                first_words.append(words[0].lower().rstrip('.,;:'))

        if not first_words:
            return 0

        # Count frequency of each first word
        word_counts = Counter(first_words)
        most_common_count = word_counts.most_common(1)[0][1]

        # Score = how often the most repeated word appears as a sentence start
        repetition_ratio = most_common_count / len(sentences)
        score = repetition_ratio * 100

        return round(min(100, score), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _get_confidence(self, probability):
        """
        Returns confidence level based on how extreme the probability is.
        Extreme scores (very high or very low) = high confidence.
        Middle scores = low confidence (ambiguous).
        """
        if probability >= 70 or probability <= 20:
            return "High"
        if probability >= 45:
            return "Medium"
        return "Low"

    def _get_verdict(self, probability):
        """Returns human-readable verdict string based on AI probability."""
        if probability >= 70: return "Likely AI-Generated"
        if probability >= 45: return "Possibly AI-Assisted"
        if probability >= 25: return "Mixed Human/AI"
        return "Likely Human-Written"

    def _get_flags(self, text):
        """
        Returns a list of specific, human-readable flags describing
        what was detected in the text.

        Returns:
            list of str — each flag describes a specific AI signal found
        """
        flags = []
        text_lower = text.lower()

        # Flag specific AI phrases found
        found_phrases = [p for p in self.AI_PHRASES if p.lower() in text_lower]
        if found_phrases:
            flags.append(f"AI phrases detected: {', '.join(found_phrases[:4])}")

        # Flag overused verbs found
        found_verbs = []
        for verb in self.OVERUSED_VERBS:
            pattern = r'\b' + re.escape(verb.rstrip('ed')) + r'\w*\b'
            if re.search(pattern, text_lower):
                found_verbs.append(verb)
        if found_verbs:
            flags.append(f"Overused verbs: {', '.join(found_verbs[:5])}")

        # Flag low TTR
        words = re.findall(r'\b[a-z]+\b', text_lower)
        if len(words) >= 50:
            ttr = len(set(words)) / len(words)
            if ttr < 0.45:
                flags.append(f"Low vocabulary diversity (TTR: {ttr:.2f}) — may indicate AI writing")

        # Flag sentence repetition
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if len(s.strip()) > 20]
        if sentences:
            first_words = [s.split()[0].lower() for s in sentences if s.split()]
            counts = Counter(first_words)
            if counts and counts.most_common(1)[0][1] >= 3:
                top_word, top_count = counts.most_common(1)[0]
                flags.append(f"Repetitive sentence starts: '{top_word}' begins {top_count} sentences")

        return flags
