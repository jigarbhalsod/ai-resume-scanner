"""
Resume Scanner Package
======================
A Python package for intelligent resume analysis using NLP, TF-IDF,
and rule-based scoring systems.

What __init__.py does:
- Marks this folder as a Python "package" (importable module)
- Controls what gets exposed when someone does `from resume_scanner import ...`
- Acts like a reception desk — you come here first, it routes you to the right room

__all__ controls what's exported when someone does `from resume_scanner import *`
Without __all__, everything would be exported — messy and hard to maintain.
"""

__version__ = "1.0.0"
__author__ = "Jigar Bhalsod"

# Import all five core classes so users can do:
# from resume_scanner import ResumeParser
# instead of:
# from resume_scanner.parser import ResumeParser
from resume_scanner.parser import ResumeParser
from resume_scanner.nlp_engine import NLPEngine
from resume_scanner.ats_scorer import ATSScorer
from resume_scanner.ai_detector import AIDetector
from resume_scanner.job_matcher import JobMatcher

# Explicitly declare public API — only these five classes are meant for external use
__all__ = [
    "ResumeParser",
    "NLPEngine",
    "ATSScorer",
    "AIDetector",
    "JobMatcher",
]
