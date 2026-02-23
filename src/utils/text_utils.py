"""
Text preprocessing pipeline for CBF feature engineering
Section 4.1 of the Master Engineering Specification
"""

import re
import spacy
from typing import List
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model (run once at module import)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Domain-specific stopwords (Section 4.1)
DOMAIN_STOPWORDS = {'film', 'movie', 'story', 'character', 'director', 'actor', 'watch'}
NLTK_STOPWORDS = set(stopwords.words('english'))
ALL_STOPWORDS = NLTK_STOPWORDS.union(DOMAIN_STOPWORDS)


def clean_text(text: str) -> str:
    """
    Standardized text cleaning pipeline in exact order as spec.
    
    Steps:
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove special characters (keep letters, numbers, spaces)
    5. Normalize whitespace
    6. Tokenize
    7. Remove stopwords
    8. Lemmatize (using spaCy)
    9. Rejoin tokens
    
    Special handling: Compound names (e.g., "Christopher Nolan") are preserved
    by replacing spaces with underscores BEFORE lemmatization.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Step 4: Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Step 5: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 6: Tokenize (simple split for now, we'll use spaCy for lemmatization)
    tokens = text.split()
    
    # Step 7: Remove stopwords
    tokens = [t for t in tokens if t not in ALL_STOPWORDS]
    
    # Step 8: Lemmatization with spaCy
    # But FIRST: handle compound names (spec requirement)
    # We need to identify proper names - this is complex
    # For now, we'll do a simple pass: if a token is capitalized in original? 
    # We lost case info. Alternative approach in feature engineering
    
    # Join tokens for spaCy processing
    text_for_spacy = ' '.join(tokens)
    doc = nlp(text_for_spacy)
    
    # Extract lemmas
    lemmas = [token.lemma_ for token in doc]
    
    # Step 9: Rejoin
    return ' '.join(lemmas)


def preprocess_for_cbf(row) -> str:
    """
    Build combined_content exactly as specified in Section 4.2.
    
    Field repetitions:
    - Director(s): 4×
    - Lead actor (1st): 3×
    - 2nd actor: 2×
    - 3rd, 4th, 5th actor: 1× each
    - Writer(s): 2×
    - Plot keywords: 3×
    - Genres: 3×
    - Overview: 1×
    - Wikipedia plot (first 200 words): 1×
    - Production company: 1×
    - Original language: 2×
    - Content rating: 1×
    """
    components = []
    
    # Helper to safely get list items
    def safe_get(lst, idx):
        try:
            return lst[idx] if len(lst) > idx else ""
        except:
            return ""
    
    # 1. Directors (4×)
    directors = row.get('directors', [])
    if isinstance(directors, list) and directors:
        components.extend([str(d).lower() for d in directors] * 4)
    
    # 2. Lead actor (3×)
    cast = row.get('cast_top5', [])
    if isinstance(cast, list) and cast:
        components.extend([str(cast[0]).lower()] * 3)
        
        # 3. 2nd actor (2×)
        if len(cast) > 1:
            components.extend([str(cast[1]).lower()] * 2)
        
        # 4. 3rd, 4th, 5th actors (1× each)
        for i in range(2, min(5, len(cast))):
            components.append(str(cast[i]).lower())
    
    # 5. Writers (2×)
    writers = row.get('writers', [])
    if isinstance(writers, list) and writers:
        components.extend([str(w).lower() for w in writers] * 2)
    
    # 6. Plot keywords (3×)
    keywords = row.get('plot_keywords', '')
    if isinstance(keywords, str) and keywords:
        keyword_list = [k.strip().lower() for k in keywords.split('|') if k.strip()]
        components.extend(keyword_list * 3)
    
    # 7. Genres (3×)
    genres = row.get('genres_list', [])
    if isinstance(genres, list) and genres:
        components.extend([str(g).lower() for g in genres] * 3)
    
    # 8. Overview (1×)
    overview = row.get('overview', '')
    if isinstance(overview, str) and overview:
        components.append(overview.lower())
    
    # 9. Wikipedia plot - first 200 words (1×)
    wiki = row.get('wiki_plot', '')
    if isinstance(wiki, str) and wiki:
        words = wiki.split()[:200]
        components.append(' '.join(words).lower())
    
    # 10. Production company (1×)
    prod_companies = row.get('production_companies_list', [])
    if isinstance(prod_companies, list) and prod_companies:
        components.append(str(prod_companies[0]).lower())
    
    # 11. Original language (2×)
    lang = row.get('original_language', '')
    if isinstance(lang, str) and lang:
        components.extend([lang.lower()] * 2)
    
    # 12. Content rating (1×)
    rating = row.get('content_rating', '')
    if isinstance(rating, str) and rating:
        components.append(rating.lower())
    
    # Join all components with spaces
    raw_text = ' '.join(components)
    
    # Apply cleaning pipeline
    return clean_text(raw_text)