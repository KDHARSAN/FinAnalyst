from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import math
import collections
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="FinAnalyst NLP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────

class DocumentRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    document: str
    question: str
    history: Optional[list] = []

# ── NLP Helpers ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","that",
    "this","these","those","it","its","we","our","they","their","he","she",
    "his","her","which","who","what","when","where","how","as","not","no",
    "also","than","then","so","if","about","into","through","during","over",
    "after","before","between","each","more","other","such","up","out","all"
}

RISK_KEYWORDS = {
    "high": [
        "lawsuit","litigation","investigation","breach","fraud","bankruptcy",
        "default","penalty","fine","loss","decline","risk","threat","failure",
        "violation","recession","downturn","volatile","uncertainty","impairment",
        "write-off","writedown","deficit","shortfall","concern","warning","danger",
        "critical","severe","significant risk","material adverse","going concern"
    ],
    "medium": [
        "challenge","headwind","pressure","competition","delay","slower","weakness",
        "uncertain","exposure","dependent","regulatory","compliance","tariff",
        "inflation","interest rate","currency","foreign exchange","cybersecurity",
        "concentration","disruption","shortage","constraint"
    ],
    "low": [
        "monitor","review","evaluate","assess","consider","potential","possible",
        "may","could","might","fluctuation","variable","seasonal","cyclical"
    ]
}

FINANCIAL_ENTITIES = {
    "money": r'\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|trillion|M|B|T|K))?\b|[\d,]+(?:\.\d+)?\s?(?:million|billion|trillion)\s?(?:dollars?|USD|INR|EUR|GBP|rupees?)?',
    "percentage": r'\d+(?:\.\d+)?\s?(?:%|percent(?:age)?)',
    "date": r'\b(?:Q[1-4]\s?\d{4}|FY\s?\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4})\b',
}

COMPANY_SUFFIXES = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC|Group|Holdings?|Technologies?|Solutions?|Services?|Ventures?|Capital|Partners?|Bank|Financial|Industries|Enterprises?)\b'
PERSON_PATTERN = r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|CEO|CFO|COO|CTO|President|Chairman|Director|VP|said)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b|([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:said|noted|added|stated|commented|announced|confirmed)'

FINANCIAL_ABBREVS = {"eps":"Earnings Per Share","roi":"Return on Investment","ebitda":"EBITDA","capex":"Capital Expenditure","yoy":"Year over Year","qoq":"Quarter over Quarter","fy":"Fiscal Year","q1":"Q1","q2":"Q2","q3":"Q3","q4":"Q4","ipo":"IPO","m&a":"Mergers & Acquisitions"}

BULLISH_WORDS = {"record","growth","increase","profit","beat","exceed","strong","robust","accelerat","outperform","revenue","gain","surge","rise","higher","positive","optimistic","expand","improve","deliver","achieve","momentum","leadership","innovate","leading"}
BEARISH_WORDS = {"loss","decline","fall","miss","below","weak","slow","concern","risk","challenge","pressure","headwind","uncertain","difficult","adverse","downturn","shrink","reduce","cut","restructur","impair","write","deficit","shortfall","warning"}

def tokenize(text: str) -> list:
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]

def extract_keywords(text: str, top_n: int = 10) -> list:
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=200, ngram_range=(1,2))
        tfidf = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return [{"word": w, "score": round(float(s), 4)} for w, s in ranked[:top_n] if s > 0]
    except:
        tokens = tokenize(text)
        freq = collections.Counter(t for t in tokens if t not in STOPWORDS and len(t) > 3)
        return [{"word": w, "score": round(c/len(tokens), 4)} for w, c in freq.most_common(top_n)]

def extract_entities(text: str) -> dict:
    money = list(set(re.findall(FINANCIAL_ENTITIES["money"], text, re.IGNORECASE)))[:8]
    percentages = list(set(re.findall(FINANCIAL_ENTITIES["percentage"], text, re.IGNORECASE)))[:6]
    dates = list(set(re.findall(FINANCIAL_ENTITIES["date"], text)))[:8]
    companies = list(set(re.findall(COMPANY_SUFFIXES, text)))[:6]
    
    people = []
    for m in re.finditer(PERSON_PATTERN, text):
        name = m.group(1) or m.group(2)
        if name and name not in people:
            people.append(name)
    
    return {
        "companies": companies,
        "people": people[:5],
        "money": [m.strip() for m in money if m.strip()],
        "percentages": [p.strip() for p in percentages if p.strip()],
        "dates": [d.strip() for d in dates if d.strip()]
    }

def detect_risk_signals(text: str) -> list:
    signals = []
    text_lower = text.lower()
    for level, words in RISK_KEYWORDS.items():
        for word in words:
            if word in text_lower:
                idx = text_lower.find(word)
                start = max(0, idx - 40)
                end = min(len(text), idx + len(word) + 40)
                context = text[start:end].strip()
                signals.append({"phrase": word, "context": context, "level": level})
    seen = set()
    unique = []
    for s in signals:
        if s["phrase"] not in seen:
            seen.add(s["phrase"])
            unique.append(s)
    return unique[:12]

def sentiment_analysis(text: str) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    tokens = tokenize(text)
    bull_count = sum(1 for t in tokens if any(b in t for b in BULLISH_WORDS))
    bear_count = sum(1 for t in tokens if any(b in t for b in BEARISH_WORDS))

    if polarity > 0.3:
        label = "Bullish"
    elif polarity > 0.1:
        label = "Cautiously Optimistic"
    elif polarity > -0.1:
        label = "Neutral"
    elif polarity > -0.3:
        label = "Cautious"
    else:
        label = "Bearish"

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label": label,
        "bullish_signals": bull_count,
        "bearish_signals": bear_count
    }

def sentence_sentiment(text: str) -> list:
    sents = sentences(text)[:8]
    result = []
    for s in sents:
        blob = TextBlob(s)
        p = blob.sentiment.polarity
        label = "positive" if p > 0.1 else "negative" if p < -0.1 else "neutral"
        result.append({
            "sentence": s[:120] + ("..." if len(s) > 120 else ""),
            "sentiment": label,
            "score": round(p, 3)
        })
    return result

def readability_score(text: str) -> dict:
    words = text.split()
    sents_list = sentences(text)
    avg_words = len(words) / max(len(sents_list), 1)
    long_words = sum(1 for w in words if len(w) > 6)
    long_pct = round(long_words / max(len(words), 1) * 100, 1)

    if avg_words < 15 and long_pct < 30:
        level = "Easy"
    elif avg_words < 25 and long_pct < 45:
        level = "Moderate"
    else:
        level = "Complex"

    return {
        "word_count": len(words),
        "sentence_count": len(sents_list),
        "avg_words_per_sentence": round(avg_words, 1),
        "complex_word_pct": long_pct,
        "readability": level
    }

def generate_summary(text: str, num_sentences: int = 3) -> str:
    sents = sentences(text)
    if len(sents) <= num_sentences:
        return text.strip()
    tokens = tokenize(text)
    freq = collections.Counter(t for t in tokens if t not in STOPWORDS and len(t) > 3)
    total = sum(freq.values()) or 1

    scored = []
    for s in sents:
        s_tokens = tokenize(s)
        score = sum(freq.get(t, 0) / total for t in s_tokens)
        scored.append((score, s))

    scored.sort(reverse=True)
    top = scored[:num_sentences]
    ordered = sorted(top, key=lambda x: sents.index(x[1]))
    return " ".join(s for _, s in ordered)

def classify_document(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["earnings","quarterly results","revenue","eps","per share"]):
        return "Earnings Report"
    if any(w in text_lower for w in ["risk factor","material adverse","forward-looking","cautionary"]):
        return "Risk Disclosure"
    if any(w in text_lower for w in ["guidance","outlook","forecast","expect","anticipate","project"]):
        return "Forward Guidance"
    if any(w in text_lower for w in ["press release","announces","today announced"]):
        return "Press Release"
    if any(w in text_lower for w in ["balance sheet","cash flow","income statement","10-k","annual report"]):
        return "Annual Report / 10-K"
    return "Financial Document"

def qa_extractive(document: str, question: str) -> str:
    sents = sentences(document)
    if not sents:
        return "No content found in document."

    q_tokens = set(tokenize(question)) - STOPWORDS
    
    scored = []
    for s in sents:
        s_tokens = set(tokenize(s))
        overlap = len(q_tokens & s_tokens)
        length_bonus = min(len(s_tokens) / 30, 1.0)
        scored.append((overlap + length_bonus * 0.3, s))
    
    scored.sort(reverse=True)
    top = [s for _, s in scored[:3] if _ > 0]
    
    if not top:
        return "The document does not contain a clear answer to this question."
    
    answer = " ".join(top[:2])
    
    q_lower = question.lower()
    entities = extract_entities(document)
    
    if any(w in q_lower for w in ["revenue","sales","income","profit","earnings"]):
        nums = entities.get("money", []) + entities.get("percentages", [])
        if nums:
            answer = f"Key figures mentioned: {', '.join(nums[:4])}. Context: {top[0]}"
    
    elif any(w in q_lower for w in ["who","ceo","cfo","person","executive","management"]):
        people = entities.get("people", [])
        if people:
            answer = f"People mentioned: {', '.join(people)}. {top[0]}"
    
    elif any(w in q_lower for w in ["when","date","quarter","year","period"]):
        dates = entities.get("dates", [])
        if dates:
            answer = f"Time references: {', '.join(dates[:4])}. {top[0]}"
    
    elif any(w in q_lower for w in ["risk","threat","challenge","concern"]):
        risks = detect_risk_signals(document)
        if risks:
            high = [r["phrase"] for r in risks if r["level"] == "high"][:3]
            answer = f"Key risks identified: {', '.join(high) if high else 'moderate risks present'}. {top[0]}"
    
    return answer[:500]

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FinAnalyst API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze")
def analyze_document(req: DocumentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Document text is required")
    if len(text) < 20:
        raise HTTPException(400, "Document too short")

    sentiment = sentiment_analysis(text)
    entities = extract_entities(text)
    keywords = extract_keywords(text, 12)
    risks = detect_risk_signals(text)
    readability = readability_score(text)
    summary = generate_summary(text, 3)
    sent_sentences = sentence_sentiment(text)
    doc_type = classify_document(text)

    return {
        "document_type": doc_type,
        "summary": summary,
        "sentiment": sentiment,
        "readability": readability,
        "keywords": keywords,
        "entities": entities,
        "risk_signals": risks,
        "sentence_sentiments": sent_sentences
    }

@app.post("/api/qa")
def question_answer(req: QARequest):
    doc = req.document.strip()
    question = req.question.strip()
    if not doc or not question:
        raise HTTPException(400, "Document and question are required")

    answer = qa_extractive(doc, question)
    sentiment = sentiment_analysis(answer)
    
    return {
        "question": question,
        "answer": answer,
        "confidence": "high" if len(answer) > 100 else "medium",
        "answer_sentiment": sentiment["label"]
    }

@app.post("/api/compare")
def compare_documents(req: dict):
    texts = req.get("texts", [])
    if len(texts) < 2:
        raise HTTPException(400, "Provide at least 2 documents to compare")
    
    results = []
    for i, text in enumerate(texts[:3]):
        s = sentiment_analysis(text)
        r = readability_score(text)
        results.append({
            "doc_index": i + 1,
            "type": classify_document(text),
            "sentiment": s,
            "word_count": r["word_count"],
            "readability": r["readability"],
            "top_keywords": extract_keywords(text, 5)
        })
    return {"comparison": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
