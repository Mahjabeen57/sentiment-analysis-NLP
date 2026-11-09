"""
Sentiment Analysis System Using NLP
===================================
A complete, production-ready sentiment analysis pipeline:

  raw text -> cleaning -> tokenization -> stopword removal -> lemmatization
           -> TF-IDF (1-2 grams) -> Logistic Regression -> sentiment + confidence
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------- #
#  NLTK (optional – the system degrades gracefully if NLTK is unavailable)
# --------------------------------------------------------------------------- #
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:                                    
    NLTK_AVAILABLE = False

_NLTK_READY = False

#: Words that flip the polarity – never remove them as stopwords.
NEGATIONS = {
    "not", "no", "nor", "never", "none", "cannot", "without", "neither",
    "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely",
}

#: Small fallback list used when NLTK's stopword corpus is not available.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as",
    "of", "at", "by", "for", "with", "about", "into", "to", "from", "in",
    "on", "off", "over", "under", "again", "further", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "can", "will", "just", "should", "now", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "ought", "s",
    "t", "don", "didn", "doesn", "isn", "aren", "wasn", "weren", "hasn",
    "haven", "hadn", "won", "wouldn", "couldn", "shouldn", "mustn", "mightn",
}

#: Common English contractions -> expanded form (keeps negation intact).
CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
    "it's": "it is", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "where's": "where is", "who's": "who is",
    "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS)) + r")\b")


def ensure_nltk_data() -> None:
    """Download the NLTK corpora required by the preprocessor (once)."""
    global _NLTK_READY
    if _NLTK_READY or not NLTK_AVAILABLE:
        return
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]
    for package, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:                         
                pass
    _NLTK_READY = True


# --------------------------------------------------------------------------- #
#  1. Text preprocessing
# --------------------------------------------------------------------------- #
class TextPreprocessor:
    """
    Full NLP cleaning pipeline.

    lower-case -> unescape HTML -> strip URLs/tags/mentions -> expand
    contractions -> remove punctuation & digits -> tokenize -> drop stopwords
    (negations preserved) -> lemmatize -> rejoin.
    """

    def __init__(self, remove_stopwords: bool = True,
                 lemmatize: bool = True, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self._stopwords = self._load_stopwords() if remove_stopwords else set()
        self._lemmatizer = None

    @staticmethod
    def _load_stopwords() -> set:
        if NLTK_AVAILABLE:
            try:
                ensure_nltk_data()
                return set(nltk_stopwords.words("english")) - NEGATIONS
            except Exception:
                pass
        return set(_FALLBACK_STOPWORDS) - NEGATIONS

    def _get_lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @staticmethod
    def _expand_contractions(text: str) -> str:
        return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if NLTK_AVAILABLE:
            try:
                ensure_nltk_data()
                return word_tokenize(text)
            except Exception:
                pass
        return re.findall(r"[a-z']+", text)

    def clean(self, text: str) -> str:
        """Normalise raw text (no tokenisation yet)."""
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)                       
        text = re.sub(r"http\S+|www\.\S+", " ", text)              
        text = re.sub(r"@\w+", " ", text)                          
        text = re.sub(r"#(\w+)", r" \1 ", text)                    
        if self.lowercase:
            text = text.lower()
        text = self._expand_contractions(text)
        text = re.sub(r"[^a-z\s']", " ", text)                     
        text = re.sub(r"'", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> List[str]:
        tokens = self._tokenize(self.clean(text))
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self.lemmatize:
            try:
                lem = self._get_lemmatizer()
                tokens = [lem.lemmatize(t, pos="v") for t in tokens]
            except Exception:
                pass
        return [t for t in tokens if len(t) > 1]

    def preprocess(self, text: str) -> str:
        """Return the final cleaned string fed to the vectorizer."""
        return " ".join(self.tokenize(text))


# --------------------------------------------------------------------------- #
#  2. Sentiment analyser
# --------------------------------------------------------------------------- #
class SentimentAnalyzer:
    """TF-IDF + Logistic Regression sentiment classifier."""

    def __init__(self, preprocessor: TextPreprocessor | None = None,
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int | None = 100_000,
                 C: float = 1.0):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None
        self.is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=self.C, max_iter=2000, solver="liblinear", random_state=42
            )),
        ])

    def _vectorize(self, texts: Iterable[str]) -> List[str]:
        return [self.preprocessor.preprocess(t) for t in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SentimentAnalyzer":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if len(texts) == 0:
            raise ValueError("empty training set")

        processed = self._vectorize(texts)
        self.label_encoder = LabelEncoder().fit(labels)
        y = self.label_encoder.transform(labels)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(processed, y)
        self.is_fitted = True
        return self

    def evaluate(self, texts: Sequence[str], labels: Sequence[str],
                 test_size: float = 0.2, random_state: int = 42) -> dict:
        """Hold-out evaluation (train/test split)."""
        processed = self._vectorize(texts)
        stratify = labels if min(np.bincount(
            np.unique(labels, return_inverse=True)[1])) >= 2 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            processed, labels, test_size=test_size,
            random_state=random_state, stratify=stratify,
        )
        model = self._build_pipeline().fit(X_tr, y_tr)
        preds = model.predict(X_te)

        class_names = sorted(set(labels))
        return {
            "accuracy": accuracy_score(y_te, preds),
            "report": classification_report(y_te, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_te, preds, labels=class_names),
            "classes": class_names,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }

    def cross_validate(self, texts: Sequence[str], labels: Sequence[str],
                       cv: int = 5) -> dict:
        """K-fold cross-validation on the full dataset."""
        processed = self._vectorize(texts)
        scores = cross_val_score(self._build_pipeline(), processed, labels,
                                 cv=cv, scoring="accuracy")
        return {"scores": scores, "mean": float(scores.mean()),
                "std": float(scores.std())}

    def predict(self, texts: str | Sequence[str]) -> List[dict]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        processed = self._vectorize(batch)

        try:
            proba = self.pipeline.predict_proba(processed)
            idx = proba.argmax(axis=1)
            confidence = proba[np.arange(len(idx)), idx]
        except AttributeError:                          
            idx = self.pipeline.predict(processed)
            proba = None
            confidence = [None] * len(idx)

        labels = self.label_encoder.inverse_transform(idx)
        results = []
        for i, (raw, clean) in enumerate(zip(batch, processed)):
            class_scores = (
                {str(c): round(float(p), 4)
                 for c, p in zip(self.label_encoder.classes_, proba[i])}
                if proba is not None else {}
            )
            results.append({
                "text": raw,
                "cleaned": clean,
                "sentiment": str(labels[i]),
                "confidence": (round(float(confidence[i]), 4)
                               if confidence[i] is not None else None),
                "scores": class_scores,
            })
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict(text)[0]

    def explain(self, text: str, top_k: int = 5) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        vec = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        if clf.coef_.shape[0] != 1:                     
            return {"positive": [], "negative": []}

        clean = self.preprocessor.preprocess(text)
        x = vec.transform([clean])
        features = vec.get_feature_names_out()
        contributions = x.toarray()[0] * clf.coef_[0]
        nonzero = np.where(contributions != 0)[0]

        if len(nonzero) == 0:
            return {"positive": [], "negative": []}

        order = nonzero[np.argsort(contributions[nonzero])]
        negative = [(features[i], round(float(contributions[i]), 4))
                    for i in order[:top_k][::-1]]
        positive = [(features[i], round(float(contributions[i]), 4))
                    for i in order[-top_k:][::-1]]
        return {"positive": positive, "negative": negative}

    def save(self, path: str = "sentiment_model.joblib") -> None:
        if not self.is_fitted:
            raise RuntimeError("Nothing to save – fit the model first.")
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "preprocessor": self.preprocessor,
            "config": {"ngram_range": self.ngram_range,
                       "max_features": self.max_features, "C": self.C},
        }, path)

    @classmethod
    def load(cls, path: str = "sentiment_model.joblib") -> "SentimentAnalyzer":
        bundle = joblib.load(path)
        obj = cls(preprocessor=bundle["preprocessor"],
                  **bundle.get("config", {}))
        obj.pipeline = bundle["pipeline"]
        obj.label_encoder = bundle["label_encoder"]
        obj.is_fitted = True
        return obj


# --------------------------------------------------------------------------- #
#  3. Optional: HuggingFace transformer backend
# --------------------------------------------------------------------------- #
class TransformerSentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline("sentiment-analysis", model=model_name)

    def predict(self, texts: str | Sequence[str]) -> List[dict]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        raw = self.pipe(batch, truncation=True)
        return [{"text": t, "sentiment": r["label"].lower(),
                 "confidence": round(float(r["score"]), 4)}
                for t, r in zip(batch, raw)]

    def predict_one(self, text: str) -> dict:
        return self.predict(text)[0]


# --------------------------------------------------------------------------- #
#  4. Built-in demo dataset
# --------------------------------------------------------------------------- #
DEMO_DATA: List[Tuple[str, str]] = [
    ("I absolutely loved this movie, the acting was brilliant.", "positive"),
    ("What a fantastic experience, highly recommended!", "positive"),
    ("The food was delicious and the service was excellent.", "positive"),
    ("This product works perfectly and exceeded my expectations.", "positive"),
    ("Great quality, fast delivery, very happy with the purchase.", "positive"),
    ("The story was touching and the visuals were stunning.", "positive"),
    ("Best purchase I have made this year.", "positive"),
    ("The staff were friendly and helpful throughout.", "positive"),
    ("Amazing performance by the entire cast.", "positive"),
    ("I am really impressed with how well it works.", "positive"),
    ("Superb design and outstanding build quality.", "positive"),
    ("The hotel room was clean, spacious and comfortable.", "positive"),
    ("Highly satisfied with the customer support team.", "positive"),
    ("A wonderful book that I could not put down.", "positive"),
    ("The app is intuitive, fast and reliable.", "positive"),
    ("Excellent value for money, I will buy again.", "positive"),
    ("The concert was incredible, I had a great time.", "positive"),
    ("Very good quality and it arrived earlier than expected.", "positive"),
    ("This is by far the best restaurant in town.", "positive"),
    ("The new update made the software much better.", "positive"),
    ("I love how easy it is to use.", "positive"),
    ("The support team solved my problem quickly and politely.", "positive"),
    ("Beautifully written and deeply moving.", "positive"),
    ("Fantastic soundtrack and great cinematography.", "positive"),
    ("The laptop is fast, light and has great battery life.", "positive"),
    ("Totally worth the price, I am delighted.", "positive"),
    ("The course was informative and well structured.", "positive"),
    ("Friendly atmosphere and reasonable prices.", "positive"),
    ("The delivery was quick and the packaging was perfect.", "positive"),
    ("A masterpiece, I would watch it again.", "positive"),
    ("I hated this movie, the plot made no sense.", "negative"),
    ("Terrible experience, I will never come back.", "negative"),
    ("The food was cold and the waiter was rude.", "negative"),
    ("This product broke after two days, waste of money.", "negative"),
    ("Very disappointing quality, do not buy this.", "negative"),
    ("The story was boring and far too long.", "negative"),
    ("Worst customer service I have ever dealt with.", "negative"),
    ("The room was dirty and the noise was unbearable.", "negative"),
    ("I regret buying this, it never works properly.", "negative"),
    ("The app keeps crashing and loses my data.", "negative"),
    ("Overpriced and poorly made.", "negative"),
    ("The staff were unhelpful and unfriendly.", "negative"),
    ("The delivery was late and the box was damaged.", "negative"),
    ("Awful performance, the actors looked bored.", "negative"),
    ("I am very unhappy with this purchase.", "negative"),
    ("The instructions were confusing and incomplete.", "negative"),
    ("The battery drains in less than an hour.", "negative"),
    ("Poor build quality, it feels cheap and flimsy.", "negative"),
    ("The movie was a complete waste of time.", "negative"),
    ("The service was slow and the food was tasteless.", "negative"),
    ("Not worth the money at all.", "negative"),
    ("I would not recommend this to anyone.", "negative"),
    ("The update made everything slower and buggy.", "negative"),
    ("Horrible sound quality, very disappointing.", "negative"),
    ("The book was dull and repetitive.", "negative"),
    ("They charged me twice and refused to refund.", "negative"),
    ("The worst restaurant in the city, avoid it.", "negative"),
    ("Everything about this product feels wrong.", "negative"),
    ("The screen is dim and the keyboard is terrible.", "negative"),
    ("A tedious film with no real story.", "negative"),
]


def load_dataset(path: str, text_col: str, label_col: str) -> Tuple[List[str], List[str]]:
    import pandas as pd
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Columns {text_col!r}/{label_col!r} not found. "
            f"Available: {list(df.columns)}"
        )
    df = df[[text_col, label_col]].dropna()
    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()


# --------------------------------------------------------------------------- #
#  5. CLI
# --------------------------------------------------------------------------- #
def _print_result(result: dict) -> None:
    print(f"\n  Text       : {result['text']}")
    print(f"  Cleaned    : {result['cleaned']}")
    print(f"  Sentiment  : {result['sentiment'].upper()}"
          + (f"  (confidence {result['confidence']:.2%})"
             if result.get("confidence") is not None else ""))
    if result.get("scores"):
        for label, score in sorted(result["scores"].items(),
                                   key=lambda kv: -kv[1]):
            bar = "█" * int(round(score * 30))
            print(f"    {label:<10} {score:>6.2%}  {bar}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NLP Sentiment Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="use the built-in demo dataset")
    src.add_argument("--csv", help="path to a CSV/TSV dataset")
    src.add_argument("--model", help="path to a saved .joblib model")
    parser.add_argument("--text-col", default="text", help="text column name (default: text)")
    parser.add_argument("--label-col", default="label", help="label column name (default: label)")
    parser.add_argument("--predict", nargs="+", help="predict sentiment for these texts")
    parser.add_argument("--interactive", action="store_true", help="start an interactive prompt")
    parser.add_argument("--save", metavar="PATH", help="save the trained model to PATH")
    parser.add_argument("--cv", type=int, default=0, help="run k-fold cross-validation")
    args = parser.parse_args(argv)

    if args.model:
        if not os.path.exists(args.model):
            parser.error(f"model file not found: {args.model}")
        print(f"[*] Loading model from {args.model} ...")
        analyzer = SentimentAnalyzer.load(args.model)
    else:
        if args.demo:
            texts = [t for t, _ in DEMO_DATA]
            labels = [l for _, l in DEMO_DATA]
            print(f"[*] Using built-in demo dataset ({len(texts)} samples)")
        else:
            texts, labels = load_dataset(args.csv, args.text_col, args.label_col)
            print(f"[*] Loaded {len(texts)} samples from {args.csv}")

        print("\n=== Hold-out evaluation (80/20 split) ===")
        metrics = analyzer_eval = SentimentAnalyzer().evaluate(texts, labels)
        print(f"Accuracy : {metrics['accuracy']:.2%}")
        print(f"Train/Test: {metrics['n_train']}/{metrics['n_test']}")
        print("\nClassification report:")
        print(metrics["report"])
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"  labels = {metrics['classes']}")
        print(metrics["confusion_matrix"])

        if args.cv > 1:
            print(f"\n=== {args.cv}-fold cross-validation ===")
            cv_res = SentimentAnalyzer().cross_validate(texts, labels, cv=args.cv)
            print("Scores :", " ".join(f"{s:.3f}" for s in cv_res["scores"]))
            print(f"Mean   : {cv_res['mean']:.2%}  (±{cv_res['std']:.3f})")

        print("\n[*] Training final model on the full dataset ...")
        analyzer = SentimentAnalyzer().fit(texts, labels)

        if args.save:
            analyzer.save(args.save)
            print(f"[*] Model saved to {args.save}")

    if args.predict:
        print("\n=== Predictions ===")
        for res in analyzer.predict(args.predict):
            _print_result(res)
            expl = analyzer.explain(res["text"])
            if expl["positive"] or expl["negative"]:
                pos = ", ".join(f"{w}(+{s})" for w, s in expl["positive"]) or "-"
                neg = ", ".join(f"{w}({s})" for w, s in expl["negative"]) or "-"
                print(f"  Pushes POSITIVE : {pos}")
                print(f"  Pushes NEGATIVE : {neg}")

    if args.interactive:
        print("\n=== Interactive mode === (type 'quit' to exit)")
        while True:
            try:
                line = input("\nEnter text > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in {"quit", "exit", "q"}:
                break
            if not line:
                continue
            _print_result(analyzer.predict_one(line))

    if not (args.predict or args.interactive):
        print("\nTip: add --predict \"your text\" or --interactive to try the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())