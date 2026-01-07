import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- SETUP ----------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 2_000_000

INPUT_DIR = "data/text"
OUTPUT_DIR = "data/nlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- TOKENIZATION ----------
def spacy_tokens(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token) > 2
        and not token.text.lower().startswith("cid")
    ]
    return tokens


def ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


documents = {}
raw_docs = {}

for fname in os.listdir(INPUT_DIR):
    if fname.endswith("_clean.txt"):
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
            raw_docs[fname] = text
            documents[fname] = spacy_tokens(text)

# ---------- WORD FREQUENCY ----------
for fname, tokens in documents.items():
    company = fname.replace("_clean.txt", "").lower()

    freq = Counter(tokens)
    freq_df = pd.DataFrame(freq.most_common(30), columns=["word", "count"])
    freq_df.to_csv(f"{OUTPUT_DIR}/{company}_word_freq.csv", index=False)

    plt.figure()
    freq_df.set_index("word")["count"].plot(kind="bar")
    plt.title(f"{company.upper()} – Word Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{company}_word_freq.png")
    plt.close()

# ---------- BIGRAM / TRIGRAM ----------
for fname, tokens in documents.items():
    company = fname.replace("_clean.txt", "").lower()

    bigrams = Counter(" ".join(bg) for bg in ngrams(tokens, 2))
    trigrams = Counter(" ".join(tg) for tg in ngrams(tokens, 3))

    ng_df = pd.DataFrame(
        bigrams.most_common(20) + trigrams.most_common(20),
        columns=["ngram", "count"]
    )

    ng_df.to_csv(f"{OUTPUT_DIR}/{company}_ngrams.csv", index=False)

# ---------- TF-IDF ----------
processed_docs = [" ".join(tokens) for tokens in documents.values()]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0
)

tfidf = vectorizer.fit_transform(processed_docs)
terms = vectorizer.get_feature_names_out()

for idx, fname in enumerate(documents.keys()):
    company = fname.replace("_clean.txt", "").lower()
    scores = tfidf[idx].toarray().flatten()

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    with open(f"{OUTPUT_DIR}/{company}_tfidf_keywords.txt", "w", encoding="utf-8") as f:
        for term, score in ranked[:20]:
            f.write(f"{term}: {score:.4f}\n")

print("Task 2 complete.")
