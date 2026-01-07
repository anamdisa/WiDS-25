import os
import re
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 2_000_000

INPUT_DIR = "data/text"
OUTPUT_DIR = "data/targets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PATTERN = re.compile(
    r"""
    (reduce|cut|decrease|lower|improve|increase|achieve)
    .*?
    (co2|carbon|emission|emissions|energy|efficiency)
    .*?
    (\d{1,3}\s?%)
    .*?
    (20\d{2})
    """,
    re.IGNORECASE | re.VERBOSE
)

rows = []

for fname in os.listdir(INPUT_DIR):
    if fname.endswith("_clean.txt"):
        company = fname.replace("_clean.txt", "")
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()

        doc = nlp(text)

        for sent in doc.sents:
            match = TARGET_PATTERN.search(sent.text)
            if match:
                rows.append({
                    "company": company,
                    "action": match.group(1).lower(),
                    "metric": match.group(2).lower(),
                    "target_value": match.group(3),
                    "target_year": match.group(4),
                    "sentence": sent.text.strip()
                })

df = pd.DataFrame(rows)
df.to_csv(f"{OUTPUT_DIR}/sustainability_targets.csv", index=False)

print("Task 3 complete.")
