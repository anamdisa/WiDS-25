"""
Complete NLP Pipeline for Sustainability Report Analysis
Combines PDF extraction, text processing, keyword analysis, and commitment extraction
"""

import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import pdfplumber
from pathlib import Path

# ---------- CONFIGURATION ----------
PDF_DIR = Path("data/reports")
OUTPUT_DIR = Path("data/nlp_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe('sentencizer')  
nlp.max_length = 2_000_000

# Emission-related keywords
EMISSION_KEYWORDS = [
    'co2', 'carbon', 'emission', 'emissions', 'greenhouse', 'ghg',
    'net-zero', 'net zero', 'carbon neutral', 'decarbonization',
    'scope 1', 'scope 2', 'scope 3', 'scope-1', 'scope-2', 'scope-3',
    'climate', 'sustainability', 'reduction', 'target', 'goal'
]

# ---------- STEP 1: PDF TO TEXT ----------
def clean_text(text):
    """Clean extracted PDF text"""
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'page \d+|\d+/\d+', '', text)  # Remove page numbers
    text = re.sub(r'\s+', ' ', text)  # Consolidate whitespace
    text = re.sub(r'\bcid\s*\d+\b', '', text)  # Remove CID artifacts
    return text.strip()

def pdf_to_text(pdf_path):
    """Extract text from PDF using pdfplumber"""
    full_text = []
    print(f"Processing: {pdf_path.name}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    
    cleaned = clean_text(" ".join(full_text))
    print(f"  Extracted {len(cleaned)} characters")
    return cleaned

# ---------- STEP 2: TEXT PROCESSING ----------
def spacy_tokens(text):
    """Tokenize and clean text using spaCy"""
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

def get_emission_sentences(text):
    """Extract sentences related to emissions using keywords"""
    doc = nlp(text)
    emission_sentences = []
    
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        # Check if sentence contains emission-related keywords
        if any(keyword in sent_lower for keyword in EMISSION_KEYWORDS):
            emission_sentences.append(sent.text.strip())
    
    return emission_sentences

# ---------- STEP 3: COMMITMENT EXTRACTION ----------
def extract_commitments(text, company_name):
    """Extract structured commitment statements from text"""
    doc = nlp(text)
    
    # Enhanced regex patterns for different commitment formats
    patterns = [
        # Pattern 1: "reduce X by Y% by YEAR"
        re.compile(
            r'(reduce|cut|decrease|lower|achieve)\s+.*?(co2|carbon|emission|emissions|scope\s*[1-3]|scope-[1-3])\s+.*?(\d{1,3}\s?%)\s+.*?(by\s+)?(\d{4})',
            re.IGNORECASE
        ),
        # Pattern 2: "net-zero by YEAR"
        re.compile(
            r'(net-zero|net zero|carbon neutral|carbon neutrality)\s+.*?(by\s+)?(\d{4})',
            re.IGNORECASE
        ),
        # Pattern 3: "Y% reduction in X by YEAR"
        re.compile(
            r'(\d{1,3}\s?%)\s+(reduction|decrease|cut)\s+.*?(co2|carbon|emission|emissions|scope\s*[1-3]|scope-[1-3])\s+.*?(by\s+)?(\d{4})',
            re.IGNORECASE
        )
    ]
    
    commitments = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        for pattern in patterns:
            match = pattern.search(sent_text)
            if match:
                groups = match.groups()
                
                # Extract year (last group that looks like a year)
                year = None
                for g in reversed(groups):
                    if g and re.match(r'\d{4}', str(g)):
                        year = g
                        break
                
                # Extract percentage
                percentage = None
                for g in groups:
                    if g and '%' in str(g):
                        percentage = g.strip()
                        break
                
                # Extract metric/scope
                metric = None
                for g in groups:
                    if g and any(keyword in str(g).lower() for keyword in ['co2', 'carbon', 'emission', 'scope']):
                        metric = g.strip()
                        break
                
                # For net-zero commitments
                if any(term in sent_text.lower() for term in ['net-zero', 'net zero', 'carbon neutral']):
                    percentage = "100%"
                    metric = "net-zero"
                
                if year:
                    commitments.append({
                        'Company': company_name,
                        'Commitment': sent_text[:200] + ('...' if len(sent_text) > 200 else ''),
                        'Target Year': year,
                        'Reduction %': percentage if percentage else 'Not specified',
                        'Metric': metric if metric else 'General emissions',
                        'Full Sentence': sent_text
                    })
                break  # Only match once per sentence
    
    return commitments

# ---------- STEP 4: VISUALIZATIONS ----------
def generate_word_cloud(tokens, company_name, output_dir):
    """Generate word cloud from tokens"""
    text_for_cloud = ' '.join(tokens)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text_for_cloud)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{company_name} - Keywords Word Cloud', fontsize=16, pad=20)
    plt.tight_layout()
    
    output_path = output_dir / f'{company_name}_wordcloud.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved word cloud: {output_path}")

def generate_keyword_frequency_plot(tokens, company_name, output_dir, top_n=30):
    """Generate keyword frequency bar plot"""
    freq = Counter(tokens)
    freq_df = pd.DataFrame(freq.most_common(top_n), columns=['Keyword', 'Frequency'])
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(freq_df)), freq_df['Frequency'], color='steelblue')
    plt.yticks(range(len(freq_df)), freq_df['Keyword'])
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Keywords', fontsize=12)
    plt.title(f'{company_name} - Top {top_n} Keywords', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = output_dir / f'{company_name}_keyword_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved keyword frequency plot: {output_path}")

def generate_emission_keyword_cloud(emission_sentences, company_name, output_dir):
    """Generate word cloud specifically from emission-related sentences"""
    emission_text = ' '.join(emission_sentences)
    
    # Tokenize emission sentences
    doc = nlp(emission_text)
    emission_tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]
    
    if emission_tokens:
        text_for_cloud = ' '.join(emission_tokens)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='RdYlGn_r',  # Red-Yellow-Green reversed
            max_words=80,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text_for_cloud)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{company_name} - Emission-Specific Keywords', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = output_dir / f'{company_name}_emission_wordcloud.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved emission word cloud: {output_path}")

# ---------- MAIN PIPELINE ----------
def process_report(pdf_path):
    """Complete pipeline for a single PDF report"""
    company_name = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {company_name}")
    print(f"{'='*60}")
    
    # Step 1: Extract and clean text
    print("\n[1] Extracting text from PDF...")
    raw_text = pdf_to_text(pdf_path)
    
    # Step 2: Tokenize
    print("[2] Tokenizing and cleaning text...")
    tokens = spacy_tokens(raw_text)
    print(f"  Generated {len(tokens)} tokens")
    
    # Step 3: Extract emission-related sentences
    print("[3] Extracting emission-related sentences...")
    emission_sentences = get_emission_sentences(raw_text)
    print(f"  Found {len(emission_sentences)} emission-related sentences")
    
    # Step 4: Extract commitments
    print("[4] Extracting commitments...")
    commitments = extract_commitments(raw_text, company_name)
    print(f"  Found {len(commitments)} commitments")
    
    # Step 5: Generate visualizations
    print("[5] Generating visualizations...")
    generate_word_cloud(tokens, company_name, OUTPUT_DIR)
    generate_keyword_frequency_plot(tokens, company_name, OUTPUT_DIR)
    generate_emission_keyword_cloud(emission_sentences, company_name, OUTPUT_DIR)
    
    # Step 6: Save outputs
    print("[6] Saving results...")
    
    # Save commitments table
    if commitments:
        commitments_df = pd.DataFrame(commitments)
        csv_path = OUTPUT_DIR / f'{company_name}_commitments.csv'
        commitments_df.to_csv(csv_path, index=False)
        print(f"  Saved commitments: {csv_path}")
    else:
        print("  No commitments found!")
    
    # Save emission sentences
    emission_path = OUTPUT_DIR / f'{company_name}_emission_sentences.txt'
    with open(emission_path, 'w', encoding='utf-8') as f:
        for sent in emission_sentences[:50]:  # Save top 50
            f.write(sent + '\n\n')
    print(f"  Saved emission sentences: {emission_path}")
    
    # Save cleaned full text
    text_path = OUTPUT_DIR / f'{company_name}_cleaned_text.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    print(f"  Saved cleaned text: {text_path}")
    
    return {
        'company': company_name,
        'commitments': commitments,
        'emission_sentences': len(emission_sentences),
        'total_tokens': len(tokens)
    }

# ---------- RUN PIPELINE ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("NLP SUSTAINABILITY REPORT ANALYSIS PIPELINE")
    print("="*60)
    
    # Ensure PDF directory exists
    if not PDF_DIR.exists():
        print(f"\nCreating PDF directory: {PDF_DIR}")
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Please place your PDF reports in: {PDF_DIR.absolute()}")
        exit(0)
    
    # Process all PDFs
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDF files found in: {PDF_DIR.absolute()}")
        print("Please add PDF reports to this directory and run again.")
        exit(0)
    
    print(f"\nFound {len(pdf_files)} PDF file(s)")
    
    all_commitments = []
    results_summary = []
    
    for pdf_file in pdf_files:
        result = process_report(pdf_file)
        results_summary.append(result)
        all_commitments.extend(result['commitments'])
    
    # Save combined commitments table
    if all_commitments:
        print(f"\n{'='*60}")
        print("SAVING COMBINED RESULTS")
        print(f"{'='*60}")
        
        combined_df = pd.DataFrame(all_commitments)
        combined_path = OUTPUT_DIR / 'all_commitments_combined.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined commitments table: {combined_path}")
        
        # Print summary table
        print("\nSummary of Extracted Commitments:")
        print(combined_df[['Company', 'Commitment', 'Target Year', 'Reduction %']].to_string(index=False))
    
    # Print processing summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal PDFs processed: {len(pdf_files)}")
    print(f"Total commitments extracted: {len(all_commitments)}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  - *_commitments.csv (commitment tables)")
    print("  - *_wordcloud.png (general word clouds)")
    print("  - *_emission_wordcloud.png (emission-specific word clouds)")
    print("  - *_keyword_frequency.png (frequency plots)")
    print("  - *_emission_sentences.txt (extracted sentences)")
    print("  - *_cleaned_text.txt (cleaned full text)")
    print("  - all_commitments_combined.csv (combined results)")