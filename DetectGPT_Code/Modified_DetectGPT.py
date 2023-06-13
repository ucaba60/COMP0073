# Attempt at adopting DetectGPTs data pre-processing
# Includes extracting Q&A from PubMed, and Prompt&Text from WPs
# After that creates a dictionary of the original data
# And extracts the Qs and Prompts, uses them to prompt GPT-3.5 to create text with max length
# Save results, GPT-Text with different temperatures


# First, data pre-processing, simple start with PREMED
import random
import datasets
import spacy
from collections import Counter
from sklearn import metrics
import spacy
from collections import Counter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


nlp = spacy.load('en_core_web_sm')

model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    return torch.exp(loss).item()

def load_pubmed():
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train')

    # combine question and long_answer and add label, and create a spaCy Doc for each text
    data = [(nlp(f'Question: {q} Answer: {a}'), 0) for q, a in zip(data['question'], data['long_answer'])]

    return data

data = load_pubmed()

# Initialize counters
pos_counts = Counter()
function_word_counts = Counter()
punctuation_counts = Counter()
uppercase_counts = Counter()

# Initialize variables for sentence length calculation
total_tokens = 0
total_sentences = 0

# Initialize list to store each text's perplexity
perplexities = []

# Iterate over all tokens in the data only once
for doc, _ in data:
    text = doc.text
    perplexities.append(calculate_perplexity(text))
    for token in doc:
        pos_counts[token.pos_] += 1
        if token.text in {'a', 'in', 'of'}:
            function_word_counts[token.text] += 1
        if token.is_punct:
            punctuation_counts[token.text] += 1
        if token.text.isupper():
            uppercase_counts[token.text] += 1
    total_tokens += len(doc)
    total_sentences += len(list(doc.sents))

# Calculate average sentence length
avg_sentence_length = total_tokens / total_sentences if total_sentences > 0 else 0

# Calculate average perplexity
avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0


fig, ax = plt.subplots()

# Generate a histogram with seaborn
sns.histplot(perplexities, bins=50, ax=ax)

# Set labels and title
ax.set_xlabel('Perplexity')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Perplexities')

# Show the plot
plt.show()