import datasets
import spacy
from collections import Counter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
DATASET = 'pubmed_qa'
SPLIT = 'pqa_labeled'
TRAIN_SPLIT = 'train'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
CHUNK_SIZE = 512

nlp = spacy.load('en_core_web_sm')
model = BertForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs["input_ids"]
    input_chunks = split_input_into_chunks(input_ids)
    return compute_perplexity(input_chunks)


def split_input_into_chunks(input_ids):
    """
    Splits the input into chunks of size `CHUNK_SIZE-2` to allow for adding `[CLS]` and `[SEP]` tokens.
    """
    chunk_sizes = [(CHUNK_SIZE-2)] * ((input_ids.size(1)) // (CHUNK_SIZE-2))
    remainder = input_ids.size(1) % (CHUNK_SIZE-2)
    if remainder != 0:
        chunk_sizes.append(remainder)
    return input_ids[0].split_with_sizes(chunk_sizes)

def compute_perplexity(input_chunks):
    total_perplexity = 0
    for chunk in input_chunks:
        total_perplexity += calculate_chunk_perplexity(chunk)
    return total_perplexity / len(input_chunks)


def calculate_chunk_perplexity(chunk):
    """
    Adds `[CLS]` and `[SEP]` tokens, and calculates perplexity for the chunk.
    """
    chunk = torch.cat([torch.tensor([tokenizer.cls_token_id]), chunk, torch.tensor([tokenizer.sep_token_id])])
    assert chunk.size(0) <= CHUNK_SIZE, "Chunk size should not exceed `CHUNK_SIZE`"
    with torch.no_grad():
        outputs = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        loss = outputs.loss
    return torch.exp(loss).item()

def load_dataset():
    data = datasets.load_dataset(DATASET, SPLIT, split=TRAIN_SPLIT)
    return [(q, (nlp(a), 0)) for q, a in zip(data['question'], data['long_answer'])]


def process_data(data):
    pos_counts, punctuation_counts, function_word_counts, total_tokens, total_sentences, sentence_perplexities = initialize_counters()
    process_each_answer(data, pos_counts, punctuation_counts, function_word_counts, total_tokens, total_sentences,
                        sentence_perplexities)
    return pos_counts, punctuation_counts, function_word_counts, total_tokens, total_sentences, sentence_perplexities


def initialize_counters():
    return Counter(), Counter(), Counter(), 0, 0, []


def process_each_answer(data, pos_counts, punctuation_counts, function_word_counts, total_tokens, total_sentences,
                        sentence_perplexities):
    for question, (answer_doc, _) in data:
        total_tokens += len(answer_doc)
        total_sentences += len(list(answer_doc.sents))
        process_each_sentence(answer_doc, sentence_perplexities)
        process_each_token(answer_doc, pos_counts, punctuation_counts, function_word_counts)


def process_each_sentence(answer_doc, sentence_perplexities):
    for sent in answer_doc.sents:
        sentence_perplexity = calculate_perplexity(sent.text)
        if sentence_perplexity < 0:
            print(f"Negative perplexity found: {sentence_perplexity} for sentence: {sent.text}")
        sentence_perplexities.append(sentence_perplexity)


def process_each_token(answer_doc, pos_counts, punctuation_counts, function_word_counts):
    for token in answer_doc:
        pos_counts[token.pos_] += 1
        if token.is_punct:
            punctuation_counts[token.text] += 1
        if token.text in {'a', 'in', 'of', 'the'}:
            function_word_counts[token.text] += 1


def calculate_averages(total_tokens, total_sentences, sentence_perplexities):
    avg_sentence_length = total_tokens / total_sentences if total_sentences > 0 else 0
    avg_sentence_perplexity = sum(sentence_perplexities) / len(sentence_perplexities) if sentence_perplexities else 0
    return avg_sentence_length, avg_sentence_perplexity


def print_results(pos_counts, punctuation_counts, function_word_counts, avg_sentence_length, avg_sentence_perplexity,
                  text_perplexity):
    print(f"Frequency of adjectives: {pos_counts['ADJ']}")
    print(f"Frequency of adverbs: {pos_counts['ADV']}")
    print(f"Frequency of conjunctions: {pos_counts['CONJ']}")
    print(f"Frequency of nouns: {pos_counts['NOUN']}")
    print(f"Frequency of numbers: {pos_counts['NUM']}")
    print(f"Frequency of pronouns: {pos_counts['PRON']}")
    print(f"Frequency of verbs: {pos_counts['VERB']}")
    print(f"Frequency of commas: {punctuation_counts[',']}")
    print(f"Frequency of fullstops: {punctuation_counts['.']}")
    print(f"Frequency of special character '-': {punctuation_counts['-']}")
    print(f"Frequency of function word 'a': {function_word_counts['a']}")
    print(f"Frequency of function word 'in': {function_word_counts['in']}")
    print(f"Frequency of function word 'of': {function_word_counts['of']}")
    print(f"Frequency of function word 'the': {function_word_counts['the']}")
    print(f"Average sentence length: {avg_sentence_length}")
    print(f"Average sentence perplexity: {avg_sentence_perplexity}")
    print(f"Overall text perplexity: {text_perplexity}")


def plot_perplexities(sentence_perplexities):
    filtered_sentence_perplexities = [p for p in sentence_perplexities if p <= 10]

    sns.set_style('white')  # Set the plot style to white background
    sns.kdeplot(filtered_sentence_perplexities, color='skyblue', fill=True)
    plt.title('Density Plot of Sentence Perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Density')

    # Set the limits of x-axis and the ticks on the x-axis
    plt.xlim(0, 10)
    plt.xticks(range(0, 13))

    plt.ylim(0, 0.2)  # Set the maximum value of the y-axis to 1

    sns.despine()  # Remove the top and right spines

    plt.show()


def main():
    # Load and process PubMed dataset
    pubmed_data = load_dataset()
    pos_counts, punctuation_counts, function_word_counts, total_tokens, total_sentences, sentence_perplexities = process_data(
        pubmed_data)
    avg_sentence_length, avg_sentence_perplexity = calculate_averages(total_tokens, total_sentences,
                                                                      sentence_perplexities)

    # Calculate overall text perplexity
    all_text = ' '.join([answer_doc.text for _, (answer_doc, _) in pubmed_data])
    text_perplexity = calculate_perplexity(all_text)

    print("Results for PubMed Dataset:")
    print_results(pos_counts, punctuation_counts, function_word_counts, avg_sentence_length, avg_sentence_perplexity,
                  text_perplexity)
    plot_perplexities(sentence_perplexities)


if __name__ == "__main__":
    main()