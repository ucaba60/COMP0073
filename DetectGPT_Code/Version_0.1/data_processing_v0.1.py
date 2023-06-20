import spacy
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
import textstat

# Constants
nlp = spacy.load('en_core_web_sm')
FUNCTION_WORDS = {'a', 'in', 'of', 'the'}


def process_prefix(dataset_name, data):
    """
    Process the prefixes in the given dataset.

    Args:
    dataset_name (str): The name of the dataset.
    data (list of tuples): The data from the dataset.

    Returns:
    list: The list of texts after stripping the prefix.
    """
    texts, labels = zip(*data)

    if dataset_name == 'pubmed_qa':
        texts = [text.split("Answer:", 1)[1].strip() for text in texts]  # Strip the 'Answer:' prefix
    elif dataset_name == 'writingprompts':
        texts = [text.split("Story:", 1)[1].strip() for text in texts]  # Stripping the 'Story: ' string

    return texts, labels


def count_pos_tags_and_special_elements(text):
    # CHECKED
    """
    Counts the frequency of POS tags, punctuation marks and function words in a given text.

    Args:
    text (str): The text for which to count POS tags and special elements.

    Returns:
    tuple: A tuple containing two dictionaries, where keys are POS tags and punctuation marks
           and values are their corresponding count.
    """
    # Use SpaCy to parse the text
    doc = nlp(text)

    # Create a counter of POS tags
    pos_counts = Counter(token.pos_ for token in doc)

    # Create a counter of punctuation marks
    punctuation_counts = Counter(token.text for token in doc if token.pos_ == 'PUNCT')

    # Create a counter of function words
    function_word_counts = Counter(token.text for token in doc if token.lower_ in FUNCTION_WORDS)

    return dict(pos_counts), dict(punctuation_counts), dict(function_word_counts)


def calculate_readability_scores(text):
    """
    Calculate the Flesch Reading Ease and Flesch-Kincaid Grade Level of a text.

    Args:
    text (str): The text to score.

    Returns:
    tuple: A tuple containing the Flesch Reading Ease and Flesch-Kincaid Grade Level.
    """
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text)

    return flesch_reading_ease, flesch_kincaid_grade_level


def load_and_count(dataset_name, data):
    # CHECKED
    # Extract texts
    texts, labels = process_prefix(dataset_name, data)

    # Calculate POS tag frequencies for the texts
    pos_frequencies, punctuation_frequencies, function_word_frequencies = zip(
        *[count_pos_tags_and_special_elements(text) for text in texts])

    # Then, sum the dictionaries to get the overall frequencies
    overall_pos_counts = Counter()
    for pos_freq in pos_frequencies:
        overall_pos_counts += Counter(pos_freq)

    overall_punctuation_counts = Counter()
    for punct_freq in punctuation_frequencies:
        overall_punctuation_counts += Counter(punct_freq)

    overall_function_word_counts = Counter()
    for function_word_freq in function_word_frequencies:
        overall_function_word_counts += Counter(function_word_freq)

    return overall_pos_counts, overall_punctuation_counts, overall_function_word_counts


def load_model():
    # CHECKED
    """
    Load the model and tokenizer.
    Returns a model and tokenizer.
    """
    model_name = 'allenai/scibert_scivocab_uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def calculate_average_word_length(texts):
    """
    Calculate the average word length of a list of texts using SpaCy.

    Args:
    texts (list): The list of texts.

    Returns:
    float: The average word length.
    """
    word_lengths = []

    for text in texts:
        doc = nlp(text)
        for token in doc:
            if not token.is_punct:  # ignore punctuation
                word_lengths.append(len(token.text))

    return mean(word_lengths)


def calculate_average_sentence_length(texts):
    # CHEKCED
    """
    Calculate the average sentence length of a list of texts using SpaCy.

    Args:
    texts (list): The list of texts.

    Returns:
    float: The average sentence length.
    """
    sentence_lengths = []

    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            sentence_lengths.append(len(sent))

    return mean(sentence_lengths)


def calculate_perplexity(text, model, tokenizer):
    # CHECKED
    """
    Calculate the perplexity of a piece of text.
    """
    # tokenize the input, add special tokens and return tensors
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # if the text is too long, skip it
    # this step has the extra effect of removing examples with low-quality/garbage content
    if len(input_ids[0]) > 512:
        return None

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
    loss = output.loss
    return torch.exp(loss).item()  # perplexity is e^loss


def summary_statistics(dataset_name, data):
    # CHECKED
    texts, labels = process_prefix(dataset_name, data)

    model, tokenizer = load_model()
    overall_pos_counts, overall_punctuation_counts, overall_function_word_counts = load_and_count(dataset_name, data)
    readability_scores = [calculate_readability_scores(text) for text in texts]
    average_flesch_reading_ease = mean(score[0] for score in readability_scores)
    average_flesch_kincaid_grade_level = mean(score[1] for score in readability_scores)
    average_word_length = calculate_average_word_length(texts)
    average_sentence_length = calculate_average_sentence_length(texts)
    text_perplexities = [calculate_perplexity(text, model, tokenizer) for text in texts]
    text_perplexities = [p for p in text_perplexities if p is not None]
    average_text_perplexity = sum(text_perplexities) / len(text_perplexities)
    sentences = [sentence.text for text in texts for sentence in nlp(text).sents]
    sentence_perplexities = [calculate_perplexity(sentence, model, tokenizer) for sentence in sentences]
    sentence_perplexities = [p for p in sentence_perplexities if p is not None]
    average_sentence_perplexity = sum(sentence_perplexities) / len(sentence_perplexities)
    return {
        'pos_freqs': overall_pos_counts,
        'punctuation_freqs': overall_punctuation_counts,
        'function_word_freqs': overall_function_word_counts,
        'average_word_length': average_word_length,
        'average_flesch_reading_ease': average_flesch_reading_ease,
        'average_flesch_kincaid_grade_level': average_flesch_kincaid_grade_level,
        'average_sentence_length': average_sentence_length,
        'average_text_perplexity': average_text_perplexity,
        'average_sentence_perplexity': average_sentence_perplexity,
        'sentence_perplexities': sentence_perplexities,  # added this
        'text_perplexities': text_perplexities  # and this
    }


def print_statistics(statistics):
    # CHECKED
    pos_freqs = statistics['pos_freqs']
    punctuation_freqs = statistics['punctuation_freqs']
    function_word_freqs = statistics['function_word_freqs']

    print(f"Frequency of adjectives: {pos_freqs.get('ADJ', 0)}")
    print(f"Frequency of adverbs: {pos_freqs.get('ADV', 0)}")
    print(f"Frequency of conjunctions: {pos_freqs.get('CCONJ', 0)}")
    print(f"Frequency of nouns: {pos_freqs.get('NOUN', 0)}")
    print(f"Frequency of numbers: {pos_freqs.get('NUM', 0)}")
    print(f"Frequency of pronouns: {pos_freqs.get('PRON', 0)}")
    print(f"Frequency of verbs: {pos_freqs.get('VERB', 0)}")
    print(f"Frequency of commas: {punctuation_freqs.get(',', 0)}")
    print(f"Frequency of fullstops: {punctuation_freqs.get('.', 0)}")
    print(f"Frequency of special character '-': {punctuation_freqs.get('-', 0)}")
    print(f"Frequency of function word 'a': {function_word_freqs.get('a', 0)}")
    print(f"Frequency of function word 'in': {function_word_freqs.get('in', 0)}")
    print(f"Frequency of function word 'of': {function_word_freqs.get('of', 0)}")
    print(f"Frequency of function word 'the': {function_word_freqs.get('the', 0)}")
    print(f"Average Flesch Reading Ease: {statistics['average_flesch_reading_ease']}")
    print(f"Average Flesch-Kincaid Grade Level: {statistics['average_flesch_kincaid_grade_level']}")
    print(f"Average word length: {statistics['average_word_length']}")
    print(f"Average sentence length: {statistics['average_sentence_length']}")
    print(f"Average sentence perplexity: {statistics['average_sentence_perplexity']}")
    print(f"Average text perplexity: {statistics['average_text_perplexity']}")


def plot_perplexities(sentence_perplexities, text_perplexities):
    """
    Plots Kernel Density Estimates of the sentence and text perplexities.

    Args:
    sentence_perplexities (list of float): The perplexities of the sentences.
    text_perplexities (list of float): The perplexities of the texts.
    """

    # Plot sentence perplexities
    plt.figure(figsize=(12, 6))
    sns.kdeplot(sentence_perplexities, color='skyblue', fill=True)
    plt.title('Density Plot of Sentence Perplexities')
    plt.xlabel('Perplexity')
    plt.xlim(0, 12)  # Limit x-axis to 12 for sentence perplexity
    plt.show()

    # Plot text perplexities
    plt.figure(figsize=(12, 6))
    sns.kdeplot(text_perplexities, color='skyblue', fill=True)
    plt.title('Density Plot of Text Perplexities')
    plt.xlabel('Perplexity')
    plt.xlim(0, 10)  # Limit x-axis to 10 for text perplexity
    plt.show()