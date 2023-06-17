import spacy
from collections import Counter
from data_generation import preprocess_data
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast
from statistics import mean


# Constants
nlp = spacy.load('en_core_web_sm')
FUNCTION_WORDS = {'a', 'in', 'of', 'the'}


def count_pos_tags_and_special_elements(text):
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


def load_and_count(dataset_name):
    # Load and preprocess the data
    data = preprocess_data(dataset_name)

    # Extract texts
    texts, labels = zip(*data)

    # Split questions and answers for pubmed_qa dataset
    if dataset_name == 'pubmed_qa':
        texts = [text.split("Answer:", 1)[1] for text in texts]

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

    # Print the frequencies
    print(f"Frequency of adjectives: {overall_pos_counts['ADJ']}")
    print(f"Frequency of adverbs: {overall_pos_counts['ADV']}")
    print(f"Frequency of conjunctions: {overall_pos_counts['CCONJ']}")
    print(f"Frequency of nouns: {overall_pos_counts['NOUN']}")
    print(f"Frequency of numbers: {overall_pos_counts['NUM']}")
    print(f"Frequency of pronouns: {overall_pos_counts['PRON']}")
    print(f"Frequency of verbs: {overall_pos_counts['VERB']}")
    print(f"Frequency of commas: {overall_punctuation_counts[',']}")
    print(f"Frequency of fullstops: {overall_punctuation_counts['.']}")
    print(f"Frequency of special character '-': {overall_punctuation_counts['-']}")
    print(f"Frequency of function word 'a': {overall_function_word_counts['a']}")
    print(f"Frequency of function word 'in': {overall_function_word_counts['in']}")
    print(f"Frequency of function word 'of': {overall_function_word_counts['of']}")
    print(f"Frequency of function word 'the': {overall_function_word_counts['the']}")


def load_model():
    """
    Load the model and tokenizer.
    Returns a model and tokenizer.
    """
    model_name = 'allenai/scibert_scivocab_uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def calculate_average_sentence_length(texts):
    """
    Calculate the average sentence length of a list of texts.

    Args:
    texts (list): The list of texts.

    Returns:
    float: The average sentence length.
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

    # Split the texts into sentences
    sentences = [sentence for text in texts for sentence in text.split('. ')]

    # Tokenize the sentences and calculate their length
    sentence_lengths = [len(tokenizer.tokenize(sentence)) for sentence in sentences]

    # Calculate and return the average sentence length
    return mean(sentence_lengths)


def calculate_perplexity(text, model, tokenizer):
    """
    Calculate the perplexity of a piece of text.
    """
    # tokenize the input, add special tokens and return tensors
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # if the text is too long, skip it
    if len(input_ids[0]) > 512:
        return None

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
    loss = output.loss
    return torch.exp(loss).item()  # perplexity is e^loss


def calculate_average_perplexities(dataset_name):
    # Load and preprocess the data
    data = preprocess_data(dataset_name)

    # Extract texts
    texts, labels = zip(*data)

    # Split questions and answers for pubmed_qa dataset
    if dataset_name == 'pubmed_qa':
        texts = [text.split("Answer:", 1)[1] for text in texts]
    elif dataset_name == 'writingprompts':
        texts = [text.split("Story:", 1)[1] for text in texts]

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Calculate the perplexity for each text
    perplexities = [calculate_perplexity(text, model, tokenizer) for text in texts]

    # Filter out None values
    perplexities = [p for p in perplexities if p is not None]

    # Calculate the average sentence length
    average_sentence_length = calculate_average_sentence_length(texts)
    print(f"Average sentence length: {average_sentence_length}")

    # Calculate and print the average text perplexity
    average_text_perplexity = sum(perplexities) / len(perplexities)
    print(f"Average text perplexity: {average_text_perplexity}")

    # Split the texts into sentences and calculate the perplexity for each sentence
    sentences = [sentence for text in texts for sentence in text.split('. ')]
    sentence_perplexities = [calculate_perplexity(sentence, model, tokenizer) for sentence in sentences]

    # Filter out None values
    sentence_perplexities = [p for p in sentence_perplexities if p is not None]

    # Calculate and print the average sentence perplexity
    average_sentence_perplexity = sum(sentence_perplexities) / len(sentence_perplexities)
    print(f"Average sentence perplexity: {average_sentence_perplexity}")