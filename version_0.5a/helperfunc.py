import spacy
from collections import Counter
import torch
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
import textstat
from sklearn.preprocessing import normalize
import pandas as pd
import tiktoken
from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse
import os
from pathlib import Path
import time
import torch
from scipy.spatial.distance import cosine

# Constants
nlp = spacy.load('en_core_web_sm')
FUNCTION_WORDS = {'a', 'in', 'of', 'the'}


def remove_prefix(dataset_name, data):
    """
    This function removes a predefined prefix from each text in a given dataset.

    Args:
    dataset_name (str): The name of the dataset.
    data (list of tuples): The data from the dataset. Each element of the list is a tuple, where the first element
    is the text and the second element is its label.

    Returns:
    texts (list): The list of texts after the prefix has been removed.
    labels (list): The list of labels corresponding to the texts.
    """

    texts, labels = zip(*data)

    if dataset_name == 'pubmed_qa':
        texts = [text.split("Answer:", 1)[1].strip() for text in texts if "Answer:" in text]
    elif dataset_name == 'writingprompts':
        texts = [text.split("Story:", 1)[1].strip() for text in texts if "Story:" in text]
    elif dataset_name == 'cnn_dailymail':
        texts = [text.split("Article:", 1)[1].strip() for text in texts if "Article:" in text]
    elif dataset_name == 'gpt':
        texts = [text.split("Answer:", 1)[1].strip() if "Answer:" in text else text for text in texts]
        texts = [text.split("Story:", 1)[1].strip() if "Story:" in text else text for text in texts]
        texts = [text.split("Article:", 1)[1].strip() if "Article:" in text else text for text in texts]

    return list(texts), list(labels)


def count_pos_tags_and_special_elements(text):
    # CHECKED
    """
      This function counts the frequency of POS (Part of Speech) tags, punctuation marks, and function words in a given text.
      It uses the SpaCy library for POS tagging.

      Args:
      text (str): The text for which to count POS tags and special elements.

      Returns:
      pos_counts (dict): A dictionary where keys are POS tags and values are their corresponding count.
      punctuation_counts (dict): A dictionary where keys are punctuation marks and values are their corresponding count.
      function_word_counts (dict): A dictionary where keys are function words and values are their corresponding count.

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
    This function calculates the Flesch Reading Ease and Flesch-Kincaid Grade Level of a text using the textstat library.

    Args:
    text (str): The text to score.

    Returns:
    flesch_reading_ease (float): The Flesch Reading Ease score of the text.
    flesch_kincaid_grade_level (float): The Flesch-Kincaid Grade Level of the text.

    """
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text)

    return flesch_reading_ease, flesch_kincaid_grade_level


def load_and_count(dataset_name, data):
    """
       This function loads the texts from the dataset and calculates the frequency of POS tags, punctuation marks,
       and function words.

       Args:
       dataset_name (str): The name of the dataset.
       data (list of tuples): The data from the dataset. Each element of the list is a tuple, where the first element
       is the text and the second element is its label.

       Returns:
       overall_pos_counts (Counter): A Counter object of POS tag frequencies.
       overall_punctuation_counts (Counter): A Counter object of punctuation mark frequencies.
       overall_function_word_counts (Counter): A Counter object of function word frequencies.
    """

    # CHECKED
    # Extract texts
    texts, labels = remove_prefix(dataset_name, data)

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
      This function loads a pre-trained model and its corresponding tokenizer from the Hugging Face model hub.

      Returns:
      model: The loaded model.
      tokenizer: The tokenizer corresponding to the model.

    """
    # model_name = 'allenai/scibert_scivocab_uncased'
    # model = AutoModelForMaskedLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def calculate_average_word_length(texts):
    """
     This function calculates the average word length of a list of texts using the SpaCy library.

     Args:
     texts (list): The list of texts.

     Returns:
     (float): The average word length.

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
    This function calculates the average sentence length of a list of texts using the SpaCy library.

    Args:
    texts (list): The list of texts.

    Returns:
    avg_sentence_length (float): The average sentence length.
    """
    sentence_lengths = []

    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            sentence_lengths.append(len(sent))

    return mean(sentence_lengths)


def calculate_perplexity(text, model, tokenizer):
    """
    Calculates the perplexity of a text using a language model and tokenizer.

    Args:
    text (str): The text for which perplexity will be calculated.
    model: The language model used to calculate perplexity.
    tokenizer: The tokenizer used to tokenize the text.

    Returns:
    perplexity (float or None): The calculated perplexity of the text, or None if the text is too long.
    """

    try:
        input_ids = tokenizer.encode(text, return_tensors='pt')
        # Truncate the text to the first 512 tokens
        input_ids = input_ids[:, :512]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()
    except Exception as e:
        print(f"An error occurred in calculate_perplexity: {e}")
        return None


def calculate_cosine_similarity(text1, text2, model, tokenizer):
    """
    This function calculates cosine similarity between two texts.

    Args:
    text1 (str): The first text.
    text2 (str): The second text.
    model: The language model used to generate word embeddings.
    tokenizer: The tokenizer used to tokenize the text.

    Returns:
    cosine_similarity (float): The cosine similarity between the word embeddings of the two texts.
    """

    # Tokenize the texts
    input_ids1 = tokenizer.encode(text1, return_tensors="pt")
    input_ids2 = tokenizer.encode(text2, return_tensors="pt")

    # Generate word embeddings for the texts
    embeddings1 = model.roberta(input_ids1)[0].mean(dim=1).squeeze().detach()
    embeddings2 = model.roberta(input_ids2)[0].mean(dim=1).squeeze().detach()

    # Convert embeddings to numpy arrays
    embeddings1_np = embeddings1.numpy()
    embeddings2_np = embeddings2.numpy()

    # Apply L2 normalization to the embeddings
    normalized_embeddings1 = normalize(embeddings1_np.reshape(1, -1)).squeeze()
    normalized_embeddings2 = normalize(embeddings2_np.reshape(1, -1)).squeeze()

    # Convert back to torch tensors
    normalized_embeddings1 = torch.from_numpy(normalized_embeddings1)
    normalized_embeddings2 = torch.from_numpy(normalized_embeddings2)

    # Calculate cosine similarity
    cosine_similarity = 1 - cosine(embeddings1.numpy(), embeddings2.numpy())

    return cosine_similarity


def extract_prompts_and_texts(dataset_name, data):
    """
    This function extracts prompts and texts from the data for a specified dataset.

    Args:
    dataset_name (str): The name of the dataset.
    data (list of tuples): The data. Each tuple consists of a text (including prompt) and a label.

    Returns:
    prompts_and_texts (list of tuples): The list of tuples where each tuple contains a prompt and a text.
    """

    prompts_and_texts = []

    full_texts, _ = zip(*data)
    texts, labels = remove_prefix(dataset_name, data)

    starting_points = ["Question:", "Prompt:", "Article:"]
    end_points = ["Answer:", "Story:", "Summary:"]

    for full_text, text in zip(full_texts, texts):
        # Split the full_text depending on the dataset
        if dataset_name == 'pubmed_qa':
            split_text = full_text.split("Question:", 1)
            if len(split_text) == 2:
                _, temp_prompt = split_text
                prompt, _ = temp_prompt.split("Answer:", 1)
        elif dataset_name == 'writingprompts':
            split_text = full_text.split("Prompt:", 1)
            if len(split_text) == 2:
                _, temp_prompt = split_text
                prompt, _ = temp_prompt.split("Story:", 1)
        elif dataset_name == 'cnn_dailymail':
            split_text = full_text.split("Article:", 1)
            if len(split_text) == 2:
                _, temp_prompt = split_text
                prompt, _ = temp_prompt.split("Summary:", 1)
        elif dataset_name == 'gpt':
            # Identify the starting point for each entry in the 'gpt' dataset
            for starting_point in starting_points:
                if starting_point in full_text:
                    split_text = full_text.split(starting_point, 1)
                    if len(split_text) == 2:
                        _, temp_prompt = split_text
                        for end_point in end_points:
                            if end_point in temp_prompt:
                                prompt, _ = temp_prompt.split(end_point, 1)
                                break
                    break

        prompt = prompt.strip()  # remove leading and trailing whitespaces
        prompts_and_texts.append((prompt, text))  # append the prompt and text to the list

    return prompts_and_texts


def calculate_cosine_similarities_for_dataset(dataset_name, model, tokenizer):
    """
    This function calculates cosine similarities for all (prompt, text) pairs in a dataset.

    Args:
    dataset_name (str): The name of the dataset.
    model: The language model used to generate word embeddings.
    tokenizer: The tokenizer used to tokenize the text.

    Returns:
    cosine_similarities (list of floats): The list of cosine similarities.
    """

    prompts_and_texts = extract_prompts_and_texts(dataset_name, data)

    cosine_similarities = []
    for prompt, text in prompts_and_texts:
        cosine_similarity = calculate_cosine_similarity(prompt, text, model, tokenizer)
        cosine_similarities.append(cosine_similarity)

    return cosine_similarities


def calculate_cosine_similarities_for_sentences_in_text(text, model, tokenizer):
    """
    This function calculates cosine similarities for all consecutive pairs of sentences in a single text.

    Args:
    text (str): The text for which to calculate cosine similarities.
    model: The language model used to generate word embeddings.
    tokenizer: The tokenizer used to tokenize the text.

    Returns:
    cosine_similarities (list of floats): The list of cosine similarities.
    """

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    cosine_similarities = []

    for i in range(len(sentences) - 1):
        cosine_similarity = calculate_cosine_similarity(sentences[i], sentences[i + 1], model, tokenizer)
        cosine_similarities.append(cosine_similarity)

    return cosine_similarities