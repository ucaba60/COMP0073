import spacy
from collections import Counter
import torch
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
import textstat
import pandas as pd
import tiktoken
from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse
import os
from pathlib import Path
import time


# ------------------------------------------------------------------------------------------#
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


    texts = data['text'].tolist()
    labels = data['label'].tolist()

    if dataset_name == 'pubmed_qa':
        texts = [text.split("Answer:", 1)[1].strip() for text in texts]  # Strip the 'Answer:' prefix'
    elif dataset_name == 'writingprompts':
        texts = [text.split("Story:", 1)[1].strip() for text in texts]  # Stripping the 'Story: ' string
    elif dataset_name == 'cnn_dailymail':
        texts = [text.split("Article:", 1)[1].strip() for text in texts]  # Stripping the 'Article: ' string
    elif dataset_name == 'gpt':
        texts = [text.split("Answer:", 1)[1].strip() if "Answer:" in text else text for text in texts]
        texts = [text.split("Story:", 1)[1].strip() if "Story:" in text else text for text in texts]
        texts = [text.split("Article:", 1)[1].strip() if "Article:" in text else text for text in texts]

    return texts, labels


def average_token_count(dataset_name, data):
    """
    Calculates the average number of tokens in the answers of a dataset.

    Returns:
        float: Average number of tokens in the answers of a dataset
    """
    texts, labels = remove_prefix(dataset_name, data)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    total_tokens = 0

    for text in texts:
        num_tokens = len(encoding.encode(text))
        total_tokens += num_tokens

    average_tokens = total_tokens / len(texts)

    return average_tokens


# PUBMED = 54
# WP = 780
# CNN = 794


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


def summary_statistics(dataset_name, data):
    # CHECKED
    """
       Calculates various summary statistics for a dataset.

       Args:
       dataset_name (str): The name of the dataset.
       data (dict): The data from the dataset.

       Returns:
       dict: A dictionary containing various summary statistics of the data.
   """
    texts, labels = remove_prefix(dataset_name, data)

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


def plot_perplexities(sentence_perplexities, text_perplexities, dataset_name):
    """
    Plots Kernel Density Estimates of the sentence and text perplexities.

    Args:
    sentence_perplexities (list of float): The perplexities of the sentences.
    text_perplexities (list of float): The perplexities of the texts.
    """

    # Define the data directory
    data_directory = 'Labelled_Data'

    # Plot sentence perplexities
    plt.figure(figsize=(12, 6))
    sns.kdeplot(sentence_perplexities, color='skyblue', fill=True)
    plt.title('Density Plot of Sentence Perplexities')
    plt.xlabel('Perplexity')
    plt.xlim(0, 12)  # Limit x-axis to 12 for sentence perplexity

    # Save the sentence perplexities plot
    plt.savefig(os.path.join(data_directory, f"{dataset_name}_sentence_perplexities_plot.png"))

    plt.show()

    # Plot text perplexities
    plt.figure(figsize=(12, 6))
    sns.kdeplot(text_perplexities, color='skyblue', fill=True)
    plt.title('Density Plot of Text Perplexities')
    plt.xlabel('Perplexity')
    plt.xlim(0, 10)  # Limit x-axis to 10 for text perplexity

    # Save the text perplexities plot
    plt.savefig(os.path.join(data_directory, f"{dataset_name}_text_perplexities_plot.png"))

    plt.show()


def main():
    # Get start time
    start_time = time.time()

    print("Script started at ", time.ctime(start_time))

    # Define the data directory
    data_directory = 'Labelled_Data'

    # Handle command-line arguments
    parser = argparse.ArgumentParser(description='Print summary statistics for datasets.')
    parser.add_argument('--datasets', nargs='*',
                        default=[f.stem.replace('_Human_data', '') for f in Path(data_directory).glob('*.csv')],
                        help='List of paths to datasets (CSV files).')
    parser.add_argument('--plot', action='store_true',
                        help='Add this flag to plot the perplexities.')
    args = parser.parse_args()

    for dataset_name in args.datasets:
        # Construct the full filename
        filename = f"{dataset_name}_Human_data.csv"

        # Construct the full path to the dataset
        dataset_path = os.path.join(data_directory, filename)

        # Load the dataset
        data = pd.read_csv(dataset_path)

        # Convert the DataFrame to a list of tuples
        data = list(data.itertuples(index=False, name=None))

        print(f"\nStatistics for {dataset_name} dataset:\n")

        # Calculate the statistics
        statistics = summary_statistics(dataset_name, data)

        # Print the statistics
        print_statistics(statistics)

        # If --plot flag is used, plot the perplexities
        if args.plot:
            plot_perplexities(statistics['sentence_perplexities'], statistics['text_perplexities'], dataset_name)

            # Save the plot before showing
            plt.savefig(os.path.join(data_directory, f"{dataset_name}_perplexities_plot.png"))

            # Then show the plot
            plt.show()

        # Save the statistics
        pd.Series(statistics).to_csv(os.path.join(data_directory, f"{dataset_name}_stats.csv"))

    # Get end time
    end_time = time.time()
    print("Script finished at ", time.ctime(end_time))

    # Calculate and print execution time
    execution_time = round((end_time - start_time) / 60, 2)  # Converts the execution time to minutes
    print(f"The script took {execution_time} minutes to execute.")


if __name__ == "__main__":
    main()