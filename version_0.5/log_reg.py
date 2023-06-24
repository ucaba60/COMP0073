import pandas as pd
from summary_statistics import count_pos_tags_and_special_elements, calculate_readability_scores, \
    calculate_average_word_length, calculate_perplexity, calculate_average_sentence_length, \
    load_model, remove_prefix
import os
import spacy
import argparse


nlp = spacy.load('en_core_web_sm')
FUNCTION_WORDS = {'a', 'in', 'of', 'the'}


def combine_data_files(data_files):
    """
    This function reads all the files from the given list of file paths and
    combines them into a large DataFrame.

    Args:
    data_files (list of str): The list of file paths.

    Returns:
    combined_data (DataFrame): A DataFrame combining all the data from the files.
    """
    # Initialize an empty list to store individual DataFrames
    data_frames = []

    for file in data_files:
        # Check if the file exists
        if os.path.isfile(file):
            # Load the file into a DataFrame and append it to the list
            data_frames.append(pd.read_csv(file))
        else:
            print(f"The file '{file}' does not exist.")

    # Concatenate all the DataFrames in the list
    combined_data = pd.concat(data_frames, ignore_index=True)

    return combined_data


def prepare_data_for_regression(data, dataset_name):
    """
    This function prepares the data for regression analysis by extracting features and labels from the data.

    Args:
    data (list of tuples): The data from the dataset. Each element of the list is a tuple, where the first element
    is the text and the second element is its label.

    Returns:
    data_matrix (DataFrame): A DataFrame where each row represents a text, each column represents a feature,
                            and the last column is the label.
    """
    # Initialize lists to store features and labels
    feature_list = []

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Remove prefixes
    texts, labels = remove_prefix(dataset_name, data)

    for text, label in zip(texts, labels):
        # Count POS tags in the text
        pos_counts, punctuation_counts, function_word_counts = count_pos_tags_and_special_elements(text)

        # Calculate the Flesch Reading Ease and Flesch-Kincaid Grade Level
        flesch_reading_ease, flesch_kincaid_grade_level = calculate_readability_scores(text)

        # Calculate the average word length
        avg_word_length = calculate_average_word_length([text])

        # Calculate the average sentence length
        avg_sentence_length = calculate_average_sentence_length([text])

        # Calculate the perplexity of the text and average sentence perplexity
        # Truncate the text to the first 512 tokens
        text_encoded = tokenizer.encode(text, truncation=True, max_length=510)
        text = tokenizer.decode(text_encoded)

        text_perplexity = calculate_perplexity(text, model, tokenizer)
        sentence_perplexities = [calculate_perplexity(sentence.text, model, tokenizer) for sentence in nlp(text).sents]
        sentence_perplexities = [p for p in sentence_perplexities if p is not None]
        avg_sentence_perplexity = sum(sentence_perplexities) / len(
            sentence_perplexities) if sentence_perplexities else None

        # Calculate the frequency of uppercase letters
        uppercase_freq = sum(1 for char in text if char.isupper()) / len(text)

        # Prepare a dictionary to append to the feature list
        features = {
            'ADJ': pos_counts.get('ADJ', 0),
            'ADV': pos_counts.get('ADV', 0),
            'CONJ': pos_counts.get('CONJ', 0),
            'NOUN': pos_counts.get('NOUN', 0),
            'NUM': pos_counts.get('NUM', 0),
            'VERB': pos_counts.get('VERB', 0),
            'COMMA': punctuation_counts.get(',', 0),
            'FULLSTOP': punctuation_counts.get('.', 0),
            'SPECIAL-': punctuation_counts.get('-', 0),
            'FUNCTION-A': function_word_counts.get('a', 0),
            'FUNCTION-IN': function_word_counts.get('in', 0),
            'FUNCTION-OF': function_word_counts.get('of', 0),
            'FUNCTION-THE': function_word_counts.get('the', 0),
            'uppercase_freq': uppercase_freq,  # new feature
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade_level': flesch_kincaid_grade_level,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'text_perplexity': text_perplexity,
            'avg_sentence_perplexity': avg_sentence_perplexity,
            'label': label
        }

        # Add the feature dictionary to the feature list
        feature_list.append(features)

    # Convert the list of dictionaries into a DataFrame
    data_matrix = pd.DataFrame(feature_list).fillna(0)

    return data_matrix


def prepare_and_save_datasets(datasets, output_dir):
    """
    Prepare data for regression for all datasets, save them, and return a combined DataFrame.

    Args:
    datasets (list of str): The list of datasets.
    output_dir (str): The directory where to save the prepared data.

    Returns:
    combined_data (DataFrame): The combined DataFrame with prepared data from all datasets.
    """
    output_dir = 'Model_Ready_Data'
    data_frames = []
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"Preparing data for regression for dataset '{dataset_name}'...")

        # Load the data
        data = pd.read_csv(f'Labelled_Data/{dataset_name}_preprocessed_data.csv')

        # Prepare the data for regression
        prepared_data = prepare_data_for_regression(data, dataset_name)

        print(f"Data prepared for dataset '{dataset_name}'. Saving to file...")

        # Save the prepared data
        prepared_data.to_csv(f'{output_dir}/{dataset_name}_reg_ready.csv', index=False)

        print(f"Data for dataset '{dataset_name}' saved successfully.")

        data_frames.append(prepared_data)

    print("Combining prepared data...")

    # Combine the prepared data
    combined_data = pd.concat(data_frames, ignore_index=True)

    print("Data combined successfully.")

    return combined_data


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    args: command-line arguments
    """
    parser = argparse.ArgumentParser(description="Prepare datasets for regression.")
    parser.add_argument('--datasets', nargs='+', help='List of datasets to process')

    return parser.parse_args()


def main():
    # Parse the command-line arguments
    args = parse_arguments()

    # Prepare and save the datasets
    output_dir = 'Model_Ready_Data'
    combined_data = prepare_and_save_datasets(args.datasets, output_dir)

    # Save the combined data
    combined_data.to_csv(f'{output_dir}/combined_data_reg_ready.csv', index=False)


if __name__ == "__main__":
    main()


