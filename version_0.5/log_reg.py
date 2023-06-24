import pandas as pd
from summary_statistics import count_pos_tags_and_special_elements, calculate_readability_scores, \
    calculate_average_word_length, calculate_perplexity, calculate_average_sentence_length, \
    load_model, remove_prefix


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
