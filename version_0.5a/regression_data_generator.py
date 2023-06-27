import pandas as pd
import os
import spacy
from helperfunc import load_model, load_and_count, remove_prefix, extract_prompts_and_texts, \
    count_pos_tags_and_special_elements, calculate_readability_scores, calculate_perplexity, \
    calculate_average_word_length, calculate_average_sentence_length, calculate_cosine_similarity, \
    calculate_cosine_similarities_for_sentences_in_text

# Constants
nlp = spacy.load('en_core_web_sm')
FUNCTION_WORDS = {'a', 'in', 'of', 'the'}


def prepare_data_for_regression(data_file, save_file='data_matrix.csv', chunk_size=5):
    """
    This function prepares the data for regression analysis by extracting features and labels from the data.

    Args:
    data_file (str): The path to the full_data.csv file.
    save_file (str): The path to the file where the processed data will be saved.
    chunk_size (int): The number of rows to process at a time.

    Returns:
    data_matrix (DataFrame): A DataFrame where each row represents a text, each column represents a feature,
                            and the last column is the label.
    """
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Load saved data if it exists
    if os.path.exists(save_file):
        saved_data = pd.read_csv(save_file)
        processed_rows = len(saved_data)
    else:
        saved_data = pd.DataFrame()
        processed_rows = 0

    total_rows_processed = processed_rows  # total rows processed in this session

    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
        feature_list = []
        data = list(chunk.itertuples(index=False, name=None))
        texts, labels = remove_prefix(data)
        prompts_and_texts = extract_prompts_and_texts(data)

        for i, ((prompt, text), label) in enumerate(zip(prompts_and_texts, labels)):
            # Skip rows that have already been processed
            if total_rows_processed < processed_rows:
                total_rows_processed += 1
                continue

            try:
                # Count POS tags in the text
                pos_counts, punctuation_counts, function_word_counts = count_pos_tags_and_special_elements(text)

                # Calculate the Flesch Reading Ease and Flesch-Kincaid Grade Level
                flesch_reading_ease, flesch_kincaid_grade_level = calculate_readability_scores(text)

                # Calculate the average word length
                avg_word_length = calculate_average_word_length([text])

                # Calculate the average sentence length
                avg_sentence_length = calculate_average_sentence_length([text])

                # Calculate the perplexity of the text and average sentence perplexity
                text_encoded = tokenizer.encode(text, truncation=True, max_length=510)
                text = tokenizer.decode(text_encoded)
                text = text.replace('<s>', '').replace('</s>', '')
                text_perplexity = calculate_perplexity(text, model, tokenizer)
                sentence_perplexities = [calculate_perplexity(sentence.text, model, tokenizer) for sentence in nlp(text).sents]
                sentence_perplexities = [p for p in sentence_perplexities if p is not None]
                avg_sentence_perplexity = sum(sentence_perplexities) / len(
                    sentence_perplexities) if sentence_perplexities else None

                # Calculate the frequency of uppercase letters
                uppercase_freq = sum(1 for char in text if char.isupper()) / len(text)

                # Calculate the cosine similarity for the prompt and text
                prompt_text_cosine_similarity = calculate_cosine_similarity(prompt, text, model, tokenizer)

                # Calculate the average cosine similarity for sentences in the text
                sentence_cosine_similarities = calculate_cosine_similarities_for_sentences_in_text(text, model, tokenizer)
                avg_sentence_cosine_similarity = None
                if sentence_cosine_similarities:
                    avg_sentence_cosine_similarity = sum(sentence_cosine_similarities) / len(sentence_cosine_similarities)
                else:
                    print("WARNING: No sentence cosine similarities calculated for text:", text)

                # Prepare a dictionary to append to the feature list
                features = {
                    'ADJ': pos_counts.get('ADJ', 0),
                    'ADV': pos_counts.get('ADV', 0),
                    'CONJ': pos_counts.get('CCONJ', 0),
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
                    'uppercase_freq': uppercase_freq,
                    'flesch_reading_ease': flesch_reading_ease,
                    'flesch_kincaid_grade_level': flesch_kincaid_grade_level,
                    'avg_word_length': avg_word_length,
                    'avg_sentence_length': avg_sentence_length,
                    'text_perplexity': text_perplexity,
                    'avg_sentence_perplexity': avg_sentence_perplexity,
                    'prompt_text_cosine_similarity': prompt_text_cosine_similarity,
                    'avg_sentence_cosine_similarity': avg_sentence_cosine_similarity,
                    'label': label
                }

                # Add the feature dictionary to the feature list
                feature_list.append(features)

                # Print progress
                print(f"Processed row {total_rows_processed + 1}")
                total_rows_processed += 1

            except Exception as e:
                print(f"Error processing row {total_rows_processed + 1}: {e}")
                continue

        try:
            # Convert the list of dictionaries into a DataFrame
            new_data = pd.DataFrame(feature_list).fillna(0)

            # Append new data to saved data and save
            saved_data = pd.concat([saved_data, new_data])
            saved_data.to_csv(save_file, index=False)

            # Clear the feature list for the next batch
            feature_list.clear()

        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

    return saved_data


prepare_data_for_regression('extracted_data/full_data.csv',save_file='data_matrix_v2.csv')


def prepare_and_save_datasets(datasets, output_dir, full_data_path=None):
    """
    Prepare data for regression for all datasets, save them, and return a combined DataFrame.

    Args:
    datasets (list of str): The list of datasets.
    output_dir (str): The directory where to save the prepared data.
    full_data_path (str): The path to the full_data.csv file.

    Returns:
    combined_data (DataFrame): The combined DataFrame with prepared data from all datasets.
    """
    output_dir = 'Model_Ready_Data'
    data_frames = []
    os.makedirs(output_dir, exist_ok=True)

    if full_data_path is None:
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
    else:
        print(f"Preparing data for regression for full dataset from '{full_data_path}'...")

        # Load the full data
        full_data = pd.read_csv(full_data_path)

        # Prepare the data for regression
        prepared_data = prepare_data_for_regression(full_data, "full_data")

        print(f"Data prepared for full dataset. Saving to file...")

        # Save the prepared data
        prepared_data.to_csv(f'{output_dir}/full_data_reg_ready.csv', index=False)

        print(f"Data for full dataset saved successfully.")

        data_frames.append(prepared_data)

    print("Combining prepared data...")

    # Combine the prepared data
    combined_data = pd.concat(data_frames, ignore_index=True)

    print("Data combined successfully.")

    return combined_data
