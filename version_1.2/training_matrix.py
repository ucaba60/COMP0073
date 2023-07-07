import os
import pandas as pd
from feature_extraction import remove_prefix, extract_prompts_and_texts
from feature_extraction import load_model, count_pos_tags_and_special_elements, calculate_readability_scores, \
    calculate_average_word_length, calculate_average_sentence_length, calculate_perplexity, calculate_cosine_similarity, \
    calculate_cosine_similarities_for_sentences_in_text, calculate_cosine_similarity_for_prompt_and_text
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

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

    # Extract the model name from the data_file
    file_name = data_file.split('/')[-1]  # split the input file string at the slash and take the last part (filename)
    model_name = file_name.split('_')[0]  # split the filename at the underscore and take the first part (model name)
    save_file = f'data_matrix_{model_name}.csv'  # create save_file name based on the model_name

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Load saved data if it exists
    if os.path.exists(save_file):
        saved_data = pd.read_csv(save_file)
        processed_rows = len(saved_data)
    else:
        saved_data = pd.DataFrame()
        processed_rows = 0

    # Calculate the top 10 words with the highest difference in TF-IDF scores and the vectorizer
    #     diff_words = compute_difference_tfidf_words(data_file, n_top_words=10)
    top_words = vocabulary = ['said', 'like', 'im', 'get', 'told', 'dont', 'say', 'know', 'think', 'look', 'conclusion',
                              'summarise', 'summarize', 'finale', 'overall', 'sum', 'end', 'summary', 'conclude']
    # Combine top_words and synonyms into one list
    all_words = list(set(top_words))

    # Create a TF-IDF vectorizer with the top 10 words as vocabulary
    vectorizer = TfidfVectorizer(vocabulary=all_words)

    total_rows_processed = 0  # total rows processed in this session

    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
        feature_list = []

        # Skip chunks that have already been processed
        if total_rows_processed < processed_rows:
            total_rows_processed += len(chunk)
            continue

        data = list(chunk.itertuples(index=False, name=None))
        texts, labels = remove_prefix(data)
        prompts_and_texts = extract_prompts_and_texts(data)

        for i, ((prompt, text), label) in enumerate(zip(prompts_and_texts, labels)):
            try:
                features = {}  # Initialize the features dictionary here

                # Count POS tags in the text
                pos_counts, punctuation_counts, function_word_counts = count_pos_tags_and_special_elements(text)

                # Calculate the Flesch Reading Ease and Flesch-Kincaid Grade Level
                flesch_reading_ease, flesch_kincaid_grade_level = calculate_readability_scores(text)

                # Calculate the average word length
                avg_word_length = calculate_average_word_length([text])

                # Calculate the average sentence length
                avg_sentence_length = calculate_average_sentence_length([text])

                # Transform the text into TF-IDF scores
                tfidf_scores = vectorizer.fit_transform([text]).toarray()

                # Calculate the perplexity of the text and average sentence perplexity
                text_encoded = tokenizer.encode(text, truncation=True, max_length=510)
                text = tokenizer.decode(text_encoded)
                text = text.replace('<s>', '').replace('</s>', '')
                text_perplexity = calculate_perplexity(text, model, tokenizer)
                sentence_perplexities = [calculate_perplexity(sentence.text, model, tokenizer) for sentence in
                                         nlp(text).sents]
                sentence_perplexities = [p for p in sentence_perplexities if p is not None]
                avg_sentence_perplexity = sum(sentence_perplexities) / len(
                    sentence_perplexities) if sentence_perplexities else None

                # Calculate the frequency of uppercase letters
                uppercase_freq = sum(1 for char in text if char.isupper()) / len(text)

                # Calculate the cosine similarity for the prompt and text
                prompt_text_cosine_similarity = calculate_cosine_similarity(prompt, text, model, tokenizer)

                # Calculate the average cosine similarity for sentences in the text
                sentence_cosine_similarities = calculate_cosine_similarities_for_sentences_in_text(text, model,
                                                                                                   tokenizer)
                avg_sentence_cosine_similarity = None
                if sentence_cosine_similarities:
                    avg_sentence_cosine_similarity = sum(sentence_cosine_similarities) / len(
                        sentence_cosine_similarities)
                else:
                    print("WARNING: No sentence cosine similarities calculated for text:", text)

                # Prepare a dictionary to append to the feature list
                features.update({
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
                })

                # If the TF-IDF scores array is not empty, zip the scores with the words to create a dictionary
                # and update the features dictionary with this new dictionary
                if tfidf_scores.size > 0:
                    word_scores = {f"tf_idfs_{word}": score for word, score in zip(all_words, tfidf_scores[0])}
                    features.update(word_scores)
                else:  # If the TF-IDF scores array is empty, assign 0 to each word's score
                    word_scores = {f"tf_idfs_{word}": 0 for word in all_words}
                    features.update(word_scores)

                features['label'] = label

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


def prepare_single_text_for_regression(input_text, prompt):
    """
    This function prepares a single text for regression analysis by extracting features.

    Args:
    input_text (str): The input text string.
    prompt (str): The prompt string.

    Returns:
    features (dict): A dictionary where each key-value pair represents a feature and its value.
    """

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Combine top_words and synonyms into one list
    all_words = ['said', 'like', 'im', 'get', 'told', 'dont', 'say', 'know', 'think', 'look', 'conclusion', 'summarise',
                 'summarize', 'finale', 'overall', 'sum', 'end', 'summary', 'conclude']

    # Create a TF-IDF vectorizer with the top 10 words as vocabulary
    vectorizer = TfidfVectorizer(vocabulary=all_words)

    # Initialize the features dictionary here
    features = {}

    # Count POS tags in the text
    pos_counts, punctuation_counts, function_word_counts = count_pos_tags_and_special_elements(input_text)

    # Calculate the Flesch Reading Ease and Flesch-Kincaid Grade Level
    flesch_reading_ease, flesch_kincaid_grade_level = calculate_readability_scores(input_text)

    # Calculate the average word length
    avg_word_length = calculate_average_word_length([input_text])

    # Calculate the average sentence length
    avg_sentence_length = calculate_average_sentence_length([input_text])

    # Calculate cosine similarity between the prompt and the text
    prompt_text_cosine_similarity = calculate_cosine_similarity_for_prompt_and_text(prompt, input_text, model,
                                                                                    tokenizer)

    # Calculate cosine similarities for each sentence in the text
    sentence_cosine_similarities = calculate_cosine_similarities_for_sentences_in_text(input_text, model, tokenizer)

    avg_sentence_cosine_similarity = None
    if sentence_cosine_similarities:
        avg_sentence_cosine_similarity = sum(sentence_cosine_similarities) / len(sentence_cosine_similarities)
    else:
        print("WARNING: No sentence cosine similarities calculated for text:", text)

    # Transform the text into TF-IDF scores
    tfidf_scores = vectorizer.fit_transform([input_text]).toarray()

    # Calculate the perplexity of the text and average sentence perplexity
    text_encoded = tokenizer.encode(input_text, truncation=True, max_length=510)
    text = tokenizer.decode(text_encoded)
    text = text.replace('<s>', '').replace('</s>', '')
    text_perplexity = calculate_perplexity(text, model, tokenizer)
    sentence_perplexities = [calculate_perplexity(sentence.text, model, tokenizer) for sentence in nlp(text).sents]
    sentence_perplexities = [p for p in sentence_perplexities if p is not None]
    avg_sentence_perplexity = sum(sentence_perplexities) / len(sentence_perplexities) if sentence_perplexities else None

    # Calculate the frequency of uppercase letters
    uppercase_freq = sum(1 for char in input_text if char.isupper()) / len(input_text)

    # Prepare a dictionary to append to the feature list
    features.update({
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
    })

    # If the TF-IDF scores array is not empty, zip the scores with the words to create a dictionary
    # and update the features dictionary with this new dictionary
    if tfidf_scores.size > 0:
        word_scores = {f"tf_idfs_{word}": score for word, score in zip(all_words, tfidf_scores[0])}
        features.update(word_scores)
    else:  # If the TF-IDF scores array is empty, assign 0 to each word's score
        word_scores = {f"tf_idfs_{word}": 0 for word in all_words}
        features.update(word_scores)

    return features
