import datasets
import re
import pandas as pd
import os
import random
import tiktoken
import argparse
import glob

# Constants
DATASETS = ['pubmed_qa', 'writingprompts', 'cnn_dailymail', 'gpt']
DATA_PATH = './data/writingPrompts'
NUM_EXAMPLES = 200
TAGS = ['[ WP ]', '[ OT ]', '[ IP ]', '[ HP ]', '[ TT ]', '[ Punch ]', '[ FF ]', '[ CW ]', '[ EU ]', '[ CC ]', '[ RF ]',
        '[ wp ]', '[ Wp ]', '[ RF ]', '[ WP/MP ]']
directory = 'Dataset_Data/'


def strip_newlines(text):
    """
    Removes newline characters from a string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with newline characters removed.
    """
    return ' '.join(text.split())


def replace_text(text, replacements):
    """
    Performs a series of replacements in a string.

    Args:
        text (str): Input text string.
        replacements (dict): Dictionary mapping old substring to new substring.

    Returns:
        str: Text with specified replacements made.
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_whitespace_before_punctuations(text):
    """
    Removes whitespace before punctuation marks in a string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with whitespace removed before punctuation marks.
    """
    return re.sub(r'\s([?.!,:;](?:\s|$))', r'\1', text)


def load_pubmed(num_examples=NUM_EXAMPLES):
    """
    Loads the PubMed QA dataset.

    Args:
        num_examples (int, optional): Number of examples to load. Defaults to NUM_EXAMPLES.

    Returns:
        list: List of tuples where each tuple is a question-answer pair and a label (always 0).
    """
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split=f'train[:{num_examples}]')
    data = [(f'Question: {q} Answer: {a}', 0) for q, a in zip(data['question'], data['long_answer'])]
    return data


def load_gpt(file_name):
    """
    Loads the GPT preprocessed dataset.

    Args:
        file_name (str): Name of the csv file containing the GPT dataset.

    Returns:
        list: List of tuples where each tuple is a text-label pair.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")

    df = pd.read_csv(file_name)
    data = [(row['Text'], row['Label']) for index, row in df.iterrows()]

    return data



def load_writingPrompts(data_path=DATA_PATH, num_examples=NUM_EXAMPLES):
    """
    Loads the WritingPrompts dataset. Combines Prompts and Stories with additional formatting.

    Args:
        data_path (str, optional): Path to the dataset. Defaults to DATA_PATH.
        num_examples (int, optional): Number of examples to load. Defaults to NUM_EXAMPLES.

    Returns:
        list: List of tuples where each tuple is a prompt-story pair and a label (always 0).
    """
    with open(f'{data_path}/valid.wp_source', 'r', encoding='utf-8') as f:
        prompts = f.readlines()[:num_examples]
    with open(f'{data_path}/valid.wp_target', 'r', encoding='utf-8') as f:
        stories = f.readlines()[:num_examples]

    prompt_replacements = {tag: '' for tag in TAGS}
    prompts = [replace_text(prompt, prompt_replacements) for prompt in prompts]
    prompts = [remove_whitespace_before_punctuations(prompt) for prompt in prompts]

    story_replacements = {
        ' ,': ',',
        ' .': '.',
        ' ?': '?',
        ' !': '!',
        ' ;': ';',
        ' \'': '\'',
        ' ’ ': '\'',
        ' :': ':',
        '<newline>': '\n',
        '`` ': '"',
        ' \'\'': '"',
        '\'\'': '"',
        '.. ': '... ',
        ' )': ')',
        '( ': '(',
        ' n\'t': 'n\'t',
        ' i ': ' I ',
        ' i\'': ' I\'',
        '\\\'': '\'',
        '\n ': '\n',
    }
    stories = [replace_text(story, story_replacements).strip() for story in stories]
    joined = ["Prompt:" + prompt + " Story: " + story for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story.lower()]
    data = [(story, 0) for story in filtered]
    return data


def load_cnn_daily_mail(num_examples=NUM_EXAMPLES):
    """
    Loads the CNN/Daily Mail dataset. Combines article and summary with additional formatting.

    Args:
        num_examples (int, optional): Number of examples to load. Defaults to NUM_EXAMPLES.

    Returns:
        list: List of tuples where each tuple is a summary-article pair and a label (always 0).
    """
    data = datasets.load_dataset('cnn_dailymail', '3.0.0', split=f'train[:{num_examples}]')

    processed_data = []
    for a, s in zip(data['article'], data['highlights']):
        # remove the string and the '--' from the start of the articles
        a = re.sub('^[^-]*--', '', a).strip()

        # remove the string 'E-mail to a friend.' from the articles, if present
        a = a.replace('E-mail to a friend .', '')
        s = s.replace('NEW:', '')
        a = a.replace(
            'Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, '
            'or redistributed.',
            '')

        # remove whitespace before punctuation marks in both article and summary
        a = remove_whitespace_before_punctuations(a)
        s = remove_whitespace_before_punctuations(s)

        processed_data.append((f'Summary: {s} Article: {a}', 0))
        data = processed_data

    return data


def load_data(dataset_name):
    """
       Loads a dataset based on its name.

       Args:
           dataset_name (str): Name of the dataset to load.

       Returns:
           list: List of data from the specified dataset.

       Raises:
           ValueError: If the dataset_name is not recognized.
    """
    if dataset_name == 'pubmed_qa':
        return load_pubmed()
    elif dataset_name == 'writingprompts':
        return load_writingPrompts()
    elif dataset_name == 'cnn_dailymail':
        return load_cnn_daily_mail()
    elif dataset_name == 'gpt':
        return load_gpt()
    else:
        raise ValueError(f"Dataset name {dataset_name} not recognized.")


def preprocess_data(dataset):
    """
        Preprocesses a dataset.

        Args:
            dataset (str): Name of the dataset to preprocess.

        Returns:
            list: List of preprocessed data from the specified dataset.

        Raises:
            ValueError: If the dataset_name is not recognized.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Dataset name {dataset} not recognized.")

    data = load_data(dataset)
    data = list(dict.fromkeys(data))
    data = [(strip_newlines(q).strip(), a) for q, a in data]

    # Getting long-enough data, not done for PubMed due to most of respones being fairly short.
    if dataset == 'writingprompts' or dataset == 'cnn_dailymail':
        long_data = [(x, y) for x, y in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
        print(f"Loaded and pre-processed {len(data)} entries from the dataset {dataset}")  # debug
        # print
    else:
        print(f"Loaded and pre-processed {len(data)} entries from the dataset {dataset}")

    return data


def preprocess_and_save(gpt_dataset=None):
    """
    Preprocesses the datasets, combines them, and saves the result to a .csv file.
    Optional argument gpt_dataset allows preprocessing the GPT dataset and combining it with existing datasets.

    Args:
        gpt_dataset (str, optional): Name of the GPT dataset csv file (without the .csv extension).

    Returns:
        None, saves the combined data to a .csv file.
    """
    if gpt_dataset:
        # Load and preprocess the GPT dataset
        gpt_data = load_data('gpt')
        gpt_data = list(dict.fromkeys(gpt_data))
        gpt_data = [(strip_newlines(q).strip(), a) for q, a in gpt_data]

        # Load the already preprocessed data from the other datasets
        combined_df = pd.read_csv('combined_source_data.csv')
        combined_data = list(zip(combined_df['Text'], combined_df['Label']))

        # Combine the data
        combined_data += gpt_data

        output_file = 'full_data.csv'
        print_message = f"Combined dataset (including GPT) saved to '{output_file}' with {len(combined_data)} entries."

    else:
        # Preprocess all the datasets
        pubmed_data = preprocess_data('pubmed_qa')
        writingprompts_data = preprocess_data('writingprompts')
        cnn_daily_mail_data = preprocess_data('cnn_dailymail')

        combined_data = pubmed_data + writingprompts_data + cnn_daily_mail_data

        output_file = 'combined_source_data.csv'
        print_message = f"Combined dataset saved to '{output_file}' with {len(combined_data)} entries."

    # Save the combined data to a .csv file
    df = pd.DataFrame(combined_data, columns=['Text', 'Label'])
    df.to_csv(output_file, index=False)

    print(print_message)



preprocess_and_save("t1_responses_preprocessed")



