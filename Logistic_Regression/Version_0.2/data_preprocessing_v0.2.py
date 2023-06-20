import datasets
import re
import pandas as pd
import os

# Constants
DATASETS = ['pubmed_qa', 'writingprompts', 'cnn_dailymail']
DATA_PATH = 'data/writingPrompts'
NUM_EXAMPLES = 150
TAGS = ['[ WP ]', '[ OT ]', '[ IP ]', '[ HP ]', '[ TT ]', '[ Punch ]', '[ FF ]', '[ CW ]', '[ EU ]']


def strip_newlines(text):
    """
    Removes newline characters from a string.

    Args:
        text (str): Input text string.

    Returns:
        str: Text with newline characters removed.
    """
    return ' '.join(text.split())


def process_text(text, replacements):
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


def load_writingPrompts(data_path=DATA_PATH, num_examples=NUM_EXAMPLES):
    """
    Loads the WritingPrompts dataset.

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
    prompts = [process_text(prompt, prompt_replacements) for prompt in prompts]
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
    stories = [process_text(story, story_replacements).strip() for story in stories]
    joined = ["Prompt:" + prompt + " Story: " + story for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story.lower()]
    data = [(story, 0) for story in filtered]
    return data


def load_cnn_daily_mail(num_examples=NUM_EXAMPLES):
    """
    Loads the CNN/Daily Mail dataset.

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
    if dataset == 'pubmed_qa':
        print(f"Loaded and pre-processed {len(data)} questions from the dataset")  # debug print

    # Getting long-enough prompts, can do the same for the articles as well
    if dataset == 'writingprompts' or dataset == 'cnn_dailymail':
        long_data = [(x, y) for x, y in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
        print(f"Loaded and pre-processed {len(data)} prompts/stories[summaries/articles] from the dataset")  # debug
        # print

    return data


def convert_to_csv(data, dataset_name, directory='Labelled_Data'):
    """
        Converts the data to a DataFrame and saves it to a CSV file in the specified directory.

        Args:
            data (list): List of data to be converted to CSV.
            dataset_name (str): Name of the dataset.
            directory (str, optional): Name of the directory to save the CSV file. Defaults to 'Labelled_Data'.

        Returns:
            None
    """
    # Check if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    # Write DataFrame to CSVv
    df.to_csv(f'{directory}/{dataset_name}_Human_data.csv', index=False)
