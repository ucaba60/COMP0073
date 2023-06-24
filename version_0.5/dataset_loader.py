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
directory = 'Labelled_Data/'


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

    # Check if the file already exists
    file_path = f'{directory}/{dataset_name}_Human_data.csv'
    if os.path.exists(file_path):
        overwrite = input(f"A file named '{file_path}' already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Data not saved to avoid overwriting the existing file.")
            return

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    # Write DataFrame to CSV
    df.to_csv(file_path, index=False)


def combine_datasets(datasets=DATASETS, extract_prompts=False, directory='Labelled_Data'):
    """
    Combines data from multiple datasets into a single dataset. If specified, extracts prompts based on dataset names,
    and saves the result to a CSV file.

    Args:
        directory: Where the file will be saved
        datasets (list, optional): List of datasets to combine. Defaults to DATASETS.
        extract_prompts (bool, optional): Whether to extract prompts from the combined data. Defaults to False.

    Returns:
        None
    """
    # Initialize a list to store the combined data
    combined_data = []

    # If specified, also store the extracted prompts
    extracted_prompts = [] if extract_prompts else None

    # Load and preprocess data from each dataset
    for dataset in datasets:
        data = preprocess_data(dataset)
        combined_data.extend(data)

        # If specified, extract prompts
        if extract_prompts:
            extracted_prompts.extend(extract_prompt(data, dataset))

    # Shuffle the combined data to ensure a mix of data from all datasets
    # random.shuffle(combined_data)
    # random.shuffle(extracted_prompts) if extract_prompts else None

    # Save the combined data to a CSV file
    convert_to_csv(combined_data, 'combined_human')

    # If specified, save the extracted prompts to a CSV file
    if extract_prompts:
        df = pd.DataFrame(extracted_prompts, columns=['text'])
        df.to_csv(f'{directory}/prompts.csv', index=False)


def extract_prompt(data, dataset_name):
    """
    Extracts the prompts from a preprocessed dataset.

    Args:
        data (list): Preprocessed data.
        dataset_name (str): Name of the dataset the data is from.

    Returns:
        list: List of extracted prompts.
    """
    prompts = []
    if dataset_name == 'pubmed_qa':
        prompts = [text.split('Answer:')[0] + 'Answer:' for text, label in data]
    elif dataset_name == 'cnn_dailymail':
        # Split the text into article and summary, then only append the summary
        prompts = [
            'Write a news article based on the following summary: ' + text.split('Summary:')[1].split('Article:')[
                0].strip() for text, label in data]
    elif dataset_name == 'writingprompts':
        prompts = [text.replace('Prompt:', '').split('Story:')[0].strip() + ' Continue the story:' for text, label in
                   data]
    return prompts


def token_count(csv_files):
    """
    Counts the number of tokens in a CSV file.

    Args:
        csv_files (str): Path to the CSV file.

    Returns:
        None
    """

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    for csv_file in csv_files:
        # Load prompts from CSV file
        df = pd.read_csv(csv_file)
        prompts = df['text'].tolist()

        # Initialize a counter for total tokens
        total_tokens = 0

        for prompt in prompts:
            num_tokens = len(encoding.encode(prompt))
            total_tokens += num_tokens

        print(f"File '{csv_file}' has {total_tokens} tokens.")

        # Estimate cost
        if csv_file == 'Labelled_Data/prompts.csv':
            cost = (total_tokens / 1000) * 0.003
            print(f"Estimated cost for '{csv_file}' is ${cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Script for loading and preprocessing datasets.')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='List of datasets to load and preprocess. Default is all datasets.')
    parser.add_argument('--prompts', action='store_true', help='Extract prompts from preprocessed datasets.')
    parser.add_argument('--tokens', action='store_true', help='Count tokens in preprocessed datasets.')
    parser.add_argument('--combined', action='store_true', help='Combine all datasets.')
    args = parser.parse_args()

    datasets = args.datasets
    combined_data = []
    for dataset in datasets:
        data = preprocess_data(dataset)  # preprocessing is always done
        combined_data.extend(data)
        convert_to_csv(data, dataset)
        if args.prompts:
            prompts = extract_prompt(data, dataset)
            df = pd.DataFrame(prompts, columns=['text'])
            df.to_csv(f'{directory}/prompts_{dataset}.csv', index=False)
        if args.tokens:
            csv_files = glob.glob('Labelled_Data/*.csv')
            token_count(csv_files)
    if args.combined:
        convert_to_csv(combined_data, 'combined')
        if args.prompts:
            combined_prompts = []
            for dataset in datasets:
                combined_prompts.extend(extract_prompt(load_data(dataset), dataset))
            df = pd.DataFrame(combined_prompts, columns=['text'])
            df.to_csv(f'{directory}/prompts_combined.csv', index=False)

    print("Script finished running successfully. Check your files.")


if __name__ == "__main__":
    main()
