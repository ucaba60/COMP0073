# Imports
import datasets
import re
import pandas as pd
import os

# Constants
DATASETS = ['pubmed_qa', 'writingprompts', 'cnn_dailymail', 'gpt']
DATA_PATH = './data/writingPrompts'  # This is required to load the writingPrompts dataset, as it is not part of the
# 'datasets' library,
NUM_EXAMPLES = 300  # Number of initial samples from each dataset, note below, the actual number of samples is ~825
# due to filtering
TAGS = ['[ WP ]', '[ OT ]', '[ IP ]', '[ HP ]', '[ TT ]', '[ Punch ]', '[ FF ]', '[ CW ]', '[ EU ]', '[ CC ]', '[ RF ]',
        '[ wp ]', '[ Wp ]', '[ RF ]', '[ WP/MP ]']


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
    if not file_name.endswith('.csv'):
        file_name += '.csv'

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


def load_data(dataset_name, gpt_filename=None):
    """
       Loads a dataset based on its name.

       Args:
           dataset_name (str): Name of the dataset to load.
           gpt_filename (str, optional): Name of the csv file containing the GPT dataset.

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
        if gpt_filename is None:
            raise ValueError("A filename must be provided to load the GPT dataset.")
        return load_gpt(gpt_filename)
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

    # Getting long-enough data, not done for PubMed due to most of the responses being fairly short.
    # This is consistent with most research approaches concering these datasets (DetectGPT paper e.g.)
    if dataset == 'writingprompts' or dataset == 'cnn_dailymail':
        long_data = [(x, y) for x, y in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
        print(f"Loaded and pre-processed {len(data)} entries from the dataset {dataset}")  # debug
        # print
    else:
        print(f"Loaded and pre-processed {len(data)} entries from the dataset {dataset}")

    return data


def preprocess_and_save(gpt_dataset=None, gpt_dataset_path=None, output_folder='extracted_data'):
    """
    Preprocesses the datasets, combines them, and saves the result to a .csv file.
    Optional argument gpt_dataset allows preprocessing the GPT dataset and combining it with existing datasets.

    Args:
        gpt_dataset (str, optional): Name of the GPT dataset csv file (without the .csv extension).
        gpt_dataset_path (str, optional): Path to the GPT dataset.
        output_folder: folder where the extracted data will be saved

    Returns:
        None, saves the combined data to a .csv file.
    """

    os.makedirs(output_folder, exist_ok=True)

    if gpt_dataset:
        # Load and preprocess the GPT dataset
        gpt_data_path = os.path.join(gpt_dataset_path, gpt_dataset)
        gpt_data = load_data('gpt', gpt_data_path)
        gpt_data = list(dict.fromkeys(gpt_data))
        gpt_data = [(strip_newlines(q).strip(), a) for q, a in gpt_data]

        # Load the already preprocessed data from the other datasets
        combined_df = pd.read_csv(os.path.join(output_folder, 'combined_human_data.csv'))
        combined_data = list(zip(combined_df['Text'], combined_df['Label']))

        # Combine the data
        combined_data += gpt_data

        model_name = gpt_dataset.split('_')[0]  # Extract model name from gpt_dataset

        output_file = f'{model_name}_and_human_data.csv'

    else:
        # Preprocess all the datasets
        pubmed_data = preprocess_data('pubmed_qa')
        writingprompts_data = preprocess_data('writingprompts')
        cnn_daily_mail_data = preprocess_data('cnn_dailymail')

        combined_data = pubmed_data + writingprompts_data + cnn_daily_mail_data

        output_file = 'combined_human_data.csv'

    output_file_path = os.path.join(output_folder, output_file)

    if os.path.exists(output_file_path):
        overwrite = input(f"'{output_file_path}' already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print(f"Not overwriting existing file '{output_file_path}'. Exiting...")
            return

    # Save the combined data to a .csv file
    df = pd.DataFrame(combined_data, columns=['Text', 'Label'])
    df.to_csv(output_file_path, index=False)

    print(f"Combined dataset saved to '{output_file_path}' with {len(combined_data)} entries.")


# preprocess_and_save(output_folder = 'extracted_data')

def extract_prompts_and_save(file_folder_path):
    """
    Extracts prompts from the combined dataset and saves them to a .csv file.

    Args:
        file_folder_path (str): The path to the folder where the combined_source_data.csv file is located.

    Returns:
        None, saves the prompts to a .csv file.
    """
    # Load the combined dataset
    combined_data_file = os.path.join(file_folder_path, 'combined_human_data.csv')
    df = pd.read_csv(combined_data_file)
    combined_data = list(zip(df['Text'], df['Label']))

    # Extract prompts from the combined data
    prompts = []
    for i, (full_text, _) in enumerate(combined_data):
        if i < 300:
            prompt = full_text.replace('Answer:', 'Write an abstract for a scientific paper that answers the Question:')
            prompt = prompt.split('Write an abstract for a scientific paper that answers the Question:')[0] + \
                     'Write an abstract for a scientific paper that answers the Question:'
            prompts.append(prompt.strip())
        elif 'Summary:' in full_text and 'Article:' in full_text:
            prompts.append('Write a news article based on the following summary: ' +
                           full_text.split('Summary:')[1].split('Article:')[0].strip())
        elif 'Prompt:' in full_text and 'Story:' in full_text:
            prompts.append(full_text.replace('Prompt:', '').split('Story:')[0].strip() + ' Continue the story:')
        else:
            print(f"Could not determine dataset for the entry: {full_text}")

    # Save the prompts to a new CSV file
    df_prompts = pd.DataFrame(prompts, columns=['Prompt'])
    df_prompts.to_csv(os.path.join(file_folder_path, 'prompts.csv'), index=False)
    print(f"Prompts extracted and saved to '{os.path.join(file_folder_path, 'prompts.csv')}' with {len(df_prompts)}"
          f" entries.")

# extract_prompts_and_save("extracted_data")
