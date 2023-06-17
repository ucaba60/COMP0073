import datasets
import re

DATASETS = ['pubmed_qa', 'writingprompts']


def strip_newlines(text):
    return ' '.join(text.split())


def load_pubmed():
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train[:150]')

    # combine question and long_answer, and label them as 0
    data = [(f'Question: {q} Answer:{a}', 0) for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    tags = ['[ WP ]', '[ OT ]', '[ IP ]', '[ HP ]', '[ TT ]', '[ Punch ]', '[ FF ]', '[ CW ]', '[ EU ]']
    for tag in tags:
        prompt = prompt.replace(tag, '')
    return prompt

def remove_whitespace_before_punctuations(text):
    text = re.sub(r'\s([?.!,:;](?:\s|$))', r'\1', text)
    return text


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writingPrompts_dataset():
    writing_path = 'data/writingPrompts'

    with open(f'{writing_path}/valid.wp_source', 'r', encoding='utf-8') as f:
        prompts = f.readlines()[:178]
    with open(f'{writing_path}/valid.wp_target', 'r', encoding='utf-8') as f:
        stories = f.readlines()[:178]

    prompts = [process_prompt(prompt) for prompt in prompts]
    prompts = [remove_whitespace_before_punctuations(prompt) for prompt in prompts]
    prompts = [prompt.rstrip() for prompt in prompts]
    stories = [process_spaces(story) for story in stories]
    joined = ["Prompt: " + prompt + " Story: " + story for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    # Label the stories as 0 to indicate they are human-generated
    data = [(story, 0) for story in filtered]

    return data


def load_data(dataset_name):
    if dataset_name == 'pubmed_qa':
        return load_pubmed()
    elif dataset_name == 'writingprompts':
        return load_writingPrompts_dataset()
    else:
        print(f"Dataset name {dataset_name} not recognized.")
        return None


def preprocess_data(dataset):
    if dataset in DATASETS:
        data = load_data(dataset)

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [(x[0].strip(), x[1]) for x in data]

    # remove newlines from each example
    data = [(strip_newlines(q), a) for q, a in data]

    # try to keep only examples with > 250 words
    if dataset == 'pubmed_qa':
        print(f"Loaded and pre-processed {len(data)} questions from the dataset")  # debug print

    if dataset == 'writingprompts':
        long_data = [(x, y) for x, y in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
        print(f"Loaded and pre-processed {len(data)} prompts/stories from the dataset")  # debug print

    return data
