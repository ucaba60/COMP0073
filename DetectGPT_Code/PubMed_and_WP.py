##COMMAS AND FULLSTOPS IN WP ARE NOT CONSISTENT, SO THEY ARE NOT A GOOD INDICATOR
import os
import spacy
from collections import Counter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
import random

# Constants
DATASET = 'pubmed_qa'
SPLIT = 'pqa_labeled'
TRAIN_SPLIT = 'train'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
CHUNK_SIZE = 512


class TextAnalysis:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.model = BertForMaskedLM.from_pretrained(MODEL_NAME)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        self.pos_counts = Counter()
        self.punctuation_counts = Counter()
        self.function_word_counts = Counter()
        self.total_tokens = 0
        self.total_sentences = 0
        self.sentence_perplexities = []
        self.answer_perplexities = []

    def calculate_perplexity(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')

        # Ensure the input_ids tensor has a maximum length of 512.
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, :512]

        # We're going to compute the loss over all tokens, so don't ignore any tokens
        no_ignore_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=(input_ids != self.tokenizer.pad_token_id), labels=input_ids,
                                 return_dict=True)

        # take the mean of the losses over all tokens (but you could also sum them, depending on what you prefer)
        perplexity = torch.exp(outputs.loss)

        return perplexity.item()

    def strip_newlines(self, text):
        return text.replace('\n', ' ')

    def preprocess_data(self, data):
        # strip whitespace around each example
        data = [(q.strip(), a.strip()) for q, a in data]

        # remove newlines from each example
        data = [(self.strip_newlines(q), self.strip_newlines(a)) for q, a in data]

        # remove duplicates from the data
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

        # random.seed(0)
        # random.shuffle(data)

        return data

    def load_pubmed(self):
        raw_data = load_dataset(DATASET, SPLIT, split=TRAIN_SPLIT)
        data = [(q, a) for q, a in zip(raw_data['question'], raw_data['long_answer']) if
                len(self.tokenizer(a)['input_ids']) <= CHUNK_SIZE]
        data = self.preprocess_data(data)

        print(f"Loaded {len(raw_data)} questions from the dataset")  # debug print
        print(f"Filtered down to {len(data)} questions with answers longer than {CHUNK_SIZE} tokens")  # debug print
        return data
    def process_prompt(self, prompt):
        return prompt.replace('[ WP ]', '').replace('[ OT ]', '').replace('[ IP ]', '').replace('[ HP ]', '').replace('[ TT ]', '').replace('[ Punch ]', '').replace('[ FF ]', '').replace('[ CW ]', '').replace('[ EU ]', '')

    def process_spaces(self, story):
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

    def load_writingPrompts_dataset(self):
        writing_path = 'data/writingPrompts'

        with open(f'{writing_path}/valid.wp_source', 'r', encoding='utf-8') as f:
            prompts = f.readlines()
        with open(f'{writing_path}/valid.wp_target', 'r', encoding='utf-8') as f:
            stories = f.readlines()

        prompts = [self.process_prompt(prompt) for prompt in prompts]
        stories = [self.process_spaces(story) for story in stories]
        joined = [prompt + " " + story for prompt, story in zip(prompts, stories)]
        filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

        long_data = [x for x in filtered if len(self.tokenizer(x)['input_ids']) <= CHUNK_SIZE]
        if len(long_data) > 0:
            filtered = long_data

        # Create a list of tuples where each tuple is a pair of a prompt and a corresponding story
        data = [(prompt.strip(), story.strip()) for prompt, story in zip(prompts, stories)]
        data = list(dict.fromkeys(data))  # Remove duplicate entries

        print(f"Loaded {len(data)} prompts and stories from the writingPrompts dataset")
        return data

    def process_data(self, data, source):
        print(f"Processing {len(data)} items")  # debug print
        for item1, item2 in data:
            if source == 'pubmed':
                question = item1
                answer = item2
                print(f"Processing question: {question[:100]}...")  # debug print
                answer_doc = self.nlp(answer)
                self.total_tokens += len(answer_doc)
                self.total_sentences += len(list(answer_doc.sents))
                self.process_each_token(answer_doc)
                self.process_each_sentence(answer_doc)

                answer_perplexity = self.calculate_perplexity(answer)  # calculate perplexity for entire answer
                if answer_perplexity < 0:
                    print(f"Negative perplexity found: {answer_perplexity} for answer: {answer}")
                self.answer_perplexities.append(answer_perplexity)  # store the answer's perplexity

            elif source == 'writingPrompts':
                prompt = item1
                story = item2
                print(f"Processing prompt: {prompt[:100]}...")  # debug print
                story_doc = self.nlp(story)
                self.total_tokens += len(story_doc)
                self.total_sentences += len(list(story_doc.sents))
                self.process_each_token(story_doc)
                self.process_each_sentence(story_doc)

                story_perplexity = self.calculate_perplexity(story)  # calculate perplexity for entire story
                if story_perplexity < 0:
                    print(f"Negative perplexity found: {story_perplexity} for story: {story}")
                self.answer_perplexities.append(story_perplexity)  # store the story's perplexity
    def process_each_sentence(self, answer_doc):
        for sent in answer_doc.sents:
            sentence_perplexity = self.calculate_perplexity(sent.text)
            if sentence_perplexity < 0:
                print(f"Negative perplexity found: {sentence_perplexity} for sentence: {sent.text}")
            self.sentence_perplexities.append(sentence_perplexity)

    def process_each_token(self, answer_doc):
        for token in answer_doc:
            self.pos_counts[token.pos_] += 1
            if token.is_punct:
                self.punctuation_counts[token.text] += 1
            if token.text in {'a', 'in', 'of', 'the'}:
                self.function_word_counts[token.text] += 1

    def calculate_averages(self):
        avg_sentence_length = self.total_tokens / self.total_sentences if self.total_sentences > 0 else 0
        avg_sentence_perplexity = sum(self.sentence_perplexities) / len(
            self.sentence_perplexities) if self.sentence_perplexities else 0
        avg_answer_perplexity = sum(self.answer_perplexities) / len(
            self.answer_perplexities) if self.answer_perplexities else 0
        return avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity

    def print_results(self, avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity):
        print(f"Frequency of adjectives: {self.pos_counts['ADJ']}")
        print(f"Frequency of adverbs: {self.pos_counts['ADV']}")
        print(f"Frequency of conjunctions: {self.pos_counts['CCONJ']}")
        print(f"Frequency of nouns: {self.pos_counts['NOUN']}")
        print(f"Frequency of numbers: {self.pos_counts['NUM']}")
        print(f"Frequency of pronouns: {self.pos_counts['PRON']}")
        print(f"Frequency of verbs: {self.pos_counts['VERB']}")
        print(f"Frequency of commas: {self.punctuation_counts[',']}")
        print(f"Frequency of fullstops: {self.punctuation_counts['.']}")
        print(f"Frequency of special character '-': {self.punctuation_counts['-']}")
        print(f"Frequency of function word 'a': {self.function_word_counts['a']}")
        print(f"Frequency of function word 'in': {self.function_word_counts['in']}")
        print(f"Frequency of function word 'of': {self.function_word_counts['of']}")
        print(f"Frequency of function word 'the': {self.function_word_counts['the']}")
        print(f"Average sentence length: {avg_sentence_length}")
        print(f"Average sentence perplexity: {avg_sentence_perplexity}")
        print(f"Average answer perplexity: {avg_answer_perplexity}")

    def plot_perplexities(self):
        sns.set_style('white')
        fig, axs = plt.subplots(2)
        sns.kdeplot(self.sentence_perplexities, color='skyblue', fill=True, ax=axs[0])
        axs[0].set_title('Density Plot of Sentence Perplexities')
        axs[0].set_xlabel('Perplexity')
        axs[0].set_ylabel('Density')

        sns.kdeplot(self.answer_perplexities, color='skyblue', fill=True, ax=axs[1])
        axs[1].set_title('Density Plot of Answer Perplexities')
        axs[1].set_xlabel('Perplexity')
        axs[1].set_ylabel('Density')

        for ax in axs:
            ax.set_xlim(0, 10)
            ax.set_xticks(range(0, 11))
            sns.despine()
        plt.tight_layout()
        plt.show()

    def reset_analysis_variables(self):
        self.pos_counts = Counter()
        self.punctuation_counts = Counter()
        self.function_word_counts = Counter()
        self.total_tokens = 0
        self.total_sentences = 0
        self.sentence_perplexities = []
        self.answer_perplexities = []

    def run(self):
        # print("\n===== Analyzing PubMed Dataset =====")
        # pubmed_data = self.load_pubmed()
        # self.process_data(pubmed_data, 'pubmed')
        # avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity = self.calculate_averages()
        # self.print_results(avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity)
        # self.plot_perplexities()
        #
        # # Resetting variables for the new analysis
        # self.reset_analysis_variables()

        print("\n===== Analyzing Writing Prompts Dataset =====")
        writingPrompts_data = self.load_writingPrompts_dataset()

        # Limit to the first 1000 prompts
        writingPrompts_data = writingPrompts_data[:1000]

        self.process_data(writingPrompts_data, 'writingPrompts')
        avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity = self.calculate_averages()
        self.print_results(avg_sentence_length, avg_sentence_perplexity, avg_answer_perplexity)
        self.plot_perplexities()


if __name__ == "__main__":
    analysis = TextAnalysis()
    analysis.run()