import spacy
from collections import Counter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

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

    def calculate_perplexity(self, text):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt')
        inputs['labels'] = inputs['input_ids'].detach().clone()

        # create mask
        mask = torch.cat([torch.tensor([1]), torch.ones(inputs['input_ids'].shape[1] - 2), torch.tensor([1])]).bool()
        inputs['input_ids'].masked_fill_(mask, self.tokenizer.mask_token_id)

        with torch.no_grad():
            loss = self.model(**inputs).loss
        return torch.exp(loss).item()

    def load_dataset(self):
        data = load_dataset(DATASET, SPLIT, split=TRAIN_SPLIT)
        filtered_data = [(q, (self.nlp(a), 0)) for q, a in zip(data['question'], data['long_answer']) if
                         len(self.tokenizer(a)['input_ids']) <= CHUNK_SIZE]
        print(f"Loaded {len(data)} questions from the dataset")  # debug print
        print(
            f"Filtered down to {len(filtered_data)} questions with answers longer than {CHUNK_SIZE} tokens")  # debug print
        return filtered_data

    def process_data(self, data):
        print(f"Processing {len(data)} items")  # debug print
        for question, (answer_doc, _) in data:
            print(f"Processing question: {question[:100]}...")  # debug print
            self.total_tokens += len(answer_doc)
            self.total_sentences += len(list(answer_doc.sents))
            self.process_each_token(answer_doc)
            self.process_each_sentence(answer_doc)

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
        return avg_sentence_length, avg_sentence_perplexity

    def print_results(self, avg_sentence_length, avg_sentence_perplexity):
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

    def plot_perplexities(self):
        filtered_sentence_perplexities = [p for p in self.sentence_perplexities if p <= 10]

        sns.set_style('white')
        sns.kdeplot(filtered_sentence_perplexities, color='skyblue', fill=True)
        plt.title('Density Plot of Sentence Perplexities')
        plt.xlabel('Perplexity')
        plt.ylabel('Density')

        plt.xlim(0, 10)
        plt.xticks(range(0, 13))

        plt.ylim(0, 0.2)
        sns.despine()
        plt.show()

    def run(self):
        pubmed_data = self.load_dataset()
        self.process_data(pubmed_data)
        avg_sentence_length, avg_sentence_perplexity = self.calculate_averages()

        print("Results for PubMed Dataset:")
        self.print_results(avg_sentence_length, avg_sentence_perplexity)
        self.plot_perplexities()


if __name__ == "__main__":
    analysis = TextAnalysis()
    analysis.run()