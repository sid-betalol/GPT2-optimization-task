import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import os

class TextDataset(Dataset):
    def __init__(self, file_path=None, tokenizer_name='gpt2', max_length=1024, dataset_name='wikitext', dataset_config_name='wikitext-103-raw-v1'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if file_path and os.path.exists(file_path):
            self.samples = self._build_dataset_from_file(file_path)
        else:
            self.samples = self._build_dataset_from_transformers(dataset_name, dataset_config_name)

    def _build_dataset_from_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokenized_text = self.tokenizer.encode(line, truncation=True, max_length=self.max_length)
                samples.append(tokenized_text)
        return samples

    def _build_dataset_from_transformers(self, dataset_name, dataset_config_name):
        dataset = load_dataset(dataset_name, dataset_config_name)
        samples = []
        for entry in dataset['train']:
            tokenized_text = self.tokenizer.encode(entry['text'], truncation=True, max_length=self.max_length)
            samples.append(tokenized_text)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokenized_text = self.samples[idx]
        input_ids = torch.tensor(tokenized_text, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_text), dtype=torch.long)
        return input_ids, attention_mask
