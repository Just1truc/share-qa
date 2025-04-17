from datasets import load_dataset
from transformers import BertTokenizerFast
import torch
import random
import json

MAX_EDU_LEN = 128
MAX_EDUS_PER_DIALOG = 100

class SAUTEDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.dataset = load_dataset("JustinDuc/MultiDomain-QADialog", split=split)

    def __len__(self):
        return len(self.dataset)

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __getitem__(self, idx):
        item = self.dataset[idx]
        edus = item['Dialog (EDUs)'][:MAX_EDUS_PER_DIALOG]
        speakers = json.loads(item['Speakers'])[:MAX_EDUS_PER_DIALOG]

        tokenized = self.tokenizer(edus, padding="max_length", truncation=True,
                                   max_length=MAX_EDU_LEN, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(1)
        attention_mask = tokenized["attention_mask"].squeeze(1)

        input_ids, labels = self.mask_tokens(input_ids)

        speaker_names = [s if s else "unknown" for s in speakers]

        return {
            "input_ids": input_ids.view(-1),
            "speaker_names": speaker_names,
            "attention_mask": attention_mask.view(-1),
            "labels": labels.view(-1)
        }
