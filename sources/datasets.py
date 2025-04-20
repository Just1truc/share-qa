from datasets import load_dataset
from transformers import BertTokenizerFast
from functools import reduce
import torch
import random
import json

MAX_EDU_LEN = 128
MAX_EDUS_PER_DIALOG = 100

# class SAUTEDataset(torch.utils.data.Dataset):
#     def __init__(self, split="train"):
#         self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#         self.dataset = load_dataset("JustinDuc/MultiDomain-QADialog", split=split)

#     def __len__(self):
#         return len(self.dataset)

#     def mask_tokens(self, inputs):
#         labels = inputs.clone()
#         probability_matrix = torch.full(labels.shape, 0.15)
#         special_tokens_mask = [
#             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#         ]
#         probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]
#         return inputs, labels

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         edus = json.loads(item['Dialog (EDUs)'])[:MAX_EDUS_PER_DIALOG]
#         speakers = json.loads(item['Speakers'])[:MAX_EDUS_PER_DIALOG]

#         # print(edus)
#         tokenized = self.tokenizer(edus, padding="max_length", truncation=True,
#                                    max_length=MAX_EDU_LEN, return_tensors="pt")
#         # print(tokenized)
#         input_ids = tokenized["input_ids"]
#         attention_mask = tokenized["attention_mask"]

#         input_ids, labels = self.mask_tokens(input_ids)

#         speaker_names = [s if s else "unknown" for s in speakers]

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "speaker_names": speaker_names,
#             "labels": labels
#         }


class SAUTEDataset(torch.utils.data.Dataset):
    
    dialog_formats = [
        "edu",
        "full"
    ]
    
    def __init__(
        self,
        split         : str = "train",
        dialog_format : str = "edu"
    ):
        assert dialog_format in SAUTEDataset.dialog_formats, f"Unknown dialog format {dialog_format}. Available dialog formats are {str(SAUTEDataset.dialog_formats)}"
        
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.dataset = load_dataset("allenai/soda", split=split)
        
        self.dialog_format = dialog_format

    def __len__(self):
        return len(self.dataset)

    def mask_tokens(
        self,
        inputs : torch.Tensor
    ):
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

    def __getitem__(
        self,
        idx
    ):
        item = self.dataset[idx]
        edus = (item['dialogue'])[:MAX_EDUS_PER_DIALOG]
        speakers = (item['speakers'])[:MAX_EDUS_PER_DIALOG]
        # print(list(zip(edus, speakers)))
        if self.dialog_format == "full":
            edus = ["\n".join(map(lambda x : "[" + x[0] + "]: " + x[1], zip(speakers, edus)))]
            # print(edus)

        # print(edus)
        tokenized = self.tokenizer(edus, padding="max_length", truncation=True,
                                   max_length=MAX_EDU_LEN, return_tensors="pt")
        # print(tokenized)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids, labels = self.mask_tokens(input_ids)

        speaker_names = [s if s else "unknown" for s in speakers]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **({"speaker_names": speaker_names} if self.dialog_format == "edu" else {}),
            "labels": labels
        }
