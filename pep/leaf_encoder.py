import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

##### Using the pretrained teacher model for the encoding #####

class LeafEncoder(nn.Module):
    
    def __init__(self, model_name='bert-base-uncased'):
        
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def encode_query(self, input_ids):
        return self.bert.embeddings.word_embeddings(input_ids)

    def forward(self, input_ids, attention_mask):
        
        # Add CLS token - MOVED TO PREPROCESSING
        # cls_token_id = self.tokenizer.cls_token_id
        # cls_tokens = torch.full((input_ids.size(0), 1), cls_token_id, dtype=torch.long, device=input_ids.device)
        # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
        # cls_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        # attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Process sequences
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return outputs.last_hidden_state, outputs.attentions[-1]
