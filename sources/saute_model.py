import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertModel, BertTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput
from sources.saute_config import SAUTEConfig

class SpeakerEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.hidden_size = hidden_size
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def forward(self, speaker_names):
        inputs = self.tokenizer(speaker_names, padding=True, truncation=True, return_tensors="pt")
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

class SAUTEUnit(nn.Module):
    def __init__(self, config : SAUTEConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

    def forward(self, x):
        return self.encoder(x)

class SAUTEModel(PreTrainedModel):
    config_class = SAUTEConfig

    def __init__(self, config : SAUTEConfig):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.speaker_encoder = SpeakerEncoder(config.hidden_size)
        self.saute_unit = SAUTEUnit(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids : torch.Tensor, speaker_names, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        word_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(positions)
        speaker_embeds = self.speaker_encoder(speaker_names).unsqueeze(1).expand(-1, seq_len, -1)

        hidden_states = word_embeds + pos_embeds + speaker_embeds
        hidden_states = self.dropout(hidden_states)

        output = self.saute_unit(hidden_states.transpose(0, 1)).transpose(0, 1)
        logits = self.lm_head(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
