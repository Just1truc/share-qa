import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA
from transformers import PreTrainedModel, BertModel, BertTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput
from sources.saute_config import SAUTEConfig

activation_to_class = {
    "gelu" : nn.GELU,
    "relu" : nn.ReLU,
    "sigmoid" : nn.Sigmoid
}

class FlashTransformerEncoder(nn.Module):
    
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, dropout=0.1, activation="gelu"):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = self.head_dim * n_heads
        self.layers   = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList([
                    MHA(embed_dim=d_model, num_heads=n_heads, dropout=dropout),
                    nn.Sequential(
                        nn.Linear(d_model, dim_feedforward),
                        activation_to_class[activation](),
                        nn.Dropout(dropout),
                        nn.Linear(dim_feedforward, d_model),
                        nn.Dropout(dropout),
                    ),
                    nn.LayerNorm(d_model),
                    nn.LayerNorm(d_model)
                ])
            )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        for mha, ffn, norm1, norm2 in self.layers:
            out = mha(x)
            x = norm1(x + out)
            x = norm2(x + ffn(x))
            
        return x

def flatten(entry : list):
    return [subsubentry for subentry in entry for subsubentry in subentry]

class SpeakerEmbeddingsModule(nn.Module):
    
    def __init__(
        self,
        hidden_size : int,
        embed_dim   : int,
        activation  : str = "gelu",
        dim_update  : int = 2048,
        dropout     : float = 0.3
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.hidden_size    = hidden_size
        self.embed_dim      = embed_dim
        self.tokenizer      = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.speaker_embeddings = None
        
        self.update_gate = nn.Sequential(
            nn.Linear(
                in_features     = self.hidden_size + self.embed_dim,
                out_features    = dim_update
            ),
            activation_to_class[activation](),
            nn.Dropout(dropout),
            nn.Linear(dim_update, self.hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.linear_projection = nn.Linear(
            in_features     = hidden_size,
            out_features    = embed_dim,
            bias=False
        )
        
    
    def init(self, unique_speakers : list[str]) -> torch.Tensor:
        # No batching first then batching later if too slow

        # Speakers to unique speakers
        inputs  = self.tokenizer(unique_speakers, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        h_0     = outputs.last_hidden_state.mean(dim=1)
        
        # Setup Speaker Embeddings 
        self.speaker_embeddings = h_0
        return h_0
    
    
    def __getitem__(self, s_i : int):
        
        h_t = self.speaker_embeddings[s_i]
        return self.linear_projection(h_t)


    def update(
        self,
        u_t : torch.Tensor,
        s_i : int
    ) -> torch.Tensor:
        
        h_t             = self.speaker_embeddings[s_i]
        update_material = torch.concat([u_t, h_t])
        new_h_t         = self.update_gate(update_material)
        
        # Residual connection
        self.speaker_embeddings[s_i] += new_h_t
        return new_h_t


class CrossSpeakerEmbeddingsModule(nn.Module):
    
    def __init__(
        self,
        hidden_size : int,
        embed_dim   : int,
        activation  : str = "gelu",
        dim_update  : int = 2048,
        dropout     : float = 0.3
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.hidden_size    = hidden_size
        self.embed_dim      = embed_dim
        self.tokenizer      = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.speaker_embeddings = None
        
        self.update_gate = nn.Sequential(
            nn.Linear(
                in_features     = self.hidden_size + self.embed_dim * 2,
                out_features    = dim_update
            ),
            activation_to_class[activation](),
            nn.Dropout(dropout),
            nn.Linear(dim_update, self.hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.linear_projection = nn.Linear(
            in_features     = hidden_size,
            out_features    = embed_dim,
            bias=False
        )

        self.attention_projection = nn.Linear(
            in_features     = embed_dim,
            out_features    = hidden_size,
            bias=False
        )
        
    
    def init(self, unique_speakers : list[str]) -> torch.Tensor:
        # No batching first then batching later if too slow

        # Speakers to unique speakers
        inputs  = self.tokenizer(unique_speakers, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        h_0     = outputs.last_hidden_state.mean(dim=1)
        
        # Setup Speaker Embeddings 
        self.speaker_embeddings = h_0
        return h_0
    
    
    def __getitem__(self, s_i : int):
        
        h_t = self.speaker_embeddings[s_i]
        return self.linear_projection(h_t)


    def update(
        self,
        u_t : torch.Tensor,
        s_i : int
    ) -> torch.Tensor:
        
        h_all = self.speaker_embeddings
        query = self.attention_projection(u_t)
        attn_logits = (h_all @ query)
        attn_weights = torch.softmax(attn_logits, dim=0)
        context = (attn_weights.unsqueeze(-1) * h_all).sum(dim=0)

        h_s = self.speaker_embeddings[s_i]
        update_material = torch.cat([u_t, h_s, context], dim=-1)
        new_h_s = self.update_gate(update_material)

        new_embeddings = self.speaker_embeddings.clone()
        new_embeddings[s_i] = h_s + new_h_s
        self.speaker_embeddings = new_embeddings
        
        return new_h_s

class SauteUnit(nn.Module):
    
    def __init__(self, config : SAUTEConfig):
        super().__init__()
        
        self.edu_encoder    = FlashTransformerEncoder(
            d_model         = config.hidden_size,
            n_heads         = config.num_attention_heads,
            n_layers        = config.num_hidden_layers,
            dim_feedforward = config.intermediate_size,
            dropout         = config.hidden_dropout_prob,
            activation      = "gelu"
        )
        
        # self.speaker_module = SpeakerEmbeddingsModule(
        self.speaker_module = CrossSpeakerEmbeddingsModule(
            hidden_size     = config.speaker_embeddings_size,
            embed_dim       = config.hidden_size,
            activation      = "gelu",
            dropout         = config.hidden_dropout_prob
        )
        
        self.token_embeddings   = nn.Embedding(
            num_embeddings      = config.vocab_size,
            embedding_dim       = config.hidden_size
        )
        
        self.token_pe           = nn.Embedding(
            num_embeddings      = config.max_position_embeddings,
            embedding_dim       = config.hidden_size
        )
        
        self.hidden_size = config.hidden_size
        self.speaker_embedding_size = config.speaker_embeddings_size
        
    def forward(
        self,
        input_ids       : torch.Tensor,
        speaker_names   : list[str],
        attention_mask  : torch.Tensor | None = None,
        hidden_state    : torch.Tensor | None = None
    ):
        if hidden_state == None:
            
            unique_names = list(set(speaker_names))
            hidden_state = self.speaker_module.init(
                unique_speakers = unique_names
            )
        
        batch, seq_len = input_ids.shape
        
        embeddings = self.token_embeddings(input_ids)
        if attention_mask == None:
            attention_mask = (input_ids != 0).int()
            
        pe_ids = torch.arange(seq_len).broadcast_to(batch, seq_len).to("cuda:0") * attention_mask
        embeddings = embeddings + self.token_pe(pe_ids)
        
        X = torch.empty(batch, seq_len, self.hidden_size).to("cuda:0")
        U = torch.empty(batch, self.hidden_size).to("cuda:0")
        hidden_state_updates = torch.empty(batch, self.speaker_embedding_size).to("cuda:0")
        
        for i, (speaker, embedding) in enumerate(zip(speaker_names, embeddings)):
            
            s_i = unique_names.index(speaker)
            h_t = self.speaker_module[s_i]

            # Applying transformer
            x = self.edu_encoder((embedding + h_t).unsqueeze(0)).squeeze(0)
            
            # Simple Mean Pooling (L, D) -> (D)
            mask = (input_ids[i] != 0).unsqueeze(-1)
            sum_x = (x * mask).sum(dim=0)
            count_x = mask.sum(dim=0).clamp(min=1e-6)
            u_t = sum_x / count_x
            # Updating speaker state
            new_h_t = self.speaker_module.update(u_t, s_i)
            
            # Outputs
            X[i] = x
            U[i] = u_t 
            hidden_state_updates[i] = new_h_t
        
        return X, (U, self.speaker_module.speaker_embeddings, hidden_state_updates)


class UtteranceEmbedings(PreTrainedModel):
    config_class = SAUTEConfig

    def __init__(self, config : SAUTEConfig):
        super().__init__(config)
        
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size)
        self.saute_unit = SauteUnit(config)
        
        self.config : SAUTEConfig = config
        
        self.init_weights()

    def forward(
        self,
        input_ids       : torch.Tensor,
        speaker_names   : list[str],
        attention_mask  : torch.Tensor  = None,
        labels          : torch.Tensor  = None
    ):

        X, _ = self.saute_unit.forward(
            input_ids       =   input_ids,
            speaker_names   =   speaker_names,
            attention_mask  =   attention_mask,
            hidden_state    =   None
        )
        
        logits = self.lm_head(X)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
