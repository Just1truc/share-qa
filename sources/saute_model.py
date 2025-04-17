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
    
    def __init__(
        self, 
        d_model             : int,
        n_layers            : int,
        n_heads             : int,
        dim_feedforward     : int,
        dropout             : float = 0.1,
        activation          : str   = "gelu"
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
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
        
        for mha, ffn, norm1, norm2 in self.layers:
            x = norm1(x + mha(x))
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
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
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
        
        self.speaker_module = SpeakerEmbeddingsModule(
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
                speaker_names = unique_names
            )
        
        batch, seq_len = input_ids.shape
        
        embeddings = self.token_embeddings(input_ids)
        if attention_mask == None:
            attention_mask = (input_ids != 0).int()
            
        pe_ids = torch.arange(seq_len).broadcast_to(batch, seq_len) * attention_mask
        embeddings = embeddings + self.token_pe(pe_ids)
        
        X = torch.empty(batch, seq_len, self.hidden_size)
        U = torch.empty(batch, self.hidden_size)
        hidden_state_updates    = torch.empty(batch, self.speaker_embedding_size)
        
        for i, (speaker, embedding) in enumerate(zip(speaker_names, embeddings)):
            
            s_i = unique_names.index(speaker)
            h_t = self.speaker_module[s_i]
            
            # Adding projected h_s_i on the embeddings
            embedding += h_t
            # Applying transformer
            x = self.edu_encoder(embedding)
            # Simple Mean Pooling (B, L, D) -> (B, D)
            u_t = torch.mean(X, dim=1)
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
        
        self.config : SAUTEConfig = config.hidden_size
        
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
