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


class ZeroInitCrossSpeakerEmbeddings(nn.Module):
    
    def __init__(
        self,
        hidden_size : int,
        embed_dim   : int,
        activation  : str = "gelu",
        dim_update  : int = 2048,
        dropout     : float = 0.3
    ):
        super().__init__()
        
        self.hidden_size    = hidden_size
        self.embed_dim      = embed_dim
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
        
        self.speaker_embeddings = torch.zeros(len(unique_speakers), self.hidden_size).to("cuda:0")
        return self.speaker_embeddings
    
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

class FlashTransformer(nn.Module):
    
    def __init__(
        self,
        d_model     : int,
        n_heads     : int,
        dim_feedforward : int,
        dropout     : float = 0.1,
        activation  : str = "gelu"
    ):
        super().__init__()
        
        assert activation in activation_to_class.keys(), f"Unknown activation : [{activation}]"
        
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = self.head_dim * n_heads
        
        self.mha = MHA(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_to_class[activation](),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x : torch.Tensor
    ):
        out = self.mha(x)
        x = self.norm1(x + out)
        x = self.norm2(x + self.mlp(x))
        
        return x

class HSauteUnit(nn.Module):
    
    # No batch support at First
    
    def __init__(
        self,
        config : SAUTEConfig
    ):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(
            num_embeddings      = config.vocab_size,
            embedding_dim       = config.hidden_size
        )
        self.token_pe           = nn.Embedding(
            num_embeddings      = config.max_position_embeddings,
            embedding_dim       = config.hidden_size
        )
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                nn.ModuleList([
                    FlashTransformer(
                        d_model         = config.hidden_size,
                        n_heads         = config.num_attention_heads,
                        dim_feedforward = config.intermediate_size,
                        dropout         = config.hidden_dropout_prob,
                        activation      = "gelu"
                    ),
                    ZeroInitCrossSpeakerEmbeddings(
                        hidden_size     = config.speaker_embeddings_size,
                        embed_dim       = config.hidden_size,
                        dropout         = config.hidden_dropout_prob,  
                        activation      = "gelu"
                    )
                ])
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
        batch, seq_len = input_ids.shape
        
        unique_names    = list(set(speaker_names))
        embeddings      = self.token_embeddings(input_ids)
        
        if attention_mask == None:
            attention_mask = (input_ids != 0).int()
            
        pe          = self.token_pe(torch.arange(seq_len).broadcast_to(batch, seq_len).to("cuda:0") * attention_mask)
        embeddings  = embeddings + pe
        
        for edu_encoder, speaker_module in self.layers:
            speaker_module.init(unique_speakers = unique_names)

            X = torch.empty(batch, seq_len, self.hidden_size).to("cuda:0")
            
            for i, (speaker, embedding) in enumerate(zip(speaker_names, embeddings)):

                s_i = unique_names.index(speaker)
                h_t = speaker_module[s_i]

                # Applying transformer
                x = edu_encoder((embedding + h_t).unsqueeze(0)).squeeze(0)

                # Simple Mean Pooling (L, D) -> (D)
                mask = (input_ids[i] != 0).unsqueeze(-1)
                sum_x = (x * mask).sum(dim=0)
                count_x = mask.sum(dim=0).clamp(min=1e-6)
                u_t = sum_x / count_x
                
                # Updating speaker state
                speaker_module.update(u_t, s_i)

                # Outputs
                X[i] = x
            
            embeddings = X
        
        # Contextual embeddings
        return embeddings, []
                
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

class BatchAwareHSauteUnit(nn.Module):
    
    def __init__(self, config : SAUTEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.speaker_embedding_size = config.speaker_embeddings_size

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_pe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.num_layers = config.num_hidden_layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    activation_to_class["gelu"](),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob),
                ),
                nn.LayerNorm(config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(self.hidden_size * 3, self.hidden_size),  # gated cross-speaker update
                nn.LayerNorm(self.hidden_size)
            ]) for _ in range(self.num_layers)
        ])

        self.token_level_proj = nn.Linear(self.speaker_embedding_size, self.hidden_size)
        self.attention_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
        self.cross_speaker_mha = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)

    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, speaker_names : list[str]):
        # print(input_ids.shape)
        B, T, L = input_ids.size()
        device = input_ids.device

        flat_speakers = [name for dialog in speaker_names for name in dialog]
        unique_speakers = list(sorted(set(flat_speakers)))
        speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}

        speaker_ids = torch.tensor([[speaker_to_id[name] for name in dialog] for dialog in speaker_names], device=device)
        num_speakers = len(unique_speakers)

        speaker_states = [torch.zeros(num_speakers, self.speaker_embedding_size, device=device) for _ in range(self.num_layers)]

        token_embed = self.token_embeddings(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, T, L)
        token_embed = token_embed + self.token_pe(pos_ids)

        X = token_embed

        for layer_index, (mha, ffn, norm1, norm2, speaker_gate, speaker_norm) in enumerate(self.encoder_layers):
            speaker_state = speaker_states[layer_index]
            X_new = torch.zeros_like(X)
            new_speaker_state = speaker_state.clone()

            for t in range(T):
                s_idx = speaker_ids[:, t]  # (B,)
                s_embed = speaker_state[s_idx]  # (B, D)
                h_input = X[:, t] + s_embed.unsqueeze(1)  # (B, L, D)

                h_attn = mha(h_input)
                h_input = norm1(h_input + h_attn)
                h_input = norm2(h_input + ffn(h_input))

                X_new[:, t] = h_input
                
                token_query = self.token_level_proj(s_embed).unsqueeze(1)  # (B, 1, D)
                token_logits = (token_query * h_input).sum(dim=-1)  # (B, L)
                token_logits = token_logits.masked_fill(attention_mask[:, t] == 0, float('-inf'))
                token_weights = torch.softmax(token_logits, dim=-1)  # (B, L)
                token_context = (token_weights.unsqueeze(-1) * h_input).sum(dim=1)  # (B, D)

                speaker_query = token_context.unsqueeze(1)  # (B, 1, D)
                expanded_state = speaker_state.unsqueeze(0).expand(B, -1, -1)  # (B, S, D)
                speaker_context, _ = self.cross_speaker_mha(speaker_query, expanded_state, expanded_state)
                speaker_context = speaker_context.squeeze(1)  # (B, D)

                concat = torch.cat([token_context, s_embed, speaker_context], dim=-1)
                updated = speaker_gate(concat)  # (B, D)
                updated = speaker_norm(updated + s_embed)
                new_speaker_state.index_add_(0, s_idx, updated)

            speaker_states[layer_index] = new_speaker_state
            X = X_new

        return X, speaker_states[-1]


class VerticalSpeakerMemoryTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.speaker_embedding_size = config.speaker_embeddings_size

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_pe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.turn_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.num_layers = config.num_hidden_layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    activation_to_class["gelu"](),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob),
                ),
                nn.LayerNorm(config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size * 3, config.hidden_size),  # for token + speaker + context
                nn.LayerNorm(config.hidden_size),
                nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=4, batch_first=True)
            ]) for _ in range(self.num_layers)
        ])

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask, speaker_names):
        B, T, L = input_ids.size()
        device = input_ids.device

        turn_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

        token_embed = self.token_embeddings(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, T, L)
        token_embed = token_embed + self.token_pe(pos_ids)

        turn_embed = self.turn_position_embeddings(turn_ids)  # (B, T, D)
        turn_embed = turn_embed.unsqueeze(2).expand(-1, -1, L, -1)  # (B, T, L, D)
        X = token_embed + turn_embed

        speaker_states = [
            [torch.zeros(len(set(dialog)), self.hidden_size, device=device) for dialog in speaker_names]
            for _ in range(self.num_layers)
        ]

        speaker_id_maps = [
            {spk: idx for idx, spk in enumerate(sorted(set(dialog)))}
            for dialog in speaker_names
        ]

        speaker_ids = torch.tensor([
            [speaker_id_maps[i][name] for name in dialog]
            for i, dialog in enumerate(speaker_names)
        ], device=device)

        for l, (mha, ffn, norm1, norm2, speaker_gate, speaker_norm, cross_attn) in enumerate(self.encoder_layers):
            X_new = torch.zeros_like(X)

            for i in range(B):
                dialog_speakers = speaker_id_maps[i]
                dialog_speaker_memory = speaker_states[l][i] if l == 0 else speaker_states[l - 1][i]
                new_speaker_memory = dialog_speaker_memory.clone()

                for spk, s_idx in dialog_speakers.items():
                    turn_mask = (speaker_ids[i] == s_idx)
                    if not turn_mask.any():
                        continue

                    reps = []
                    for t in torch.nonzero(turn_mask, as_tuple=False).squeeze(1):
                        # print(dialog_speaker_memory.shape, s_idx)
                        # print(dialog_speakers)
                        s_embed = dialog_speaker_memory[s_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                        h = X[i, t] + s_embed  # (L, D)

                        h_masked = h * attention_mask[i, t].unsqueeze(-1)

                        h = mha(h_masked).squeeze(0)
                        h = norm1(h + X[i, t])
                        h = norm2(h + ffn(h))
                        X_new[i, t] = h
                        reps.append(h)

                    reps_tensor = torch.stack(reps)
                    token_context = reps_tensor.mean(dim=1)

                    query = token_context.unsqueeze(1)
                    memory = dialog_speaker_memory.unsqueeze(0).expand(token_context.size(0), -1, -1)
                    cross_context, _ = cross_attn(query, memory, memory)
                    cross_context = cross_context.squeeze(1)

                    fused = torch.cat([
                        token_context,
                        dialog_speaker_memory[s_idx].unsqueeze(0).expand_as(token_context),
                        cross_context
                    ], dim=-1)

                    updated = speaker_gate(fused)
                    updated = speaker_norm(updated + dialog_speaker_memory[s_idx])
                    new_speaker_memory[s_idx] = updated.mean(dim=0)

                speaker_states[l][i] = new_speaker_memory
            X = X_new

        # final_states = torch.stack([torch.stack(dialog_states) for dialog_states in speaker_states[-1]])
        return X, speaker_states[-1]


class SpeakerMemory(nn.Module):
    def __init__(self, hidden_size : int):
        super().__init__()
        self.gate = nn.GRUCell(hidden_size, hidden_size)

    def forward(
        self,
        speaker_memory : torch.Tensor,
        speaker_ids : torch.Tensor,
        edu_reps : torch.Tensor
    ):
        updated_memory = speaker_memory.clone()
        for idx, sid in enumerate(speaker_ids):
            updated_memory[sid] = self.gate(edu_reps[idx], speaker_memory[sid])
        return updated_memory


class SpeakerGatedLocalTransformer(nn.Module):
    
    def __init__(self, config : SAUTEConfig):
    
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_pe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.max_speakers, config.hidden_size)

        self.edu_token_encoder = nn.ModuleList([
            nn.ModuleList([
                MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    activation_to_class["gelu"](),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob)
                ),
                nn.LayerNorm(config.hidden_size),
                nn.LayerNorm(config.hidden_size)
            ]) for _ in range(config.num_token_layers)
        ])

        self.speaker_memory = SpeakerMemory(config.hidden_size)

        # self.edu_local_encoder = nn.ModuleList([
        #     nn.ModuleList([
        #         MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
        #         nn.Sequential(
        #             nn.Linear(config.hidden_size, config.intermediate_size),
        #             activation_to_class["gelu"](),
        #             nn.Dropout(config.hidden_dropout_prob),
        #             nn.Linear(config.intermediate_size, config.hidden_size),
        #             nn.Dropout(config.hidden_dropout_prob)
        #         ),
        #         nn.LayerNorm(config.hidden_size),
        #         nn.LayerNorm(config.hidden_size)
        #     ]) for _ in range(config.num_edu_layers)
        # ])

    def forward(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        speaker_names : list[str]
    ):
        B, T, L = input_ids.size()
        device = input_ids.device

        # Infer speaker IDs from names
        batch_speaker_maps = []
        speaker_ids = []
        for dialog in speaker_names:
            speaker_map = {name: idx for idx, name in enumerate(sorted(set(dialog)))}
            batch_speaker_maps.append(speaker_map)
            speaker_ids.append(torch.tensor([speaker_map[name] for name in dialog], device=device))

        speaker_ids = torch.stack(speaker_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, T, L)
        token_embeds += self.token_pe(position_ids)

        speaker_embeds = self.speaker_embeddings(speaker_ids).unsqueeze(2).expand(-1, -1, L, -1)
        token_embeds += speaker_embeds

        token_embeds = token_embeds.view(B*T, L, -1)
        attention_mask = attention_mask.view(B*T, L)

        for mha, ffn, norm1, norm2 in self.edu_token_encoder:
            x = mha(token_embeds, key_padding_mask=~attention_mask.bool())
            token_embeds = norm1(token_embeds + x)
            x = ffn(token_embeds)
            token_embeds = norm2(token_embeds + x)

        token_embeds = token_embeds.view(B, T, L, -1)
        attention_mask_exp = attention_mask.view(B, T, L).unsqueeze(-1)
        edu_reps = (token_embeds * attention_mask_exp).sum(dim=2) / attention_mask_exp.sum(dim=2).clamp(min=1e-6)

        num_speakers = max(len(map) for map in batch_speaker_maps)
        speaker_memory = torch.zeros(num_speakers, self.hidden_size, device=device)
        speaker_memory = self.speaker_memory(speaker_memory, speaker_ids.view(-1), edu_reps.view(-1, self.hidden_size))

        return token_embeds.view(B, T, L, -1), speaker_memory


class SelectiveMemoryUnit(nn.Module):
    def __init__(self, hidden_size, retrieval_topk=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.retrieval_topk = retrieval_topk

        self.speaker_query = nn.Linear(hidden_size, hidden_size)
        self.content_query = nn.Linear(hidden_size, hidden_size)

        self.speaker_proj = nn.Linear(hidden_size, hidden_size)
        self.content_proj = nn.Linear(hidden_size, hidden_size)

        self.memory_updater = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, edu_reps, speaker_ids, batch_speaker_maps):
        B, T, D = edu_reps.shape
        device = edu_reps.device

        new_memories = []

        for b in range(B):
            dialog = edu_reps[b]  # (T, D)
            speaker_id_dialog = speaker_ids[b]  # (T,)
            speaker_map = batch_speaker_maps[b]
            num_speakers = len(speaker_map)

            memory = torch.zeros(num_speakers, D, device=device)

            for t in range(T):
                current_edu = dialog[t]
                current_speaker = speaker_id_dialog[t]

                past_edus = dialog[:t] if t > 0 else torch.empty(0, D, device=device)
                if past_edus.size(0) == 0:
                    continue

                speaker_scores = (self.speaker_query(current_edu) @ self.speaker_proj(past_edus).transpose(0, 1))
                content_scores = (self.content_query(current_edu) @ self.content_proj(past_edus).transpose(0, 1))

                total_scores = speaker_scores + content_scores

                topk_vals, topk_idx = torch.topk(total_scores, k=min(self.retrieval_topk, past_edus.size(0)))
                selected_edus = past_edus[topk_idx]

                retrieved_summary = selected_edus.mean(dim=0)

                memory[current_speaker] = self.memory_updater(retrieved_summary, memory[current_speaker])

            new_memories.append(memory)

        return new_memories

class SelectiveMemoryTransformer(nn.Module):
    def __init__(self, config : SAUTEConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_pe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.max_speakers, config.hidden_size)

        self.initial_encoder = nn.ModuleList([
            nn.ModuleList([
                MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    activation_to_class["gelu"](),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob)
                ),
                nn.LayerNorm(config.hidden_size),
                nn.LayerNorm(config.hidden_size)
            ]) for _ in range(config.num_token_layers)
        ])

        self.memory_unit = SelectiveMemoryUnit(config.hidden_size, retrieval_topk=5)

        self.final_encoder = nn.ModuleList([
            nn.ModuleList([
                MHA(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.hidden_dropout_prob),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    activation_to_class["gelu"](),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob)
                ),
                nn.LayerNorm(config.hidden_size),
                nn.LayerNorm(config.hidden_size)
            ]) for _ in range(config.num_token_layers)
        ])

    def forward(self, input_ids, attention_mask, speaker_names):
        B, T, L = input_ids.size()
        device = input_ids.device

        batch_speaker_maps = []
        speaker_ids = []
        for dialog in speaker_names:
            speaker_map = {name: idx for idx, name in enumerate(sorted(set(dialog)))}
            batch_speaker_maps.append(speaker_map)
            speaker_ids.append(torch.tensor([speaker_map[name] for name in dialog], device=device))

        speaker_ids = torch.stack(speaker_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, T, L)
        token_embeds += self.token_pe(position_ids)

        speaker_embeds = self.speaker_embeddings(speaker_ids).unsqueeze(2).expand(-1, -1, L, -1)
        token_embeds += speaker_embeds

        token_embeds = token_embeds.view(B*T, L, -1)
        attention_mask = attention_mask.view(B*T, L)

        for mha, ffn, norm1, norm2 in self.initial_encoder:
            x = mha(token_embeds, key_padding_mask=~attention_mask.bool())
            token_embeds = norm1(token_embeds + x)
            x = ffn(token_embeds)
            token_embeds = norm2(token_embeds + x)

        token_embeds = token_embeds.view(B, T, L, -1)
        attention_mask_exp = attention_mask.view(B, T, L).unsqueeze(-1)
        edu_reps = (token_embeds * attention_mask_exp).sum(dim=2) / attention_mask_exp.sum(dim=2).clamp(min=1e-6)

        speaker_memories = self.memory_unit(edu_reps, speaker_ids, batch_speaker_maps)

        # Inject speaker memory into token embeddings
        updated_token_embeds = token_embeds.clone()
        for b in range(B):
            for t in range(T):
                speaker_idx = speaker_ids[b, t]
                memory = speaker_memories[b][speaker_idx]
                updated_token_embeds[b, t] += memory.unsqueeze(0)

        updated_token_embeds = updated_token_embeds.view(B*T, L, -1)

        for mha, ffn, norm1, norm2 in self.final_encoder:
            x = mha(updated_token_embeds, key_padding_mask=~attention_mask.bool())
            updated_token_embeds = norm1(updated_token_embeds + x)
            x = ffn(updated_token_embeds)
            updated_token_embeds = norm2(updated_token_embeds + x)

        updated_token_embeds = updated_token_embeds.view(B, T, L, -1)

        return updated_token_embeds, []
    


class UtteranceEmbedings(PreTrainedModel):
    config_class = SAUTEConfig

    def __init__(self, config : SAUTEConfig):
        super().__init__(config)
        
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size)
        # self.saute_unit = HSauteUnit(config)
        # self.saute_unit = BatchAwareHSauteUnit(config)
        # self.saute_unit = VerticalSpeakerMemoryTransformer(config)
        # self.saute_unit  = SpeakerGatedLocalTransformer(config)
        self.saute_unit = SelectiveMemoryTransformer(config)

        self.config : SAUTEConfig = config
        
        self.init_weights()

    def forward(
        self,
        input_ids       : torch.Tensor,
        speaker_names   : list[str],
        attention_mask  : torch.Tensor  = None,
        labels          : torch.Tensor  = None
    ):
        # print(input_ids.shape)
        X, _ = self.saute_unit.forward(
            input_ids       =   input_ids,
            speaker_names   =   speaker_names,
            attention_mask  =   attention_mask,
            # hidden_state    =   None
        )
        # print(X.shape)
        
        logits = self.lm_head(X)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
