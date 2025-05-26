import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

import math
import os

from leaf_encoder import LeafEncoder

import matplotlib.pyplot as plt


class FFTCATLeaf(nn.Module):
    def __init__(self, hidden_dim, chunk_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.kernel_proj = nn.Linear(hidden_dim, chunk_size)

    def forward(self, chunks, queries):
        """
        chunks: (B, C, T, D)
        queries: (B, D)
        returns: (B, C, D)
        """
        B, C, T, D = chunks.shape
        assert T == self.chunk_size

        X_fft = torch.fft.fft(chunks.reshape(B * C, T, D).transpose(1, 2), dim=-1)  # (B*C, D, T)

        kernels = self.kernel_proj(queries)  # (B, T)
        kernels = kernels.unsqueeze(1).repeat(1, C, 1).reshape(B * C, T)
        kernels_fft = torch.fft.fft(kernels, dim=-1).unsqueeze(1)  # (B*C, 1, T)

        conv_fft = X_fft * kernels_fft  # (B*C, D, T)
        conv_time = torch.fft.ifft(conv_fft, dim=-1).real  # (B*C, D, T)

        summaries = conv_time.mean(dim=-1).reshape(B, C, D)  # (B, C, D)
        return summaries


class BinaryTreeAttention(nn.Module): # CE version
    def __init__(self, hidden_size=768, chunk_size=64, embedding_size=512, visuals=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.depth = int(math.log2(embedding_size // chunk_size))

        self.cat_leaf = FFTCATLeaf(hidden_size, chunk_size)

        self.merge_layers = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.depth)
        ]) # TODO: consider replacing with Conv1d like in CAT paper

        self.query_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)) for _ in range(self.depth)
        ])
        self.visuals = visuals
        if visuals:
            os.makedirs("visuals", exist_ok=True)

    def forward(self, x, query, label_positions=None):
        B, L, D = x.shape
        assert L % self.chunk_size == 0, "Sequence length must be divisible by chunk size"
        num_chunks = L // self.chunk_size

        x_chunks = x.view(B, num_chunks, self.chunk_size, D)

        # summaries = x_chunks.mean(dim=2)  # (B, num_chunks, D)
        summaries = self.cat_leaf(x_chunks, query) # (B, num_chunks, D)

        attn_weights = torch.zeros(B, L, device=x.device)
        per_level_loss = 0.0

        for level in range(self.depth):
            q_proj = query[0] * self.query_vectors[level]
            q_proj = q_proj.detach().cpu().numpy()
            if self.visuals:
                plt.figure()
                plt.plot(q_proj)
                plt.title(f"Query projection at level {level}")
                plt.xlabel("Dimension")
                plt.ylabel("Value")
                plt.savefig(f"visuals/q_proj_level_{level}.png")
                plt.close()

        for b in range(B):
            q_i = query[b]  # (D,)
            idx_range = list(range(num_chunks))
            current_level = 0

            if label_positions is not None:
                target_chunk = label_positions[b].item() // self.chunk_size

            while len(idx_range) > 1 and current_level < self.depth:
                next_range = []
                losses = []
                for j in range(0, len(idx_range), 2):
                    left_idx = idx_range[j]
                    if j + 1 >= len(idx_range):
                        next_range.append(left_idx)
                        continue
                    right_idx = idx_range[j + 1]
                    left = summaries[b, left_idx]
                    right = summaries[b, right_idx]
                    merged = self.merge_layers[current_level](torch.cat([left, right], dim=-1))
                    q_proj = q_i * self.query_vectors[current_level]
                    score_left = torch.dot(q_proj, left)
                    score_right = torch.dot(q_proj, right)
                    logit = torch.stack([score_left, score_right])

                    if label_positions is not None:
                        decision = 0 if target_chunk % 2 == 0 else 1
                        loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([decision], device=x.device))
                        target_chunk = target_chunk // 2
                    else:
                        loss = torch.tensor(0.0, device=x.device)
                        decision = torch.argmax(logit).item()

                    losses.append(loss)
                    chosen = left_idx if decision == 0 else right_idx
                    next_range.append(chosen)
                idx_range = next_range
                current_level += 1
                if losses:
                    per_level_loss += sum(losses) / len(losses)

            final_chunk = idx_range[0]
            start = final_chunk * self.chunk_size
            end = start + self.chunk_size
            local_scores = torch.matmul(q_i, x[b, start:end].T)
            local_attn = F.softmax(local_scores, dim=-1)
            attn_weights[b, start:end] = local_attn

        return attn_weights, per_level_loss / B


class HierarchicalBinaryTree(nn.Module):
    def __init__(self, model_name='bert-base-cased', hidden_size=768, chunk_size=64):
        super().__init__()
        self.encoder = LeafEncoder(model_name=model_name)  # frozen BERT embedder
        self.tree_model = BinaryTreeAttention(hidden_size=hidden_size, chunk_size=chunk_size)

    def forward(self, input_ids, attention_mask, label_pos, teacher_attn):
        # Step 1: Embed tokens using frozen BERT
        with torch.no_grad():
            # x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        query = x[:, 0, :] # CLS token as query
        # Step 2: Tree-based attention and loss
        tree_attn, gate_loss = self.tree_model(x, query, label_pos)
        # Step 3: KL divergence loss (student vs teacher)
        attn_loss = F.kl_div((tree_attn + 1e-8).log(), teacher_attn[:, 0], reduction='batchmean')
        return attn_loss + gate_loss
