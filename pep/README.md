# Hierarchical Retrieval Models

This project explores three models for long-context retrieval based on hierarchical representations:

### 1. Binary Merger Model

* **Structure**: Uses fixed binary merges per level to build summaries.
* **Problem**: Error propagation. Since each summary is repeatedly compressed through log(d) layers, only dominant features survive, and all query projections tend to collapse to similar patterns across layers.
* **Effect**: Poor expressivity and increased noise at upper layers.

### 2. Convolutional Merger Model

* **Structure**: Uses convolutional layers to build summaries with increasing kernel sizes.
* **Problem**: Lacks positional flexibility. It assumes fixed spatial patterns, but token importance is not defined by fixed location — it's defined by the actual token content.
* **Effect**: Inflexible information selection.

### 3. Attention Merger Model

* **Structure**: At each level, learns attention weights over groups of leaf summaries using a learnable router.
* **Advantage**: Learns what features to retain per level, allowing dynamic routing based on query-relevant information. Summaries are always derived from the leaf nodes.

---

## Training Procedure

* **Loss**: Cross-entropy or negative log-likelihood over attention scores per level.
* **Routing**: Currently uses soft attention routing per level (softmax over all summaries).
* **Teacher**: Randomly initialized teacher model using softmax over context.
* **Data**: Randomly sampled token sequences.

---

## Limitations and Next Steps

* **Random Data**: Training on unstructured, random sequences doesn't allow the model to learn meaningful patterns.
* **Teacher Quality**: A randomly initialized teacher has no useful attention supervision.

### Planned Improvements

1. **Replace Teacher**: Use a transformer trained with masked language modeling (MLM) as a softmax teacher.
2. **Use Real Data**: Replace random input with datasets like HotpotQA, TriviaQA, or LongBench to introduce real structure.
3. **Depth Testing**: Experiment with different tree depths to study model scaling.

---

This setup aims to evaluate whether hierarchical attention mechanisms can match or exceed softmax attention performance in both accuracy and efficiency.
