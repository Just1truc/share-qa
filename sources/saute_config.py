from transformers import PretrainedConfig

class SAUTEConfig(PretrainedConfig):
    model_type = "saute"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        max_position_embeddings=512,
        max_edus_per_dialog=100,
        max_edu_length=128,
        num_attention_heads=12,
        num_hidden_layers=6,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_speaker_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.max_edus_per_dialog = max_edus_per_dialog
        self.max_edu_length = max_edu_length
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_speaker_embeddings = num_speaker_embeddings
