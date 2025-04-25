from transformers import TrainerCallback
import torch
import wandb

class WandbPredictionLoggerCallback(TrainerCallback):
    def __init__(self, fixed_batch, tokenizer, log_every_steps=500):
        self.fixed_batch = fixed_batch
        self.tokenizer = tokenizer
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_steps == 0:
            self.log_predictions(model, state.global_step)

    def log_predictions(self, model, step):
        device = model.device
        # print(inputs["inpu"])

        inputs = {
            'input_ids': self.fixed_batch['input_ids'].to(device),
            'attention_mask': self.fixed_batch['attention_mask'].to(device),
            **({'speaker_names': self.fixed_batch['speaker_names']} if "speaker_names" in self.fixed_batch else {}),  # no move needed
            'labels': self.fixed_batch['labels'].to(device)
        }

        model.eval()
        with torch.no_grad():
            outputs = model(
                **{**inputs, "labels" : None}
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
        model.train()

        table = wandb.Table(columns=["Step", "Masked Input", "Target Word", "Predicted Word"])

        if len(inputs.shape) == 2:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
            inputs["labels"] = inputs["labels"].unsqueeze(0)

        batch_size, dialog_len, seq_len = preds.shape

        for b in range(batch_size):
            for t in range(dialog_len):
                input_ids = inputs['input_ids'][b, t]       # (L,)
                labels = inputs['labels'][b, t]             # (L,)
                preds_b = preds[b, t]                       # (L,)

                # Replace with [MASK] token if label was masked
                masked_input_tokens = input_ids.clone()
                for i in range(seq_len):
                    if labels[i] == 103:
                        masked_input_tokens[i] = self.tokenizer.mask_token_id

                input_text_with_mask = self.tokenizer.decode(masked_input_tokens, skip_special_tokens=False)

                true_tokens = []
                pred_tokens = []
                for i in range(seq_len):
                    if labels[i] != -100:
                        true_tokens.append(self.tokenizer.decode([labels[i]]))
                        pred_tokens.append(self.tokenizer.decode([preds_b[i]]))

                table.add_data(
                    step,
                    input_text_with_mask.replace("[SEP]", "").replace("[PAD]", "").replace("[CLS]", ""),
                    ",".join(true_tokens),
                    ",".join(pred_tokens)
                )


        # batch_size, seq_len = preds.shape
        # for b in range(batch_size):
        #     input_ids = inputs['input_ids'][b]
        #     labels = inputs['labels'][b]
        #     preds_b = preds[b]

        #     masked_input_tokens = input_ids.clone()
        #     for i in range(seq_len):
        #         if labels[i] == 103:
        #             masked_input_tokens[i] = self.tokenizer.mask_token_id

        #     input_text_with_mask = self.tokenizer.decode(masked_input_tokens, skip_special_tokens=False)

        #     true_tokens = []
        #     pred_tokens = []
        #     for i in range(seq_len):
        #         if labels[i] != -100:
        #             true_tokens.append(self.tokenizer.decode([labels[i]]))
        #             pred_tokens.append(self.tokenizer.decode([preds_b[i]]))

        #     table.add_data(step, input_text_with_mask.replace("[SEP]", "").replace("[PAD]", "").replace("[CLS]", ""), ",".join(true_tokens), ",".join(pred_tokens))

        wandb.log({"MLM Predictions Evolution": table})
