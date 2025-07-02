from transformers import TrainerCallback
import torch
import time

class ThroughputLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.total_tokens = 0
        self.last_log_time = None
        self.last_log_step = None
        self._last_input_ids = None

    def on_step_begin(self, args, state, control, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs and "input_ids" in inputs:
            self._last_input_ids = inputs["input_ids"]

    def on_step_end(self, args, state, control, **kwargs):
        if self._last_input_ids is not None:
            input_ids = self._last_input_ids
            if isinstance(input_ids, torch.Tensor):
                non_pad_tokens = (input_ids != self.pad_token_id).sum().item()
            else:
                non_pad_tokens = sum(len([tok for tok in ex if tok != self.pad_token_id]) for ex in input_ids)
            self.total_tokens += non_pad_tokens
            self._last_input_ids = None  # Clear after use

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_time = time.time()
        if self.last_log_time is not None and self.last_log_step is not None:
            time_diff = current_time - self.last_log_time
            step_diff = state.global_step - self.last_log_step

            if time_diff > 0 and step_diff > 0:
                tokens_per_sec = self.total_tokens / time_diff
                logs["tokens_per_second"] = round(tokens_per_sec, 2)

        self.last_log_time = current_time
        self.last_log_step = state.global_step
        self.total_tokens = 0
