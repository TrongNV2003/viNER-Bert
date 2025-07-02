from transformers import TrainerCallback
import torch

class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        self.memory_stats = {}

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2

            self.memory_stats = {
                # "allocated_mem_MB": round(allocated, 2),
                "GPU_reserved": round(reserved, 2),
                # "peak_mem_MB": round(peak, 2)
            }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs.update(self.memory_stats)
