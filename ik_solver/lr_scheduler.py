import torch.optim as optim  # For optimizers and schedulers
import numpy as np  # If you use numpy for metric calculations in the manager



class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = min(self.current_step / self.warmup_steps, 1.0)
        new_lr = self.target_lr * progress
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class LRSchedulerManager:
    def __init__(self, optimizers, initial_lr, warmup_steps=1000, min_lr=1e-6):
        self.optimizers = optimizers
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0

        # Create schedulers for each optimizer
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append({
                'warmup': LinearWarmupScheduler(optimizer, warmup_steps, initial_lr),
                'plateau': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=100, min_lr=min_lr)
            })

    def step(self, metrics=None):
        self.step_count += 1
        # Use warmup scheduler if in warmup period
        if self.step_count < self.warmup_steps:
            for scheduler in self.schedulers:
                scheduler['warmup'].step()
        # Switch to plateau scheduler after warmup
        elif metrics is not None:
            for scheduler in self.schedulers:
                scheduler['plateau'].step(metrics)

    def get_last_lr(self):
        """Get current learning rates for all optimizers"""
        return [[param_group['lr'] for param_group in optimizer.param_groups] for optimizer in self.optimizers]
