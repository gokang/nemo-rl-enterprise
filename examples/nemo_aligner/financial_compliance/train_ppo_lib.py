# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Enterprise PPO Library for Financial Alignment

import torch
from nemo_aligner.utils.train_utils import setup_trainer
from nemo_aligner.models.nlp.gpt.gpt_ppo_model import GPTPPOModel
from nemo_aligner.data.nlp.ppo_dataset import PPODataset

class FinancialPPOModel(GPTPPOModel):
    """
    Extends NeMo's GPTPPOModel with enterprise-specific safety logging.
    """
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        # Custom logging for reward stability during financial alignment
        if batch_idx % 10 == 0:
            rewards = outputs.get('rewards')
            if rewards is not None:
                mean_reward = torch.mean(rewards)
                print(f"[Enterprise Log] Step {batch_idx}: Mean Compliance Reward = {mean_reward:.4f}")

def launch_training(cfg):
    """
    Initializes the trainer and model, then begins the PPO loop.
    """
    trainer = setup_trainer(cfg)
    
    # In a real scenario, weights would be loaded via cfg.model.restore_from_path
    model = FinancialPPOModel(cfg.model, trainer=trainer)
    
    trainer.fit(model)
