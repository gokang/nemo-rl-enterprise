# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Real-world training entry point using NeMo Aligner PPO API

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo_aligner.models.nlp.gpt.gpt_ppo_model import GPTPPOModel
from nemo_aligner.utils.train_utils import setup_trainer

@hydra_runner(config_path="configs", config_name="ppo_config")
def main(cfg):
    """
    Launches the PPO alignment training for the financial advisor.
    This script utilizes the NeMo Aligner library for distributed RLHF.
    """
    # 1. Setup Distributed Trainer (Multi-GPU/Node)
    trainer = setup_trainer(cfg)

    # 2. Initialize PPO Model from Config
    # Uses NeMo's save/restore connector to load sharded weights
    model = GPTPPOModel(
        cfg.model, 
        trainer=trainer, 
        save_restore_connector=NLPSaveRestoreConnector()
    )

    print("--- 🚀 Starting Enterprise Financial Compliance PPO Alignment ---")
    
    # 3. Launch the RLHF Training Loop
    # This automatically handles rollouts, advantage estimation, and policy updates
    trainer.fit(model)

if __name__ == "__main__":
    main()
