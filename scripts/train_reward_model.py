# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Main entry point for training the Reward Model (The Compliance Judge)

from nemo.core.config import hydra_runner
from nemo_aligner.utils.train_utils import setup_trainer
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel

@hydra_runner(config_path="configs", config_name="reward_model_config")
def main(cfg):
    """
    Train a Reward Model on financial preference data.
    This model will later provide the reward signal for PPO.
    """
    trainer = setup_trainer(cfg)
    
    # Initialize Reward Model
    # It learns a scalar output representing 'compliance' quality
    model = GPTRewardModel(cfg.model, trainer=trainer)

    print("--- ⚖️ Training Enterprise Reward Model (Compliance Judge) ---")
    trainer.fit(model)

if __name__ == "__main__":
    main()
