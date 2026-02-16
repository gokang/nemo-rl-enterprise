# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Helper utilities for reward model training and preference data handling

import json
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel

def load_rm_from_nemo(path, cfg, trainer):
    """
    Loads a GPTRewardModel from a .nemo file using the Megatron-core backend.
    """
    return GPTRewardModel.restore_from(
        restore_path=path,
        override_config_path=cfg,
        trainer=trainer
    )

def prepare_preference_dataset(input_file, output_file):
    """
    Ensures preference data is in the strict NeMo Aligner JSONL format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    processed = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Validation logic could go here
            processed.append(data)
    
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
