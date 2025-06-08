"""
Configuration file for Deep Learning with Differential Privacy experiments.
Based on the paper: "Deep Learning with Differential Privacy" by Abadi et al.
"""

import numpy as np

# Dataset configuration
DATASET_CONFIG = {
    "name": "mnist",
    "num_classes": 10,
    "input_shape": (28, 28, 1),
    "batch_size": 250,  # As used in the paper
    "validation_split": 0.1,
}

# Model architecture configuration (CNN as in the paper)
MODEL_CONFIG = {
    "conv_layers": [
        {"filters": 16, "kernel_size": 8, "strides": 2, "activation": "tanh"},
        {"filters": 32, "kernel_size": 4, "strides": 2, "activation": "tanh"},
    ],
    "dense_layers": [32],  # Hidden layer size
    "dropout_rate": 0.25,
    "output_activation": None,  # Use None for logits (required for DP training)
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 60,  # As in the paper
    "learning_rate": 0.15,  # Initial learning rate from paper
    "momentum": 0.9,
    "verbose": 1,
}

# Differential Privacy configurations
# Different privacy budgets to test (epsilon, delta pairs)
DP_CONFIGS = [
    # Non-private baseline
    {
        "name": "non_private",
        "l2_norm_clip": None,
        "noise_multiplier": 0.0,
        "epsilon": float("inf"),
        "delta": 0.0,
        "microbatches": 1,
    },
    # Strong privacy
    {
        "name": "strong_privacy",
        "l2_norm_clip": 1.0,
        "noise_multiplier": 1.3,
        "epsilon": 2.0,
        "delta": 1e-5,
        "microbatches": 250,  # Same as batch_size for individual example gradients
    },
    # Medium privacy
    {
        "name": "medium_privacy",
        "l2_norm_clip": 1.0,
        "noise_multiplier": 1.1,
        "epsilon": 4.0,
        "delta": 1e-5,
        "microbatches": 250,
    },
    # Weak privacy
    {
        "name": "weak_privacy",
        "l2_norm_clip": 1.0,
        "noise_multiplier": 0.9,
        "epsilon": 8.0,
        "delta": 1e-5,
        "microbatches": 250,
    },
    # Very weak privacy
    {
        "name": "very_weak_privacy",
        "l2_norm_clip": 1.0,
        "noise_multiplier": 0.7,
        "epsilon": 16.0,
        "delta": 1e-5,
        "microbatches": 250,
    },
]

# Results and logging configuration
RESULTS_CONFIG = {
    "save_models": True,
    "model_dir": "saved_models",
    "results_dir": "results",
    "plots_dir": "plots",
    "log_file": "experiment_log.txt",
}

# Privacy accounting configuration
PRIVACY_CONFIG = {
    "target_delta": 1e-5,
    "noise_multipliers": np.arange(0.5, 2.0, 0.1),
    "l2_norm_clips": [0.5, 1.0, 1.5, 2.0],
    "sample_rate": None,  # Will be calculated as batch_size / num_train_examples
}
