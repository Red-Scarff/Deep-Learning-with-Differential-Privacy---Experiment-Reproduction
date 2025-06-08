"""
Differential Privacy training implementation using TensorFlow Privacy.
Implements DP-SGD training with DPKerasSGDOptimizer.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional
import json

# TensorFlow Privacy imports
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from models import create_dp_cnn_model
from config import TRAINING_CONFIG, DATASET_CONFIG


class DPTrainer:
    """Differential Privacy trainer using TensorFlow Privacy."""

    def __init__(self, dp_config: Dict, save_dir: str = "results"):
        """
        Initialize DP trainer.

        Args:
            dp_config: Differential privacy configuration
            save_dir: Directory to save results
        """
        self.dp_config = dp_config
        self.save_dir = save_dir
        self.model = None
        self.history = None

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        print(f"Initialized DP Trainer for: {dp_config['name']}")
        print(f"Privacy settings:")
        print(f"  - l2_norm_clip: {dp_config.get('l2_norm_clip', 'None')}")
        print(f"  - noise_multiplier: {dp_config['noise_multiplier']}")
        print(f"  - target epsilon: {dp_config['epsilon']}")
        print(f"  - target delta: {dp_config['delta']}")

    def create_optimizer(self, num_train_examples: int) -> keras.optimizers.Optimizer:
        """
        Create optimizer (DP or regular SGD).

        Args:
            num_train_examples: Number of training examples for privacy accounting

        Returns:
            Optimizer instance
        """
        learning_rate = TRAINING_CONFIG["learning_rate"]
        momentum = TRAINING_CONFIG["momentum"]

        if self.dp_config["noise_multiplier"] > 0:
            # Create DP-SGD optimizer
            print("Creating DP-SGD optimizer...")

            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=self.dp_config["l2_norm_clip"],
                noise_multiplier=self.dp_config["noise_multiplier"],
                num_microbatches=self.dp_config["microbatches"],
                learning_rate=learning_rate,
                momentum=momentum,
            )

            print(f"DP-SGD optimizer created:")
            print(f"  - l2_norm_clip: {self.dp_config['l2_norm_clip']}")
            print(f"  - noise_multiplier: {self.dp_config['noise_multiplier']}")
            print(f"  - num_microbatches: {self.dp_config['microbatches']}")

        else:
            # Create regular SGD optimizer for non-private baseline
            print("Creating regular SGD optimizer (non-private)...")
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        return optimizer

    def compute_privacy_spent(self, num_train_examples: int, epochs: int, batch_size: int) -> Dict[str, float]:
        """
        Compute privacy spent using TensorFlow Privacy.

        Args:
            num_train_examples: Number of training examples
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with epsilon and delta values
        """
        if self.dp_config["noise_multiplier"] == 0:
            # Non-private case
            return {"epsilon": float("inf"), "delta": 0.0, "steps": epochs * (num_train_examples // batch_size)}

        # Calculate number of steps
        steps = epochs * (num_train_examples // batch_size)

        # Compute privacy spent
        try:
            # Use the correct function name for TensorFlow Privacy 0.9.0
            epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
                n=num_train_examples,
                batch_size=batch_size,
                noise_multiplier=self.dp_config["noise_multiplier"],
                epochs=epochs,
                delta=self.dp_config["delta"],
            )
        except Exception as e:
            print(f"Warning: Could not compute privacy spent: {e}")
            epsilon = float("inf")

        return {"epsilon": epsilon, "delta": self.dp_config["delta"], "steps": steps}

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Train model with differential privacy.

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data (optional)
            x_test, y_test: Test data (optional)

        Returns:
            Training results dictionary
        """
        print(f"\nStarting training for: {self.dp_config['name']}")
        print(f"Training data shape: {x_train.shape}")

        # Create model
        self.model = create_dp_cnn_model()

        # Create optimizer
        optimizer = self.create_optimizer(len(x_train))

        # Compile model with proper loss for DP training
        if self.dp_config["noise_multiplier"] > 0:
            # For DP training, use vector loss (not reduced)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        else:
            # For non-private training, use regular loss with logits
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        print(f"Model compiled with {type(optimizer).__name__}")

        # Prepare validation data
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)

        # Training parameters
        epochs = TRAINING_CONFIG["epochs"]
        batch_size = DATASET_CONFIG["batch_size"]

        print(f"Training parameters:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {TRAINING_CONFIG['learning_rate']}")

        # Start training
        start_time = time.time()

        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=TRAINING_CONFIG["verbose"],
            shuffle=True,
        )

        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Compute privacy spent
        privacy_spent = self.compute_privacy_spent(len(x_train), epochs, batch_size)

        print(f"Privacy spent: ε = {privacy_spent['epsilon']:.3f}, δ = {privacy_spent['delta']}")

        # Evaluate model
        results = self._evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test)

        # Compile final results
        final_results = {
            "config_name": self.dp_config["name"],
            "dp_config": self.dp_config,
            "privacy_spent": privacy_spent,
            "training_time": training_time,
            "final_train_accuracy": results["train_accuracy"],
            "final_val_accuracy": results["val_accuracy"],
            "test_accuracy": results["test_accuracy"],
            "history": {
                "loss": self.history.history["loss"],
                "accuracy": self.history.history["accuracy"],
                "val_loss": self.history.history.get("val_loss", []),
                "val_accuracy": self.history.history.get("val_accuracy", []),
            },
        }

        # Save results
        self._save_results(final_results)

        return final_results

    def _evaluate_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        x_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
    ) -> Dict:
        """Evaluate model on all datasets."""
        results = {}

        # Training accuracy
        train_loss, train_acc = self.model.evaluate(x_train, y_train, verbose=0)
        results["train_accuracy"] = train_acc
        print(f"Final training accuracy: {train_acc:.4f}")

        # Validation accuracy
        if x_val is not None and y_val is not None:
            val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=0)
            results["val_accuracy"] = val_acc
            print(f"Final validation accuracy: {val_acc:.4f}")
        else:
            results["val_accuracy"] = None

        # Test accuracy
        if x_test is not None and y_test is not None:
            test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
            results["test_accuracy"] = test_acc
            print(f"Final test accuracy: {test_acc:.4f}")
        else:
            results["test_accuracy"] = None

        return results

    def _save_results(self, results: Dict):
        """Save training results to file."""
        filename = f"{self.dp_config['name']}_results.json"
        filepath = os.path.join(self.save_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.floating):
                        serializable_results[key][k] = [float(x) for x in v]
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    # Test DP trainer
    from data_loader import load_mnist_data
    from config import DP_CONFIGS

    print("Testing DP Trainer...")

    # Load small subset for testing
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

    # Use small subset for quick test
    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]
    x_test_small = x_test[:200]
    y_test_small = y_test[:200]

    # Test with medium privacy config
    test_config = DP_CONFIGS[2].copy()  # medium_privacy

    # Override training config for quick test
    import config

    config.TRAINING_CONFIG["epochs"] = 2
    config.DATASET_CONFIG["batch_size"] = 50
    test_config["microbatches"] = 50  # Must equal batch_size

    trainer = DPTrainer(test_config, save_dir="test_results")
    results = trainer.train_model(x_train_small, y_train_small, x_test=x_test_small, y_test=y_test_small)

    print(f"Test completed! Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Privacy spent: ε = {results['privacy_spent']['epsilon']:.3f}")
