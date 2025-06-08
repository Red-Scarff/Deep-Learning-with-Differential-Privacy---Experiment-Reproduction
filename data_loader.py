"""
Data loading and preprocessing for MNIST dataset.
Implements the data preparation as described in the DP paper.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict
from config import DATASET_CONFIG


class MNISTDataLoader:
    """Data loader for MNIST dataset with preprocessing for DP training."""
    
    def __init__(self, validation_split: float = 0.1):
        self.validation_split = validation_split
        self.num_classes = DATASET_CONFIG['num_classes']
        
    def load_and_preprocess(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                         Tuple[np.ndarray, np.ndarray], 
                                         Tuple[np.ndarray, np.ndarray]]:
        """
        Load and preprocess MNIST data.
        
        Returns:
            train_data: (x_train, y_train)
            val_data: (x_val, y_val) 
            test_data: (x_test, y_test)
        """
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Split training data into train and validation
        if self.validation_split > 0:
            val_size = int(len(x_train) * self.validation_split)
            
            # Shuffle before splitting
            indices = np.random.permutation(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            # Split
            x_val = x_train[:val_size]
            y_val = y_train[:val_size]
            x_train = x_train[val_size:]
            y_train = y_train[val_size:]
        else:
            x_val, y_val = None, None
            
        print(f"Training samples: {len(x_train)}")
        if x_val is not None:
            print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def create_tf_dataset(self, x: np.ndarray, y: np.ndarray, 
                         batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Args:
            x: Input features
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(x))
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_data_info(self, x_train: np.ndarray) -> Dict[str, int]:
        """Get dataset information for privacy accounting."""
        return {
            'num_train_examples': len(x_train),
            'input_shape': x_train.shape[1:],
            'num_classes': self.num_classes
        }


def load_mnist_data():
    """Convenience function to load MNIST data."""
    loader = MNISTDataLoader(validation_split=DATASET_CONFIG['validation_split'])
    return loader.load_and_preprocess()


if __name__ == "__main__":
    # Test data loading
    print("Testing MNIST data loading...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
    
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    if x_val is not None:
        print(f"Validation shape: {x_val.shape}, {y_val.shape}")
    print(f"Test shape: {x_test.shape}, {y_test.shape}")
    
    print(f"Data range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print("Data loading test completed successfully!")
