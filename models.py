"""
Model architectures for differential privacy experiments.
Implements the CNN model from "Deep Learning with Differential Privacy" paper.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
from config import MODEL_CONFIG, DATASET_CONFIG


class DPCNNModel:
    """CNN model for differential privacy experiments."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = None, 
                 num_classes: int = None):
        """
        Initialize the DP CNN model.
        
        Args:
            input_shape: Shape of input data (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape or DATASET_CONFIG['input_shape']
        self.num_classes = num_classes or DATASET_CONFIG['num_classes']
        
    def build_model(self) -> keras.Model:
        """
        Build the CNN model as described in the DP paper.
        
        Architecture:
        - Conv layer 1: 16 filters, 8x8 kernel, stride 2, tanh activation
        - Conv layer 2: 32 filters, 4x4 kernel, stride 2, tanh activation  
        - Flatten
        - Dense layer: 32 units, tanh activation
        - Dropout: 0.25
        - Output layer: num_classes units, softmax activation
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional layer
            layers.Conv2D(
                filters=MODEL_CONFIG['conv_layers'][0]['filters'],
                kernel_size=MODEL_CONFIG['conv_layers'][0]['kernel_size'],
                strides=MODEL_CONFIG['conv_layers'][0]['strides'],
                activation=MODEL_CONFIG['conv_layers'][0]['activation'],
                padding='same',
                name='conv1'
            ),
            
            # Second convolutional layer
            layers.Conv2D(
                filters=MODEL_CONFIG['conv_layers'][1]['filters'],
                kernel_size=MODEL_CONFIG['conv_layers'][1]['kernel_size'],
                strides=MODEL_CONFIG['conv_layers'][1]['strides'],
                activation=MODEL_CONFIG['conv_layers'][1]['activation'],
                padding='same',
                name='conv2'
            ),
            
            # Flatten for dense layers
            layers.Flatten(),
            
            # Dense hidden layer
            layers.Dense(
                MODEL_CONFIG['dense_layers'][0],
                activation='tanh',
                name='dense1'
            ),
            
            # Dropout for regularization
            layers.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # Output layer
            layers.Dense(
                self.num_classes,
                activation=MODEL_CONFIG['output_activation'],
                name='output'
            )
        ])
        
        return model
    
    def compile_model(self, model: keras.Model, optimizer, 
                     loss: str = 'categorical_crossentropy',
                     metrics: list = None) -> keras.Model:
        """
        Compile the model with given optimizer and loss.
        
        Args:
            model: Keras model to compile
            optimizer: Optimizer instance (can be DP or regular)
            loss: Loss function name
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
            
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def get_model_summary(self, model: keras.Model) -> str:
        """Get model architecture summary."""
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()


def create_dp_cnn_model(input_shape: Tuple[int, int, int] = None,
                       num_classes: int = None) -> keras.Model:
    """
    Convenience function to create a DP CNN model.
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of output classes
        
    Returns:
        Uncompiled Keras model
    """
    model_builder = DPCNNModel(input_shape, num_classes)
    return model_builder.build_model()


if __name__ == "__main__":
    # Test model creation
    print("Testing DP CNN model creation...")
    
    model = create_dp_cnn_model()
    print(f"Model created with input shape: {DATASET_CONFIG['input_shape']}")
    print(f"Number of classes: {DATASET_CONFIG['num_classes']}")
    
    # Print model summary
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    print("Model creation test completed successfully!")
