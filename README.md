# Deep Learning with Differential Privacy - Experiment Reproduction

This project reproduces the experiments from the paper "Deep Learning with Differential Privacy" by Abadi et al. using TensorFlow Privacy.

## 🎯 Project Overview

This implementation provides:
- **DP-SGD Training**: Using TensorFlow Privacy's `DPKerasSGDOptimizer`
- **CNN Model**: Reproducing the paper's CNN architecture for MNIST
- **Privacy Analysis**: Computing privacy budgets (ε, δ) for different configurations
- **Comprehensive Evaluation**: Multiple privacy settings with accuracy measurements
- **Publication-Quality Visualizations**: Privacy-accuracy trade-off plots

## 📁 Project Structure

```
privacy/
├── config.py              # Experiment configurations and hyperparameters
├── data_loader.py          # MNIST data loading and preprocessing
├── models.py              # CNN model architecture
├── dp_training.py         # Differential privacy training logic
├── run_experiments.py     # Main experiment runner
├── visualization.py       # Results visualization and plotting
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Experiments

```bash
python run_experiments.py
```

This will run experiments with different privacy budgets:
- Non-private baseline
- Strong privacy (ε=2.0)
- Medium privacy (ε=4.0) 
- Weak privacy (ε=8.0)
- Very weak privacy (ε=16.0)

### 3. Generate Visualizations

```bash
python visualization.py
```

### 4. Run Single Experiment

```bash
python run_experiments.py --config strong_privacy
```

## 📊 Expected Results

The experiments will generate:

1. **Privacy-Accuracy Trade-off Plot**: Shows how test accuracy decreases as privacy increases
2. **Training Curves**: Training/validation accuracy over epochs for each privacy setting
3. **Noise Impact Analysis**: Effect of noise multiplier on model performance
4. **Summary Table**: Comprehensive results table with all metrics

## 🔧 Configuration

### Privacy Settings

Edit `config.py` to modify privacy parameters:

```python
DP_CONFIGS = [
    {
        'name': 'custom_privacy',
        'l2_norm_clip': 1.0,           # Gradient clipping threshold
        'noise_multiplier': 1.2,       # Noise level
        'epsilon': 3.0,                # Target privacy budget
        'delta': 1e-5,                 # Privacy parameter
        'microbatches': 250            # Number of microbatches
    }
]
```

### Model Architecture

The CNN model follows the paper specification:
- Conv1: 16 filters, 8×8 kernel, stride 2, tanh activation
- Conv2: 32 filters, 4×4 kernel, stride 2, tanh activation  
- Dense: 32 units, tanh activation
- Dropout: 0.25
- Output: 10 classes, softmax activation

### Training Parameters

- **Batch Size**: 250 (as in paper)
- **Epochs**: 60
- **Learning Rate**: 0.15
- **Momentum**: 0.9

## 📈 Understanding Results

### Privacy Budget (ε, δ)
- **Lower ε**: Stronger privacy, typically lower accuracy
- **Higher ε**: Weaker privacy, typically higher accuracy
- **δ**: Probability of privacy breach (usually 1e-5)

### Key Metrics
- **Test Accuracy**: Final model performance on test set
- **Privacy Spent**: Actual (ε, δ) computed by TF Privacy
- **Training Time**: Time to complete training

## 🔬 Experiment Details

### Differential Privacy Implementation
- Uses `DPKerasSGDOptimizer` from TensorFlow Privacy
- Implements per-example gradient clipping
- Adds calibrated Gaussian noise to gradients
- Computes privacy spent using RDP analysis

### Privacy Accounting
- Uses Rényi Differential Privacy (RDP) for tight analysis
- Converts RDP to (ε, δ)-DP for interpretability
- Accounts for composition over training steps

## 📋 Results Interpretation

Expected accuracy ranges (approximate):
- **Non-private**: ~99% test accuracy
- **ε=16**: ~97-98% test accuracy  
- **ε=8**: ~95-97% test accuracy
- **ε=4**: ~92-95% test accuracy
- **ε=2**: ~85-92% test accuracy

## 🛠 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size in `config.py`
2. **Slow Training**: Reduce number of epochs for testing
3. **Import Errors**: Ensure TensorFlow Privacy is installed correctly

### Performance Tips

- Use GPU for faster training: `pip install tensorflow-gpu`
- Reduce dataset size for quick testing
- Adjust microbatch size based on available memory

## 📚 References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) - Abadi et al.
- [TensorFlow Privacy](https://github.com/tensorflow/privacy) - Official implementation
- [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy) - Background

## 🤝 Contributing

Feel free to:
- Add new model architectures
- Implement additional datasets
- Improve visualization plots
- Add more privacy analysis tools

## 📄 License

This project is for educational and research purposes, reproducing published academic work.
