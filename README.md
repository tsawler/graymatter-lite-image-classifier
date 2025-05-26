# Character Recognition Neural Network

A comprehensive Go-based neural network system for recognizing alphanumeric characters and punctuation marks using deep learning. This system can learn to identify 94 different character classes including uppercase letters (A-Z), lowercase letters (a-z), digits (0-9), and common punctuation marks.

## 🎯 What This System Does

This application trains a neural network to recognize individual characters from images. Think of it as teaching a computer to read characters the same way a human child learns the alphabet - by showing it thousands of examples until it can recognize patterns and identify new characters it hasn't seen before.

**Key Capabilities:**
- Recognizes 94 different character types (letters, digits, punctuation)
- Trains deep neural networks from scratch
- Automatically handles image preprocessing and resizing
- Provides comprehensive training visualizations
- Supports flexible data sampling for efficient development
- Saves and loads trained models for reuse

## 🧠 Machine Learning Concepts (For Go Developers)

If you're new to machine learning, here are the core concepts this system demonstrates:

### Neural Networks
A neural network is like a simplified model of how neurons in the brain work. It consists of layers of interconnected mathematical functions that process information and learn patterns from examples.

### Training Process
1. **Start with random "knowledge"** - The network begins knowing nothing
2. **Show examples** - Feed it thousands of character images with correct labels
3. **Make predictions** - Let it guess what each character is (wrong at first)
4. **Measure errors** - Calculate how wrong the predictions are
5. **Adjust parameters** - Tweak internal settings to reduce errors
6. **Repeat** - Continue until the network learns to recognize patterns

### Key Components
- **Forward Pass**: How the network makes predictions
- **Backpropagation**: How the network learns from mistakes
- **Cost Function**: Measures how wrong predictions are
- **Activation Functions**: Add non-linearity for complex pattern recognition

## 🏗️ Architecture Overview

The system uses a feedforward neural network with the following structure:

```
Input Layer (784 neurons) → Hidden Layer (128 neurons) → Hidden Layer (128 neurons) → Output Layer (94 neurons)
```

- **Input**: 28×28 grayscale images flattened to 784 pixel values
- **Hidden Layers**: Process and find patterns in the image data
- **Output**: 94 probabilities (one for each possible character)

## 📋 Prerequisites

- **Go 1.23 or later**
- **Git** (for cloning dependencies)
- **Python 3.13+** with matplotlib (for the plotting sidecar)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repository-url>
cd character-recognition
go mod init character-recognition
go mod tidy
```

### 2. Install Dependencies

The system uses the `github.com/tsawler/graymatter-lite` neural network library:

```bash
go get github.com/tsawler/graymatter-lite
```

### 3. Setup the Plotting Sidecar (Optional but Recommended)

For training visualizations, install the plotting service:

```bash
git clone https://github.com/tsawler/graymatter-sidecar
cd graymatter-sidecar
pip install -r requirements.txt
python app.py
```

Or, run it in Docker:

```bash
git clone https://github.com/tsawler/graymatter-sidecar
cd graymatter-sidecar
docker-compose up --build -d
```

The sidecar runs on `http://localhost:8080` and provides comprehensive training plots.

### 4. Prepare Your Data

Organize your training images in this directory structure:

```
data/
├── upper/          # Uppercase letters
│   ├── A/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── B/
│   └── ...
├── lower/          # Lowercase letters
│   ├── a/
│   ├── b/
│   └── ...
├── digits/         # Numbers
│   ├── 0/
│   ├── 1/
│   └── ...
└── punctuation/    # Punctuation marks
    ├── asterisk/   # For * symbol
    ├── dot/        # For . symbol
    ├── question/   # For ? symbol
    └── ...
```

**Important Notes:**
- All images should be approximately 28×28 pixels (automatic resizing is supported)
- Images can be PNG or JPEG format
- Each directory should contain multiple examples (100+ recommended per character)

### 5. Run the System

**Basic training with all available data:**
```bash
go run .
```

**Training with data sampling (for faster development):**
```bash
# Use 100 samples per character class
go run . -samples=100 -iterations=200

# Use 10 samples per character (very fast for testing)
go run . -samples=10 -iterations=50
```

**Load an existing model:**
```bash
go run . -load=./image_classifier_final.json
```

**Full parameter control:**
```bash
go run . -samples=500 -iterations=1000 -lr=0.001 -batchsize=32
```

## 🎛️ Command Line Options

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `-samples` | Images per character class (0 = all) | 0 | `-samples=100` |
| `-iterations` | Training epochs | 30 | `-iterations=500` |
| `-lr` | Learning rate | 0.001 | `-lr=0.01` |
| `-batchsize` | Batch size | 64 | `-batchsize=32` |
| `-predict` | Image file for prediction | "a.png" | `-predict=test.jpg` |
| `-load` | Load existing model | "" | `-load=model.json` |

## 📊 Data Sampling Feature

One of the key features is **flexible data sampling** that dramatically improves development efficiency:

### Why Use Sampling?

When you have thousands of images per character, training on the full dataset can take hours. Sampling lets you:
- **Rapid prototyping**: Test changes in seconds with small samples
- **Parameter tuning**: Find good settings with medium samples
- **Final training**: Use full dataset for production models

### Sampling Strategies

```bash
# Phase 1: Architecture testing (very fast)
go run . -samples=10 -iterations=50

# Phase 2: Hyperparameter tuning (fast)
go run . -samples=100 -iterations=200

# Phase 3: Model validation (moderate)
go run . -samples=1000 -iterations=500

# Phase 4: Production training (full quality)
go run . -samples=0 -iterations=1000
```

### Performance Impact

| Sample Size | Training Time | Expected Accuracy |
|-------------|---------------|-------------------|
| 10/class | ~1 minute | 70-80% of full |
| 100/class | ~10 minutes | 85-90% of full |
| 1000/class | ~1 hour | 95-98% of full |
| All data | ~3+ hours | 100% (maximum) |

## 🔧 Configuration

The system uses a configuration-driven approach. Key settings in `config.go`:

```go
config := &Config{
    ImageWidth:      28,    // Input image dimensions
    ImageHeight:     28,
    InputSize:       784,   // 28 × 28 pixels
    OutputSize:      94,    // Character classes
    HiddenSize:      128,   // Neurons per hidden layer
    SamplesPerClass: 0,     // Data sampling (0 = all)
    DataDir:         "data",
    ModelPath:       "image_classifier",
}
```

## 📈 Understanding Training Output

During training, you'll see output like:

```
Loading training data...
Loaded 15000 training samples
Training samples: 12000
Validation samples: 3000

Training network with 500 iterations, batch size 32, learning rate 0.001000...

Epoch 50/500 [████████░░] 20.0% | Cost: 0.234567 | Train Acc: 0.891 | Val Acc: 0.875 | ETA: 08:45

Training complete!
Final training cost: 0.123456
```

**Key Metrics:**
- **Cost**: Lower is better (measures prediction errors)
- **Train Acc**: Accuracy on training data
- **Val Acc**: Accuracy on validation data (more important!)
- **ETA**: Estimated time remaining

## 🎨 Visualization Features

When the plotting sidecar is running, the system generates:

1. **Training Curves**: Cost and accuracy over time
2. **Network Architecture**: Visual representation of the neural network
3. **Weight Distributions**: Health analysis of learned parameters
4. **Confusion Matrix**: Which characters are confused with each other

## 💾 Model Persistence

The system automatically saves trained models:

- `image_classifier_final.json`: Complete trained model
- `image_classifier_best.json`: Best performing checkpoint
- `image_classifier_epoch_N.json`: Regular training checkpoints

**Loading Models:**
```bash
# Load specific model
go run . -load=./image_classifier_best.json

# System automatically loads image_classifier_final.json if it exists
go run .
```

## 🔍 Making Predictions

Once trained, test the model on new images:

```bash
# Predict on default test image
go run . -predict=test_image.png

# Load existing model and predict
go run . -load=trained_model.json -predict=my_character.jpg
```

The system will output:
```
Predicted character: 'A'
Confidence: 94.56%
Assessment: Very confident prediction
Character type: Uppercase letter
```

## 🏭 Production Deployment

For production use:

1. **Train final model with full dataset:**
   ```bash
   go run . -samples=0 -iterations=1000 -lr=0.001
   ```

2. **Create prediction-only service:**
   ```go
   // Load trained model
   classifier, metadata, err := LoadModelForInference("production_model.json")
   
   // Make predictions
   prediction, confidence, err := classifier.Predict("user_image.png")
   ```

3. **Deploy with appropriate scaling** based on prediction volume

## 🔬 Technical Deep Dive

### Neural Network Implementation

The system uses the `graymatter-lite` library which provides:

- **Automatic differentiation**: Computes gradients for backpropagation
- **Multiple activation functions**: ReLU, Sigmoid, Softmax
- **Various cost functions**: MSE, Cross-entropy, Categorical cross-entropy
- **Progress tracking**: Real-time training monitoring
- **Model serialization**: Save/load functionality

### Image Processing Pipeline

1. **Load image** (PNG/JPEG support)
2. **Resize to 28×28** (maintains aspect ratio)
3. **Convert to grayscale** using luminance formula
4. **Normalize pixel values** to 0.0-1.0 range
5. **Flatten to 784-element array** for neural network input
6. **Auto-detect and invert polarity** if needed

### Data Flow

```
Raw Image → Preprocessing → Neural Network → Probabilities → Character Prediction
     ↓            ↓             ↓              ↓              ↓
  Various     28×28 pixels   Forward Pass    [0.02, 0.95,   Argmax:
  formats     normalized     784→128→128→94   0.01, ...]     'B'
```

## 📚 Dependencies and Credits

This project builds upon several excellent libraries:

### Core Dependencies
- **[github.com/tsawler/graymatter-lite](https://github.com/tsawler/graymatter-lite)**: Neural network engine
  - Provides the core neural network implementation
  - Handles training, prediction, and model persistence
  - Created by Trevor Sawler for educational use

### Supporting Libraries
- **[gonum.org/v1/gonum](https://gonum.org/)**: Matrix operations and linear algebra
- **Standard Go libraries**: Image processing, HTTP client, JSON serialization

### Visualization Sidecar
- **[github.com/tsawler/graymatter-sidecar](https://github.com/tsawler/graymatter-sidecar)**: Plotting service
  - Python-based visualization service
  - Generates training curves, confusion matrices, and network diagrams
  - Uses matplotlib and seaborn for high-quality plots

## 🤝 Contributing

This is an educational project demonstrating neural network concepts in Go. Contributions welcome:

1. **Bug fixes**: Improve reliability and error handling
2. **Performance optimizations**: Faster training or prediction
3. **Additional features**: New activation functions, regularization techniques
4. **Documentation**: Better explanations or examples
5. **Testing**: Unit tests and integration tests

## 📖 Learning Resources

If you want to understand the machine learning concepts deeper:

### Books
- "Deep Learning" by Ian Goodfellow (comprehensive technical reference)
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### Online Courses
- CS231n: Convolutional Neural Networks (Stanford)
- Machine Learning Course (Andrew Ng, Coursera)
- Fast.ai Practical Deep Learning

### Key Concepts to Explore
- **Gradient Descent**: How networks learn through optimization
- **Backpropagation**: The algorithm that makes learning possible
- **Regularization**: Techniques to prevent overfitting
- **Activation Functions**: How to add non-linearity to linear models

## 🔧 Troubleshooting

### Common Issues

**"No training data found"**
- Check that your `data/` directory structure matches the expected format
- Ensure image files are in PNG or JPEG format
- Verify directory names match exactly (case-sensitive)

**"Plotting service unavailable"**
- Make sure the plotting sidecar is running on `http://localhost:8080`
- Install required Python dependencies: `pip install matplotlib flask`
- Training will continue without plots if sidecar is unavailable

**"Network validation failed"**
- Usually indicates dimension mismatches in saved models
- Try training a new model from scratch
- Check that your data preprocessing matches the model's expectations

**Low training accuracy**
- Increase training iterations: `-iterations=1000`
- Adjust learning rate: `-lr=0.01` (higher) or `-lr=0.0001` (lower)
- Ensure sufficient training data per class
- Check that input images are clear and properly labeled

**Training takes too long**
- Use data sampling: `-samples=100` for faster iteration
- Reduce batch size: `-batchsize=16`
- Start with fewer training iterations for testing

### Performance Tips

1. **Use sampling for development**: Start small, scale up
2. **Monitor validation accuracy**: More important than training accuracy
3. **Save checkpoints**: Enable model saving to avoid losing progress
4. **Visualize training**: Use plotting sidecar to debug issues
5. **Balanced data**: Ensure similar numbers of examples per character

## 📝 License

This project is released under the MIT License. See `LICENSE.md` for details.

The MIT License allows you to freely use, modify, and distribute this software for both commercial and non-commercial purposes.

## 🙏 Acknowledgments

- **The Go Team** for creating an excellent language for systems programming
- **The Gonum Project** for providing high-quality numerical computing libraries
- **The machine learning community** for developing the algorithms and techniques implemented here

---

**Happy Learning!** 🚀

This project demonstrates that machine learning doesn't require complex frameworks - with understanding of the fundamentals and good engineering practices, you can build powerful systems in any language. The combination of Go's performance and simplicity with well-designed neural network libraries creates an excellent platform for learning and production use.