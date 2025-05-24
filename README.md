# Character Recognition Neural Network

A Go program that trains a neural network to recognize handwritten characters (A-Z, a-z, 0-9) using machine learning techniques.

## What This Program Does

Imagine you want to build a system that can look at an image of a handwritten letter and automatically tell you whether it's an "A", "B", "7", etc. This is exactly what this program does! It's similar to how:

- Postal services automatically read ZIP codes on mail
- Banks process handwritten amounts on checks  
- Document scanning apps convert handwritten notes to text
- Captcha systems verify you're human by asking you to identify distorted characters

## Machine Learning in Simple Terms

**Traditional Programming**: You write explicit rules
```
if (pixel pattern looks like two vertical lines with horizontal lines) {
    return "H"
}
```

**Machine Learning**: You provide examples and let the computer figure out the rules
```
Show computer 1000 images labeled "H"
Show computer 1000 images labeled "A" 
Show computer 1000 images labeled "7"
... (computer learns patterns automatically)
Now computer can recognize H, A, 7 in new images it's never seen
```

This program uses a **neural network** - a type of machine learning inspired by how brain neurons work. The network learns by studying thousands of example images and gradually adjusting its internal parameters to recognize character patterns.

## Prerequisites

### Required Software
- **Go 1.19+**: The programming language this is written in
- **Python 3.8+**: For the plotting service that generates training charts
- **Git**: To clone the repository

### Required Go Module
This program depends on the `graymatter-lite` neural network library:
```bash
go mod tidy  # This will automatically download the required module
```

### Training Data
You need organized image data in this structure:
```
data/
├── upper/          # Uppercase letters
│   ├── A/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── B/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
├── lower/          # Lowercase letters  
│   ├── a/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
└── digits/         # Numbers
    ├── 0/
    │   ├── image1.png
    │   └── ...
    └── ...
```

**Image Requirements:**
- Format: PNG or JPEG
- Size: Will be automatically resized to 28×28 pixels
- Content: Single character, roughly centered
- Quality: Clear, high-contrast images work best

## How Neural Networks Work (Simplified)

Think of a neural network like a very sophisticated pattern recognition system made of interconnected "neurons" (mathematical functions):

### 1. Input Layer (784 neurons)
Takes your 28×28 pixel image and converts it to 784 numbers (one per pixel). Black pixels become 0.0, white pixels become 1.0, gray pixels become values in between.

### 2. Hidden Layers (128 neurons each)
These layers detect patterns:
- **First hidden layer**: Might learn to detect edges, curves, and basic shapes
- **Second hidden layer**: Might learn to combine edges into letter parts (like the top of an "A" or the curve of an "O")

### 3. Output Layer (62 neurons)
One neuron for each possible character (26 uppercase + 26 lowercase + 10 digits). Each outputs a probability:
- High value (0.9): "I'm 90% sure this is an A"
- Low value (0.1): "I'm only 10% sure this is a B"

### Learning Process
1. **Forward Pass**: Image goes through the network, produces a guess
2. **Error Calculation**: Compare guess to correct answer, measure how wrong it was
3. **Backward Pass**: Adjust all the internal weights to reduce the error
4. **Repeat**: Do this thousands of times with thousands of examples

This process is called **backpropagation** - the network literally learns from its mistakes!

## Installation & Setup

### 1. Clone and Build
```bash
git clone <repository-url>
cd character-recognition
go mod tidy
go build
```

### 2. Set Up Plotting Service (Optional but Recommended)
The program can generate helpful training visualizations if you set up the plotting service:

```bash
# Install Python dependencies (matplotlib, flask, etc.)
pip install -r plotting-requirements.txt  # If provided

# Start the plotting service on port 8080
python plotting-service.py &
```

If you skip this step, disable plotting in the config:
```go
TrainingOptions: graymatter.TrainingOptions{
    EnablePlotting: false,  // Set to false
    // ... other options
}
```

### 3. Prepare Training Data
Organize your character images according to the directory structure shown above. You need:
- **Minimum**: 50-100 examples per character for basic functionality
- **Recommended**: 500-1000 examples per character for good performance
- **Ideal**: 1000+ examples per character for production-quality results

## Usage

### Basic Training and Prediction
```bash
# Train the network (this may take several minutes)
./character-recognition

# The program will:
# 1. Load all training images from ./data/
# 2. Train a neural network for 500 epochs
# 3. Generate visualization plots (if enabled)
# 4. Save the trained model to disk
# 5. Test prediction on "a.png"
```

### Configuration Options

Edit `config.go` to customize the training:

```go
func NewDefaultConfig() *Config {
    return &Config{
        // Image dimensions (all images resized to this)
        ImageWidth:  28,
        ImageHeight: 28,
        
        // Network architecture
        HiddenSize: 128,  // Neurons per hidden layer
        
        // Training settings
        TrainingOptions: graymatter.TrainingOptions{
            Iterations:   500,   // How many times to see all training data
            BatchSize:    32,    // Process 32 images before updating weights
            LearningRate: 0.001, // How aggressively to adjust weights
            
            // Save trained models
            EnableSaving: true,
            SavePath:     "image_classifier",
            
            // Generate training plots
            EnablePlotting: true,
            PlottingURL:    "http://localhost:8080",
        },
    }
}
```

## Understanding the Training Process

### What You'll See During Training

```
Loading training data...
Loaded 15000 training samples
Training samples: 12000
Validation samples: 3000
Creating neural network...
Starting training...

Epoch 0: Cost 3.245612, Train Acc 0.1234, Val Acc 0.1156, Val Cost 3.287543
Epoch 100: Cost 1.456789, Train Acc 0.6543, Val Acc 0.6234, Val Cost 1.523456
Epoch 200: Cost 0.876543, Train Acc 0.7890, Val Acc 0.7654, Val Cost 0.923456
...
Epoch 500: Cost 0.234567, Train Acc 0.9456, Val Acc 0.9123, Val Cost 0.287654

Training completed! Final cost: 0.234567
Best validation accuracy: 0.9123 at epoch 450
```

### Key Metrics Explained

- **Cost/Loss**: How wrong the network's predictions are (lower is better)
- **Training Accuracy**: Percentage correct on data the network learns from
- **Validation Accuracy**: Percentage correct on data the network hasn't seen (more important!)
- **Epoch**: One complete pass through all training data

### Healthy Training Signs
- ✅ Cost decreases over time
- ✅ Both training and validation accuracy increase
- ✅ Training and validation metrics stay close to each other

### Problem Signs
- ❌ Cost increases or stays flat
- ❌ Training accuracy much higher than validation accuracy (overfitting)
- ❌ Both accuracies plateau at low values (underfitting)

## Files Generated

After training, you'll find these files:

### Model Files
- `image_classifier_final.json`: The trained model saved at the end
- `image_classifier_best.json`: The version with highest validation accuracy
- `image_classifier_epoch_*.json`: Checkpoints saved during training

### Visualization Files (if plotting enabled)
- `training_loss.png`: How the error decreased during training
- `training_accuracy.png`: How accuracy improved during training  
- `network_architecture.png`: Visual diagram of the neural network structure
- `weight_distributions.png`: Health check of the network's internal parameters
- `confusion_matrix.png`: Shows which characters get confused with each other

## Making Predictions

### In Code
```go
// Load a trained model
classifier, metadata, err := LoadModelForInference("image_classifier_best.json")
if err != nil {
    log.Fatal(err)
}

// Predict on a new image
prediction, confidence, err := classifier.Predict("test_image.png")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Predicted: %s (%.2f%% confident)\n", prediction, confidence*100)
```

### Interpreting Results
- **90%+ confidence**: Very likely correct
- **70-90% confidence**: Probably correct, but worth double-checking
- **50-70% confidence**: Uncertain, could be wrong
- **<50% confidence**: Likely incorrect

## Troubleshooting

### Poor Training Performance

**Problem**: Accuracy stays low (below 70%)
```
Possible Causes:
- Not enough training data (need 500+ examples per character)
- Images are too different from each other
- Learning rate too high/low
- Network architecture too simple

Solutions:
- Collect more training data
- Ensure consistent image quality
- Try learning rates: 0.1, 0.01, 0.001, 0.0001
- Increase HiddenSize to 256 or 512
```

**Problem**: Training accuracy high but validation accuracy low (overfitting)
```
Training Acc: 95%, Validation Acc: 60%

Causes:
- Network memorizing instead of learning patterns
- Not enough training data diversity
- Network too complex for the amount of data

Solutions:
- Collect more varied training data
- Reduce HiddenSize or add fewer layers
- Stop training earlier (when validation accuracy peaks)
```

### Technical Issues

**Problem**: "Failed to load training data"
```bash
# Check directory structure
ls -la data/
ls -la data/upper/A/

# Ensure images are in correct format
file data/upper/A/*.png
```

**Problem**: "Plotting service unavailable" 
```bash
# Check if plotting service is running
curl http://localhost:8080/health

# Start plotting service
python plotting-service.py &
```

**Problem**: Out of memory during training
```go
// Reduce batch size in config
TrainingOptions: graymatter.TrainingOptions{
    BatchSize: 16,  // Reduced from 32
    // ...
}
```

## Advanced Usage

### Custom Network Architecture
```go
// Experiment with different architectures
LayerSizes: []int{
    784,  // Input (28×28 pixels)
    256,  // First hidden layer (increased from 128)
    128,  // Second hidden layer  
    64,   // Third hidden layer (added)
    62,   // Output (62 character classes)
}
```

### Hyperparameter Tuning
Try different combinations systematically:

| Learning Rate | Batch Size | Hidden Size | Expected Result |
|---------------|------------|-------------|-----------------|
| 0.1           | 32         | 128         | Fast but unstable |
| 0.01          | 32         | 128         | Good starting point |
| 0.001         | 32         | 128         | Slow but stable |
| 0.01          | 64         | 256         | Better for large datasets |

### Evaluation on Test Data
```go
// Evaluate final performance on completely unseen data
accuracy, err := classifier.EvaluateModel("test_data/")
fmt.Printf("Final test accuracy: %.2f%%\n", accuracy*100)
```

## Understanding the Code Structure

### Main Components

- **`main.go`**: Program entry point, orchestrates the entire process
- **`config.go`**: All configuration and hyperparameters
- **`types.go`**: Data structures representing images and the classifier
- **`image-processing.go`**: Converts image files to numbers the network can use
- **`data-loading.go`**: Loads and organizes training data from directories
- **`training.go`**: The actual machine learning training process
- **`prediction.go`**: Uses trained models to classify new images
- **`model-utils.go`**: Saving, loading, and evaluating trained models
- **`constants.go`**: Image processing constants and character mappings

### Design Patterns Used

- **Configuration Pattern**: All settings centralized in Config struct
- **Facade Pattern**: ImageClassifier hides neural network complexity
- **Strategy Pattern**: Different activation and cost functions
- **Builder Pattern**: Gradual construction of training datasets

## Performance Expectations

### Training Time
- **Small dataset** (1,000 images): 1-2 minutes
- **Medium dataset** (10,000 images): 5-15 minutes  
- **Large dataset** (100,000 images): 30-60 minutes

### Accuracy Expectations
- **Toy dataset**: 60-80% accuracy
- **Good dataset**: 85-95% accuracy
- **Excellent dataset**: 95-99% accuracy

### Resource Usage
- **Memory**: ~100MB for training, ~10MB for inference
- **CPU**: Single-threaded, benefits from faster CPUs
- **Storage**: Models are typically 1-10MB each

## Next Steps

### Improving Performance
1. **More Data**: Collect more diverse, high-quality training images
2. **Data Augmentation**: Rotate, scale, or add noise to existing images
3. **Better Architecture**: Experiment with different network designs
4. **Hyperparameter Tuning**: Systematically try different learning rates, batch sizes

### Production Deployment
1. **Web Service**: Wrap the classifier in an HTTP API
2. **Batch Processing**: Process many images at once
3. **Mobile App**: Convert model to mobile-friendly format
4. **Real-time Processing**: Optimize for low-latency predictions

### Advanced Techniques
1. **Convolutional Neural Networks**: Better for image recognition
2. **Transfer Learning**: Start with pre-trained models
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Active Learning**: Smart selection of training examples

## Contributing

This project demonstrates fundamental machine learning concepts in Go. Areas for improvement:

- Add data augmentation capabilities
- Implement different network architectures
- Add more comprehensive evaluation metrics
- Create web interface for easy testing
- Add support for real-time image capture

## License

MIT license.

---

**Remember**: Machine learning is as much art as science. Don't be discouraged if your first attempts don't work perfectly. Experiment with different approaches, collect better data, and iterate on your solution. The key to success in ML is persistence and systematic experimentation!