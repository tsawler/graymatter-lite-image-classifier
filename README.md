# Character Recognition Neural Network

A Go program that trains a neural network to recognize handwritten characters (A-Z, a-z, 0-9, and punctuation marks) using machine learning techniques. Now supports 94 different character classes with advanced training features.

## What This Program Does

Imagine you want to build a system that can look at an image of a handwritten character and automatically tell you whether it's an "A", "B", "7", "!", "*", etc. This is exactly what this program does! It's similar to how:

- Postal services automatically read ZIP codes on mail
- Banks process handwritten amounts on checks  
- Document scanning apps convert handwritten notes to text
- OCR systems digitize printed and handwritten documents
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
Show computer 1000 images labeled "!"
... (computer learns patterns automatically)
Now computer can recognize H, A, 7, ! in new images it's never seen
```

This program uses a **neural network** - a type of machine learning inspired by how brain neurons work. The network learns by studying thousands of example images and gradually adjusting its internal parameters to recognize character patterns.

## New Features (Enhanced Version)

### 🎯 Expanded Character Support
- **94 total character classes** (previously 62)
- **32 punctuation marks**: !, @, #, $, %, &, *, +, =, ?, etc.
- **Comprehensive text recognition** suitable for real-world documents

### 🚀 Command Line Interface
- **Flexible training parameters** via command line flags
- **Data sampling** for faster experimentation with large datasets
- **Configurable batch sizes, learning rates, and iterations**

### 📊 Enhanced Monitoring
- **Smart model loading** (reuses existing trained models)
- **Processed image saving** for debugging predictions
- **Comprehensive training feedback** with validation tracking

### ⚡ Performance Optimizations
- **Data sampling options** for faster training cycles
- **Batch processing optimizations** for large datasets
- **Automatic best model saving** during training

## Prerequisites

### Required Software
- **Go 1.21+**: The programming language this is written in
- **Python 3.8+**: For the plotting sidecar service that generates training charts
- **Git**: To clone the repositories

### Required Services
- **Plotting Sidecar** (optional): For generating training visualization charts
  - Repository: https://github.com/tsawler/graymatter-sidecar
  - Language: Python (Flask-based web service)
  - Purpose: Creates plots of training progress, network architecture, confusion matrices

### Required Go Module
This program depends on the `graymatter-lite` neural network library:
```bash
go mod tidy  # This will automatically download the required module
```

### Training Data
You need organized image data in this structure:
```
data/
├── upper/          # Uppercase letters (A-Z)
│   ├── A/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── B/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
├── lower/          # Lowercase letters (a-z)
│   ├── a/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
├── digits/         # Numbers (0-9)
│   ├── 0/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
└── punctuation/    # Punctuation marks (NEW!)
    ├── asterisk/   # * symbol
    │   ├── image1.png
    │   └── ...
    ├── question/   # ? symbol
    │   ├── image1.png
    │   └── ...
    ├── !           # ! symbol (can use actual character as directory name)
    │   ├── image1.png
    │   └── ...
    ├── dot/        # . symbol
    ├── comma/      # , symbol (uses descriptive name)
    ├── @           # @ symbol
    └── ...         # 32 punctuation marks total
```

**Image Requirements:**
- Format: PNG or JPEG
- Size: Will be automatically resized to 28×28 pixels
- Content: Single character, roughly centered
- Quality: Clear, high-contrast images work best
- **Background consistency**: All training images should have the same background color

**Punctuation Directory Names:**
Some punctuation characters can't be used as directory names, so we use descriptive names:
- `asterisk` → `*`
- `question` → `?`
- `slash` → `/`
- `backslash` → `\`
- `colon` → `:`
- `quote` → `"`
- `pipe` → `|`
- `lt` → `<`
- `gt` → `>`
- `dot` → `.`

## How Neural Networks Work (Enhanced)

The enhanced network now handles 94 different character classes:

### 1. Input Layer (784 neurons)
Takes your 28×28 pixel image and converts it to 784 numbers (one per pixel). Black pixels become 0.0, white pixels become 1.0, gray pixels become values in between.

### 2. Hidden Layers (128 neurons each)
These layers detect patterns:
- **First hidden layer**: Might learn to detect edges, curves, and basic shapes
- **Second hidden layer**: Might learn to combine edges into character parts (letter components, punctuation features)

### 3. Output Layer (94 neurons) - EXPANDED
One neuron for each possible character:
- **26 uppercase letters** (A-Z)
- **26 lowercase letters** (a-z)
- **10 digits** (0-9)
- **32 punctuation marks** (!, @, #, $, %, etc.)

Each outputs a probability:
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
The program can generate helpful training visualizations using the separate plotting sidecar service:

```bash
# Clone the plotting sidecar service
git clone https://github.com/tsawler/graymatter-sidecar
cd graymatter-sidecar

# Install Python dependencies
pip install -r requirements.txt

# Start the plotting service on port 8080
python app.py &

# Return to your character recognition project
cd ../character-recognition
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
- **Minimum**: 100-500 examples per character for basic functionality
- **Recommended**: 1000-2000 examples per character for good performance
- **Ideal**: 5000+ examples per character for production-quality results
- **Large scale**: 10,000+ examples per character for maximum accuracy

## Usage

### Command Line Interface

The program now supports extensive command line configuration:

```bash
# Basic usage (uses all defaults)
./character-recognition

# Custom training parameters
./character-recognition -batchsize 128 -lr 0.01 -iterations 300

# Fast experimentation with data sampling
./character-recognition -samples 1000 -iterations 200 -batchsize 64 -lr 0.001

# Production training with large batches
./character-recognition -batchsize 512 -lr 0.02 -iterations 400

# Quick prediction test
./character-recognition -predict my_test_image.png

# Combined example: fast training with custom settings
./character-recognition -samples 2000 -batchsize 256 -lr 0.015 -iterations 250 -predict test.png
```

### Command Line Flags

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `-batchsize` | Number of images processed per weight update | 32 | `-batchsize 128` |
| `-iterations` | Number of training epochs | 500 | `-iterations 300` |
| `-lr` | Learning rate (step size for weight updates) | 0.001 | `-lr 0.01` |
| `-samples` | Max images per class (0 = use all) | 0 | `-samples 2000` |
| `-predict` | Image file to test prediction on | "a.png" | `-predict test_char.png` |

### Data Sampling Feature

**Why Use Data Sampling?**
With large datasets (10,000+ images per class), training can take hours. Data sampling allows faster experimentation:

```bash
# Quick experiment (30-60 min training)
./character-recognition -samples 1000 -iterations 200

# Balanced approach (1-2 hour training)  
./character-recognition -samples 2000 -iterations 300

# Near-optimal results (2-4 hour training)
./character-recognition -samples 5000 -iterations 400

# Full dataset (6+ hour training)
./character-recognition -batchsize 512 -lr 0.02
```

**Expected Results:**
- **1000 samples**: ~88-91% accuracy, very fast training
- **2000 samples**: ~91-93% accuracy, moderate training time
- **5000 samples**: ~93-95% accuracy, longer training time  
- **No sampling**: ~94-96% accuracy, maximum training time

## Enhanced Training Process

### What You'll See During Training

```
Starting Enhanced Image Classification System...
Supporting 94 character classes: A-Z, a-z, 0-9, and punctuation marks
Data sampling enabled: Using maximum 2000 images per class

Loading training data...
Loaded 25000 training samples for class 'A'
Sampled 2000 images for class 'A' (from 25000 available)
... (similar for all 94 classes)
Total training samples loaded: 188000

Training samples: 150400
Validation samples: 37600
Creating neural network...
Starting training...

Epoch 0: Cost 3.245612, Train Acc 0.1234, Val Acc 0.1156, Val Cost 3.287543
Epoch 100: Cost 1.456789, Train Acc 0.6543, Val Acc 0.6234, Val Cost 1.523456
Epoch 200: Cost 0.876543, Train Acc 0.7890, Val Acc 0.7654, Val Cost 0.923456
...
Epoch 300: Cost 0.234567, Train Acc 0.9456, Val Acc 0.9123, Val Cost 0.287654

Training completed! Final cost: 0.234567
Best validation accuracy: 0.9123 at epoch 285

Saving trained model as best model...  
Model saved successfully: ./image_classifier_final.json
```

### Enhanced Training Features
- **Comprehensive validation tracking** with best model saving
- **Epoch-by-epoch monitoring** of cost and accuracy metrics
- **Automatic model checkpointing** during training

### Smart Model Loading

The program automatically detects existing trained models:

```bash
# First run: Trains new model
./character-recognition -samples 2000

# Subsequent runs: Loads existing model
./character-recognition -predict new_image.png

# Output:
# Found existing best model: ./image_classifier_final.json
# Loading pre-trained model...
# Successfully loaded pre-trained model!
# Model description: Enhanced character classifier with 94 classes...
```

## Understanding the Training Process (Enhanced)

### Key Metrics Explained

- **Cost/Loss**: How wrong the network's predictions are (lower is better)
- **Training Accuracy**: Percentage correct on data the network learns from  
- **Validation Accuracy**: Percentage correct on data the network hasn't seen (more important!)
- **Epoch**: One complete pass through all training data
- **Best Model Tracking**: Automatically saves the model with highest validation accuracy

### Healthy Training Signs
- ✅ Cost decreases over time
- ✅ Both training and validation accuracy increase
- ✅ Training and validation metrics stay reasonably close to each other
- ✅ Best validation accuracy improves during training

### Problem Signs
- ❌ Cost increases or stays flat
- ❌ Training accuracy much higher than validation accuracy (overfitting)
- ❌ Both accuracies plateau at low values (underfitting)
- ❌ Validation accuracy decreases while training accuracy increases

## Prediction and Debugging

### Enhanced Prediction Process

The program now saves the processed prediction image for debugging:

```bash
./character-recognition -predict handwriting_sample.jpg

# Output:
# Saved processed image as: image_to_predict.png
# Making prediction on 'handwriting_sample.jpg'...
# ✓ Prediction successful!
#   Predicted character: 'A'
#   Confidence: 94.25%
#   Assessment: Very confident prediction
#   Character type: Uppercase letter
```

### Debugging with Saved Images

The program automatically saves `image_to_predict.png` showing exactly what the neural network analyzed:
- **28×28 pixels** (network input size)
- **Grayscale conversion** result
- **Aspect ratio preservation** and centering
- **Background normalization**

**Use this for troubleshooting:**
1. Check if the processed image looks like your training data
2. Verify background colors match (all training images should have consistent backgrounds)
3. Ensure character is clearly visible and properly centered
4. Compare processed image quality to your training examples

### Background Color Consistency

**Important**: All training images must have the same background color as prediction images:
- **If training images**: White characters on black background
- **Then prediction images**: Should also be white characters on black background
- **Mismatched backgrounds**: Will cause poor prediction accuracy

The program processes prediction images the same way as training images, so consistency is crucial for good results.

## Files Generated

After training, you'll find these files:

### Model Files
- `image_classifier_final.json`: The trained model saved at the end
- `image_classifier_best.json`: The version with highest validation accuracy (if validation data used)
- `image_classifier_epoch_*.json`: Checkpoints saved during training (if enabled)

### Visualization Files (if plotting enabled)
- `training_loss.png`: How the error decreased during training
- `training_accuracy.png`: How accuracy improved during training  
- `network_architecture.png`: Visual diagram of the neural network structure
- `weight_distributions.png`: Health check of the network's internal parameters
- `confusion_matrix.png`: Shows which characters get confused with each other

### Debug Files
- `image_to_predict.png`: The processed 28×28 image that was fed to the network for prediction

## Making Predictions

### In Code
```go
// Load a trained model
classifier, metadata, err := LoadModelForInference("image_classifier_final.json")
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

## Training Time Expectations

With the enhanced 94-class system and data sampling:

| Samples Per Class | Total Images | Expected Training Time | Expected Accuracy |
|-------------------|--------------|------------------------|-------------------|
| 500 | 47,000 | 15-30 minutes | 85-88% |
| 1000 | 94,000 | 30-60 minutes | 88-91% |
| 2000 | 188,000 | 1-2 hours | 91-93% |
| 5000 | 470,000 | 2-4 hours | 93-95% |
| No limit | 1M+ images | 4-8+ hours | 94-96% |

**Factors affecting training time:**
- **Dataset size**: More images = longer training
- **Batch size**: Larger batches = faster training (with sufficient RAM)
- **Learning rate**: Higher rates = faster convergence (but risk instability)
- **Hardware**: Better CPU = faster training (this implementation is CPU-only)

## Troubleshooting

### Poor Training Performance

**Problem**: Accuracy stays low (below 70%)
```
Possible Causes:
- Not enough training data (need 1000+ examples per character for 94 classes)
- Images are too different from each other or inconsistent backgrounds
- Learning rate too high/low
- Network architecture too simple for 94-class problem

Solutions:
- Collect more training data, especially for punctuation marks
- Ensure consistent image quality and background colors
- Try learning rates: 0.1, 0.01, 0.001, 0.0001
- Increase HiddenSize to 256 or 512
- Use data sampling to experiment faster: -samples 1000
```

**Problem**: Training accuracy high but validation accuracy low (overfitting)
```
Training Acc: 95%, Validation Acc: 60%

Causes:
- Network memorizing instead of learning patterns
- Not enough training data diversity (especially problematic with 94 classes)
- Network too complex for the amount of data

Solutions:
- Collect more varied training data for all 94 classes
- Use data sampling to ensure balanced training: -samples 2000
- Reduce HiddenSize or add fewer layers
- Stop training earlier (when validation accuracy peaks)
```

**Problem**: Poor punctuation recognition
```
Letters and digits work well, but punctuation marks are often wrong

Causes:
- Punctuation marks are visually more similar to each other
- Less training data for punctuation compared to letters
- Punctuation marks are often smaller/different in images

Solutions:
- Collect extra training examples for punctuation marks
- Ensure punctuation images are clear and well-centered
- Verify punctuation directory structure matches the mapping
- Consider training with more samples: -samples 3000 or higher
```

### Technical Issues

**Problem**: "Failed to load training data"
```bash
# Check directory structure
ls -la data/
ls -la data/upper/A/
ls -la data/punctuation/

# Verify punctuation directory names
ls -la data/punctuation/ | grep asterisk  # Should exist for * symbol
ls -la data/punctuation/ | grep question  # Should exist for ? symbol

# Ensure images are in correct format
file data/upper/A/*.png
file data/punctuation/asterisk/*.png
```

**Problem**: "Unknown punctuation directory" warnings
```bash
# Check that punctuation directories use the correct names
# These should exist:
data/punctuation/asterisk/     # for *
data/punctuation/question/     # for ?
data/punctuation/slash/        # for /
data/punctuation/dot/          # for .
# etc.

# These can use the actual character:
data/punctuation/!/            # for !
data/punctuation/@/            # for @
data/punctuation/#/            # for #
```

**Problem**: "Plotting service unavailable" 
```bash
# Check if plotting service is running
curl http://localhost:8080/health

# If not running, start the sidecar service
cd path/to/graymatter-sidecar
python app.py &

# Or disable plotting
./character-recognition -samples 1000  # (plotting disabled in code)
```

**Problem**: Out of memory during training
```bash
# Reduce batch size
./character-recognition -batchsize 16 -samples 2000

# Use data sampling to reduce dataset size
./character-recognition -samples 1000 -batchsize 32
```

**Problem**: Training takes too long
```bash
# Use data sampling for faster experimentation
./character-recognition -samples 1000 -iterations 200

# Increase batch size (if you have enough RAM)
./character-recognition -batchsize 128 -samples 2000

# Use higher learning rate for faster convergence
./character-recognition -lr 0.01 -samples 1500
```

### Prediction Issues

**Problem**: Good training accuracy but poor real-world predictions
```
Causes:
- Background color mismatch between training and prediction images
- Different image quality or style
- Characters not properly centered

Solutions:
- Check the saved image_to_predict.png file
- Ensure training images have same background as prediction images
- Verify processed image looks similar to training examples
- Collect training data that matches your real-world use case
```

**Problem**: Low confidence on all predictions
```
Causes:
- Model not well-trained
- Preprocessing mismatch
- Network confusion due to too many similar classes

Solutions:
- Train longer or with more data: -iterations 400 -samples 3000
- Check background color consistency
- Examine confusion matrix to see which classes are being mixed up
```

## Advanced Usage

### Custom Network Architecture
```go
// Experiment with different architectures in config.go
LayerSizes: []int{
    784,  // Input (28×28 pixels)
    256,  // First hidden layer (increased from 128)
    128,  // Second hidden layer  
    64,   // Third hidden layer (added)
    94,   // Output (94 character classes)
}
```

### Hyperparameter Tuning
Try different combinations systematically:

| Learning Rate | Batch Size | Samples | Hidden Size | Expected Result |
|---------------|------------|---------|-------------|-----------------|
| 0.1           | 32         | 1000    | 128         | Fast but unstable |
| 0.01          | 64         | 2000    | 128         | Good starting point |
| 0.001         | 128        | 5000    | 256         | Slow but stable |
| 0.01          | 256        | 0       | 512         | Best for large datasets |

### Workflow for Large Datasets
```bash
# Step 1: Quick experiment
./character-recognition -samples 500 -iterations 100 -lr 0.01

# Step 2: If promising, scale up
./character-recognition -samples 2000 -iterations 200 -batchsize 64

# Step 3: Final training with more data
./character-recognition -samples 5000 -iterations 300 -batchsize 128

# Step 4: Production model (if needed)
./character-recognition -batchsize 256 -lr 0.01 -iterations 400
```

### Evaluation on Test Data
```go
// Evaluate final performance on completely unseen data
accuracy, err := classifier.EvaluateModel("test_data/")
fmt.Printf("Final test accuracy: %.2f%%\n", accuracy*100)
```

## Understanding the Code Structure

### Main Components

- **`main.go`**: Program entry point with command line interface
- **`config.go`**: All configuration, hyperparameters, and data sampling settings
- **`types.go`**: Data structures representing images and the classifier
- **`constants.go`**: Character mappings and punctuation directory translations
- **`image-processing.go`**: Converts image files to numbers, saves processed images
- **`data-loading.go`**: Loads and organizes training data, implements data sampling
- **`training.go`**: The actual machine learning training process
- **`prediction.go`**: Uses trained models to classify new images
- **`model-utils.go`**: Saving, loading, and evaluating trained models

### Enhanced Design Patterns

- **Command Line Configuration**: Flexible parameter control via flags
- **Data Sampling Strategy**: Random sampling for faster experimentation
- **Smart Model Management**: Automatic loading of existing models
- **Debug Image Saving**: Transparent preprocessing visualization
- **Punctuation Mapping**: Clean handling of filesystem-incompatible characters

## Performance Expectations

### Accuracy Expectations (94-Class System)
- **Toy dataset** (500 samples): 70-85% accuracy
- **Small dataset** (1000 samples): 85-90% accuracy
- **Medium dataset** (2000 samples): 90-93% accuracy
- **Good dataset** (5000 samples): 93-95% accuracy
- **Excellent dataset** (10000+ samples): 95-97% accuracy

### Resource Usage
- **Memory**: ~200MB for training, ~20MB for inference
- **CPU**: Single-threaded, benefits from faster CPUs
- **Storage**: Models are typically 2-15MB each (larger due to 94 classes)

## Next Steps

### Improving Performance
1. **More Data**: Collect more diverse, high-quality training images for all 94 classes
2. **Data Augmentation**: Rotate, scale, or add noise to existing images
3. **Better Architecture**: Experiment with different network designs
4. **Hyperparameter Tuning**: Use data sampling for systematic experimentation

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

This enhanced project demonstrates comprehensive machine learning concepts in Go. Areas for improvement:

- Add data augmentation capabilities
- Implement different network architectures  
- Add more comprehensive evaluation metrics
- Create web interface for easy testing
- Add support for real-time image capture
- Optimize training performance for very large datasets

## License

MIT license. See the [LICENSE](LICENSE.md) file for more details.

---

**Remember**: Machine learning with 94 character classes is significantly more challenging than simpler problems. Don't be discouraged if your first attempts don't work perfectly. Use data sampling for rapid experimentation, ensure consistent image preprocessing, and iterate systematically. The key to success in ML is persistence, good data, and systematic experimentation!