# Enhanced Character Recognition Neural Network

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
- **Smart model loading** (reuses existing trained models)
- **Data sampling options** for faster training cycles
- **Batch processing optimizations** for large datasets

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
    ├── exclamation/# ! symbol
    │   ├── image1.png
    │   └── ...
    ├── dot/        # . symbol
    ├── comma/      # , symbol
    ├── at/         # @ symbol
    └── ...         # 32 punctuation marks total
```

**Image Requirements:**
- Format: PNG or JPEG
- Size: Will be automatically resized to 28×28 pixels
- Content: Single character, roughly centered
- Quality: Clear, high-contrast images work best
- **Background consistency**: All training images should have the same background color

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
./character-recognition -samples 1000 -iterations 200 -batchsize 64

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