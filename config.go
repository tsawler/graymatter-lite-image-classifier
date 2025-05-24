package main

import "github.com/tsawler/graymatter-lite"

// Config holds all configuration parameters for our image classification neural network.
//
// WHY CONFIGURATION MATTERS:
// Neural networks have many "hyperparameters" - settings that control how the network
// is structured and how it learns. These aren't learned during training; you have to
// set them beforehand. Getting these right can be the difference between a network
// that achieves 95% accuracy and one that barely works at 60%.
//
// CONFIGURATION PHILOSOPHY:
// Rather than hardcoding values throughout the program, we centralize all settings
// in this configuration struct. This makes it easy to experiment with different
// values and ensures consistency across the entire program.
type Config struct {
	// IMAGE DIMENSIONS
	// These define the size of input images our network expects
	
	// ImageWidth, ImageHeight: Dimensions of input images in pixels
	// 
	// WHY 28x28?
	// This is a common size for character recognition tasks, popularized by the
	// MNIST dataset. It's large enough to capture essential character features
	// but small enough to be computationally efficient. Each character is
	// centered and normalized to fit within this square.
	//
	// CONSISTENCY IS CRUCIAL:
	// ALL images (training and prediction) must be exactly this size. The
	// preprocessing pipeline will resize images to match these dimensions.
	ImageWidth, ImageHeight int

	// NETWORK ARCHITECTURE
	// These parameters define the structure of our neural network
	
	// InputSize: Total number of input neurons (typically width × height for images)
	// For 28×28 images, this equals 784 neurons - one for each pixel
	//
	// WHY FLATTEN IMAGES?
	// Neural networks expect 1-dimensional input, but images are 2-dimensional.
	// We "flatten" the 28×28 image into a single row of 784 pixel values.
	// The network will learn to recognize spatial patterns even in this format.
	InputSize int

	// HiddenSize: Number of neurons in each hidden layer
	//
	// HIDDEN LAYERS DO THE HEAVY LIFTING:
	// These are the layers between input and output that find patterns in the data.
	// More neurons = more capacity to learn complex patterns, but also more risk
	// of overfitting and slower training.
	//
	// CHOOSING HIDDEN SIZE:
	// - Too small (e.g., 10): Network can't learn complex character patterns
	// - Too large (e.g., 5000): Network might memorize instead of generalize
	// - Just right (128): Good balance for character recognition tasks
	//
	// RULE OF THUMB: Start with something between input size and output size
	HiddenSize int

	// OutputSize: Number of output neurons (one for each possible class)
	// 26 (A-Z) + 26 (a-z) + 10 (0-9) = 62 possible characters
	//
	// ONE NEURON PER CLASS:
	// In classification, we typically have one output neuron for each possible
	// answer. The network will output a probability for each class, and we
	// choose the one with the highest probability as our prediction.
	OutputSize int

	// DATA AND MODEL PATHS
	// These specify where to find training data and where to save the trained model
	
	// DataDir: Directory containing training images organized by class
	// Expected structure:
	// data/
	//   upper/           <- uppercase letters
	//     A/
	//       image1.png
	//       image2.png
	//     B/
	//       image1.png
	//   lower/           <- lowercase letters
	//     a/
	//       image1.png
	//   digits/          <- numbers
	//     0/
	//       image1.png
	//
	// WHY THIS STRUCTURE?
	// Organizing images by class in separate folders makes it easy to automatically
	// label the training data. The folder name becomes the label for all images inside.
	DataDir string

	// ModelPath: Base filename for saving the trained model (without extension)
	// The actual saved files will have suffixes like "_final.json", "_best.json"
	//
	// MODEL PERSISTENCE:
	// After spending time and computational resources training a network,
	// you want to save it! This allows you to:
	// - Use the trained model later without retraining
	// - Share models with colleagues
	// - Deploy models to production environments
	// - Resume training from checkpoints if interrupted
	ModelPath string

	// VISUALIZATION SETTINGS
	
	// PlottingURL: Address of the plotting service for generating training charts
	//
	// WHY SEPARATE PLOTTING SERVICE?
	// Training generates lots of data (loss curves, accuracy metrics, etc.) that
	// are best visualized as charts. Rather than building plotting into our
	// Go program, we use a separate Python service that specializes in creating
	// beautiful visualizations. This separation of concerns keeps our ML code
	// focused on learning algorithms.
	PlottingURL string

	// TRAINING CONFIGURATION
	// These parameters control how the neural network learns
	
	// TrainingOptions: Detailed settings for the learning process
	// This includes learning rate, batch size, number of epochs, etc.
	// These hyperparameters are critical for successful training.
	TrainingOptions graymatter.TrainingOptions
}

// NewDefaultConfig creates a configuration with sensible defaults for character recognition.
//
// THESE DEFAULTS ARE CAREFULLY CHOSEN:
// The values here represent a good starting point for character recognition tasks.
// They're based on common practices in the field and should work reasonably well
// for most similar problems. However, you might need to adjust them based on:
// - The complexity of your specific dataset
// - Available computational resources
// - Desired training time vs. accuracy trade-offs
//
// EXPERIMENTATION ENCOURAGED:
// Don't treat these as gospel! Machine learning is often about experimentation.
// Try different values and see what works best for your specific problem.
func NewDefaultConfig() *Config {
	// Standard image dimensions for character recognition
	imageWidth, imageHeight := 28, 28
	
	return &Config{
		// IMAGE PREPROCESSING SETTINGS
		ImageWidth:  imageWidth,
		ImageHeight: imageHeight,
		InputSize:   imageWidth * imageHeight, // 784 pixels total
		
		// NETWORK ARCHITECTURE
		// This creates a relatively simple but effective architecture:
		// 784 input neurons → 128 hidden neurons → 128 hidden neurons → 62 output neurons
		HiddenSize: 128, // Good balance between capacity and efficiency
		OutputSize: 62,  // All possible characters (A-Z, a-z, 0-9)
		
		// FILE SYSTEM PATHS
		DataDir:   "data",              // Look for training images in ./data/
		ModelPath: "image_classifier",  // Save trained model as image_classifier_*.json
		
		// EXTERNAL SERVICES
		PlottingURL: "http://localhost:8080", // Plotting service running locally
		
		// TRAINING HYPERPARAMETERS
		TrainingOptions: graymatter.TrainingOptions{
			// LEARNING SCHEDULE
			Iterations:   500,   // Number of complete passes through training data
			BatchSize:    32,    // Process 32 images at a time before updating weights
			LearningRate: 0.001, // How big steps to take when adjusting weights
			
			// MODEL PERSISTENCE
			EnableSaving: true,                // Save the model during/after training
			SavePath:     "image_classifier",  // Base name for saved model files
			SaveInterval: 100,                 // Save checkpoint every 100 epochs
			
			// VISUALIZATION
			EnablePlotting: true,                     // Generate training progress charts
			PlottingURL:    "http://localhost:8080",  // Where to send plotting requests
		},
	}
}

// UNDERSTANDING THE HYPERPARAMETERS:

// ITERATIONS (500):
// Each iteration processes the entire training dataset once. 500 iterations means
// the network sees each training image 500 times. This might seem like a lot, but
// neural networks need repetition to learn patterns gradually.

// BATCH SIZE (32):
// Instead of updating weights after each individual image, we process 32 images
// and then update weights based on the average error. This is more efficient and
// often leads to more stable learning.

// LEARNING RATE (0.001):
// This controls how aggressively the network adjusts its weights when it makes
// mistakes. Too high (0.1) and the network might overshoot good solutions.
// Too low (0.00001) and learning will be painfully slow. 0.001 is conservative
// but safe for most problems.

// HIDDEN SIZE (128):
// This determines the network's capacity to learn complex patterns. For character
// recognition, 128 neurons can capture the essential features that distinguish
// different letters and numbers without being so large that training becomes
// unwieldy.

// These defaults should work well for most character recognition tasks, but don't
// hesitate to experiment! Machine learning is as much art as science, and finding
// the right hyperparameters often requires trial and error.