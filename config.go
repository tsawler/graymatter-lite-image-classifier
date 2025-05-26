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
//
// ENHANCED WITH DATA SAMPLING:
// Now includes flexible data sampling capabilities that allow you to work with
// subsets of your training data for faster development and experimentation.
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
	// UPDATED: 26 (A-Z) + 26 (a-z) + 10 (0-9) + 32 (punctuation) = 94 possible characters
	//
	// ONE NEURON PER CLASS:
	// In classification, we typically have one output neuron for each possible
	// answer. The network will output a probability for each class, and we
	// choose the one with the highest probability as our prediction.
	//
	// EXPANDED CHARACTER SET:
	// Now includes comprehensive punctuation support for real-world text recognition:
	// - All standard letters and digits (62 classes)
	// - Common punctuation marks (32 additional classes)
	// - Total: 94 different character classes
	OutputSize int

	// DATA SAMPLING CONFIGURATION
	// These parameters control how much training data to use

	// SamplesPerClass: Number of images to use per character class
	// NEW FEATURE for flexible dataset sizing
	//
	// SAMPLING STRATEGY:
	// 0 = Use all available images (full dataset, ~13,000+ per class)
	// N > 0 = Use exactly N images per class (balanced sampling)
	//
	// WHY SAMPLING IS VALUABLE:
	// - Rapid prototyping: Test model changes with small datasets
	// - Resource management: Train on smaller datasets when appropriate
	// - Progressive development: Start small, scale up as needed
	// - Debugging: Work with manageable sizes during development
	//
	// SAMPLING ENSURES BALANCE:
	// When sampling, we ensure equal representation across all 94 classes.
	// This prevents bias toward classes with more training examples.
	//
	// RECOMMENDED SAMPLING SIZES:
	// - 10-50: Rapid prototyping and architecture testing
	// - 100-500: Development and hyperparameter tuning
	// - 1000-5000: Validation and accuracy assessment
	// - 0 (all): Final model training and production
	SamplesPerClass int

	// DATA AND MODEL PATHS
	// These specify where to find training data and where to save the trained model

	// DataDir: Directory containing training images organized by class
	// UPDATED EXPECTED STRUCTURE:
	// data/
	//   upper/           <- uppercase letters
	//     A/
	//       image1.png
	//       image2.png
	//       ... (up to 13000+ images)
	//     B/
	//       image1.png
	//       ... (up to 13000+ images)
	//   lower/           <- lowercase letters
	//     a/
	//       image1.png
	//       ... (up to 13000+ images)
	//   digits/          <- numbers
	//     0/
	//       image1.png
	//       ... (up to 13000+ images)
	//   punctuation/     <- punctuation marks (NEW!)
	//     asterisk/      <- * symbol
	//       image1.png
	//       ... (up to 13000+ images)
	//     dot/           <- . symbol
	//       image1.png
	//       ... (up to 13000+ images)
	//     question/      <- ? symbol
	//       image1.png
	//       ... (up to 13000+ images)
	//
	// SAMPLING IMPACT ON DIRECTORY STRUCTURE:
	// The sampling feature randomly selects N images from each character's
	// directory, ensuring balanced representation across the full character set.
	//
	// WHY THIS STRUCTURE?
	// Organizing images by class in separate folders makes it easy to automatically
	// label the training data. The folder name becomes the label for all images inside.
	// For punctuation, we use descriptive names since some punctuation characters
	// can't be used as directory names on most file systems.
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
// UPDATED FOR 94 CLASSES WITH SAMPLING:
// The configuration now supports 94 different character classes including
// comprehensive punctuation support and flexible data sampling for efficient
// development workflows.
//
// EXPERIMENTATION ENCOURAGED:
// Don't treat these as gospel! Machine learning is often about experimentation.
// Try different values and see what works best for your specific problem.
// The sampling feature makes this experimentation much more efficient.
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
		// 784 input neurons → 128 hidden neurons → 128 hidden neurons → 94 output neurons
		HiddenSize: 128, // Good balance between capacity and efficiency
		OutputSize: 94,  // All possible characters (A-Z, a-z, 0-9, punctuation marks)

		// DATA SAMPLING CONFIGURATION
		// Default to using all available data (no sampling)
		// Users can override this via command line or programmatically
		SamplesPerClass: 0, // 0 = use all available images per class

		// FILE SYSTEM PATHS
		DataDir:   "data",             // Look for training images in ./data/
		ModelPath: "image_classifier", // Save trained model as image_classifier_*.json

		// EXTERNAL SERVICES
		PlottingURL: "http://localhost:8080", // Plotting service running locally

		// TRAINING HYPERPARAMETERS
		TrainingOptions: graymatter.TrainingOptions{
			// LEARNING SCHEDULE
			Iterations:   500,   // Number of complete passes through training data
			BatchSize:    32,    // Process 32 images at a time before updating weights
			LearningRate: 0.001, // How big steps to take when adjusting weights

			// MODEL PERSISTENCE
			EnableSaving: true,               // Save the model during/after training
			SavePath:     "image_classifier", // Base name for saved model files
			SaveInterval: 100,                // Save checkpoint every 100 epochs

			// VISUALIZATION
			EnablePlotting: true,                    // Generate training progress charts
			PlottingURL:    "http://localhost:8080", // Where to send plotting requests
			ShowProgress:   true,
		},
	}
}

// UNDERSTANDING THE HYPERPARAMETERS WITH SAMPLING:

// ITERATIONS (500):
// Each iteration processes the entire training dataset once. 500 iterations means
// the network sees each training image 500 times. With sampling, this becomes:
// - Full dataset: Each of ~1.3M images seen 500 times
// - 100 samples/class: Each of ~9.4K images seen 500 times
// - 10 samples/class: Each of ~940 images seen 500 times
// Smaller datasets may need fewer iterations to avoid overfitting.

// BATCH SIZE (32):
// Instead of updating weights after each individual image, we process 32 images
// and then update weights based on the average error. This is more efficient and
// often leads to more stable learning. The batch size remains appropriate
// regardless of sampling size.

// LEARNING RATE (0.001):
// This controls how aggressively the network adjusts its weights when it makes
// mistakes. We use a conservative learning rate (0.001) because:
// - With 94 classes, the network needs to make careful adjustments
// - Smaller datasets (via sampling) may be more sensitive to learning rate
// - Conservative rates are generally safer for stable training

// SAMPLES PER CLASS (0 = all):
// This new parameter controls dataset size:
// - 0: Use all available images (~13,000+ per class, ~1.3M total)
// - 10: Use 10 images per class (940 total images)
// - 100: Use 100 images per class (9,400 total images)
// - 1000: Use 1000 images per class (94,000 total images)

// HIDDEN SIZE (128):
// This determines the network's capacity to learn complex patterns. 128 neurons
// should be sufficient to distinguish between 94 different character classes,
// even with smaller training datasets achieved through sampling.

// OUTPUT SIZE (94):
// Matches our expanded character set:
// - 26 uppercase letters (A-Z)
// - 26 lowercase letters (a-z)
// - 10 digits (0-9)
// - 32 punctuation marks (!, @, #, $, %, etc.)

// SAMPLING STRATEGY CONSIDERATIONS:

// BALANCED SAMPLING:
// When SamplesPerClass > 0, we ensure equal representation across all classes.
// This prevents bias toward classes with more training examples and maintains
// consistent performance across the full character set.

// RANDOM SELECTION:
// Images are randomly selected from each class directory to ensure
// diversity within the sampled dataset. This helps maintain the statistical
// properties of the full dataset even with smaller sample sizes.

// PROGRESSIVE DEVELOPMENT WORKFLOW:

// PHASE 1: RAPID PROTOTYPING (10-50 samples/class)
// - Purpose: Test model architecture and basic functionality
// - Training time: Seconds to minutes
// - Use case: Initial development, debugging, architecture experiments
// - Expected accuracy: Lower, but sufficient for validation

// PHASE 2: DEVELOPMENT VALIDATION (100-500 samples/class)
// - Purpose: Hyperparameter tuning and model refinement
// - Training time: Minutes to tens of minutes
// - Use case: Parameter optimization, feature testing
// - Expected accuracy: Moderate, good for development decisions

// PHASE 3: ACCURACY ASSESSMENT (1000-5000 samples/class)
// - Purpose: Performance evaluation and model validation
// - Training time: Tens of minutes to hours
// - Use case: Pre-production testing, performance benchmarking
// - Expected accuracy: High, approaching full dataset performance

// PHASE 4: PRODUCTION TRAINING (0 = all samples)
// - Purpose: Final model for deployment
// - Training time: Hours
// - Use case: Production model creation, maximum accuracy
// - Expected accuracy: Maximum achievable with current architecture

// SAMPLING RECOMMENDATIONS BY USE CASE:

// ARCHITECTURE EXPERIMENTATION:
// Use 10-50 samples per class to quickly test different network architectures,
// activation functions, or layer configurations. Fast feedback enables rapid iteration.

// HYPERPARAMETER TUNING:
// Use 100-500 samples per class to optimize learning rates, batch sizes,
// and regularization parameters. Sufficient data for meaningful comparisons.

// FEATURE DEVELOPMENT:
// Use 500-1000 samples per class when adding new features like data augmentation,
// preprocessing improvements, or training techniques. Balance between speed and accuracy.

// FINAL VALIDATION:
// Use 0 (all samples) for final model training and production deployment.
// Maximum dataset size provides best possible accuracy and robustness.

// PERFORMANCE SCALING WITH SAMPLING:

// TRAINING TIME SCALING:
// - 10 samples/class: ~1% of full training time
// - 100 samples/class: ~10% of full training time  
// - 1000 samples/class: ~75% of full training time
// - All samples: 100% (baseline)

// ACCURACY SCALING (approximate):
// - 10 samples/class: 70-80% of full accuracy
// - 100 samples/class: 85-90% of full accuracy
// - 1000 samples/class: 95-98% of full accuracy
// - All samples: 100% (maximum achievable)

// MEMORY USAGE SCALING:
// Proportional to dataset size - smaller samples use less RAM during training,
// making it possible to experiment on resource-constrained systems.

// These defaults provide a flexible foundation for both rapid development
// and production-quality model training, with the sampling feature enabling
// efficient workflows that scale from quick prototypes to final deployment.