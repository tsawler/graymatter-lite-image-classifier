package main

import "github.com/tsawler/graymatter-lite"

// Config holds all configuration parameters for our image classification neural network.
//
// WHY CENTRALIZED CONFIGURATION MATTERS:
// Neural networks have many "hyperparameters" - settings that control how the network
// is structured and how it learns. These parameters aren't learned during training;
// you have to set them beforehand. Having good hyperparameter values can be the
// difference between a network that achieves 95% accuracy and one that barely works at 60%.
//
// CONFIGURATION PHILOSOPHY:
// Rather than scattering magic numbers and hardcoded values throughout the program,
// we centralize all settings in this configuration struct. This approach provides:
// - Easy experimentation with different parameter values
// - Consistency across the entire program (everyone uses the same settings)
// - Clear documentation of what each parameter does
// - Version control for different experimental setups
// - Easy deployment with different configurations for dev/test/production
//
// DATA SAMPLING:
// The configuration includes flexible data sampling capabilities that allow
// developers to work with subsets of training data for faster development cycles
// and more efficient experimentation workflows.
type Config struct {
	// IMAGE DIMENSIONS
	// These define the size of input images our network expects

	// ImageWidth, ImageHeight: Dimensions of input images in pixels
	//
	// WHY 28x28 PIXELS?
	// This dimension is a sweet spot for character recognition, popularized by
	// the famous MNIST dataset. It's large enough to capture essential character
	// features and details, but small enough to be computationally efficient.
	// Each character is centered and normalized to fit within this square.
	//
	// CONSISTENCY IS ABSOLUTELY CRUCIAL:
	// ALL images (training, validation, and prediction) must be exactly this size.
	// The preprocessing pipeline automatically resizes images to match these
	// dimensions, but the network architecture is built around this specific size.
	ImageWidth, ImageHeight int

	// NETWORK ARCHITECTURE
	// These parameters define the internal structure of our neural network

	// InputSize: Total number of input neurons (typically width × height for images)
	// For 28×28 images, this equals 784 neurons - one for each pixel
	//
	// WHY FLATTEN 2D IMAGES TO 1D?
	// Neural networks expect 1-dimensional input vectors, but images are naturally
	// 2-dimensional grids. We "flatten" the 28×28 image into a single row of 784
	// pixel values. The network will learn to recognize spatial patterns even in
	// this flattened format, though more advanced architectures (like CNNs) can
	// work directly with 2D structure.
	InputSize int

	// HiddenSize: Number of neurons in each hidden layer
	//
	// HIDDEN LAYERS: THE BRAIN OF THE NETWORK
	// Hidden layers are the intermediate layers between input and output that
	// discover patterns in the data. They're called "hidden" because we can't
	// directly observe what they learn - they develop their own internal
	// representations of the input patterns.
	//
	// SIZE CONSIDERATIONS:
	// - Too small (e.g., 10 neurons): Network lacks capacity to learn complex patterns
	// - Too large (e.g., 5000 neurons): Risk of overfitting and slow training
	// - Just right (128 neurons): Good balance for character recognition tasks
	//
	// RULE OF THUMB: Start with a size between your input size (784) and output
	// size (94). The value 128 has proven effective for this type of problem.
	HiddenSize int

	// OutputSize: Number of output neurons (one for each possible character class)
	// UPDATED: 26 (A-Z) + 26 (a-z) + 10 (0-9) + 32 (punctuation) = 94 total classes
	//
	// ONE NEURON PER CLASS PRINCIPLE:
	// In multi-class classification, we typically have one output neuron for each
	// possible answer. The network outputs a probability for each class, and we
	// choose the one with the highest probability as our prediction.
	//
	// CHARACTER SET FOR REAL-WORLD USE:
	// The 94-class system includes comprehensive punctuation support,
	// making it suitable for real-world text recognition applications:
	// - Standard alphanumeric characters (62 classes)
	// - Common punctuation marks (32 additional classes)  
	// - Symbols used in programming and technical writing
	OutputSize int

	// DATA SAMPLING CONFIGURATION
	// These parameters control how much training data to use - a key feature for
	// efficient development workflows

	// SamplesPerClass: Number of images to use per character class
	// NEW FEATURE for dramatically improved development efficiency
	//
	// SAMPLING STRATEGY OPTIONS:
	// - 0: Use all available images (full dataset, potentially 1M+ images)
	// - N > 0: Use exactly N randomly selected images per class (balanced sampling)
	//
	// WHY DATA SAMPLING IS TRANSFORMATIVE FOR DEVELOPMENT:
	// - Rapid prototyping: Test model architecture changes in minutes vs hours
	// - Resource management: Train on resource-constrained development machines
	// - Progressive development: Start small, scale up as architecture matures
	// - Debugging: Work with manageable dataset sizes during troubleshooting
	// - Cost efficiency: Reduce cloud computing costs during experimentation
	//
	// SAMPLING ENSURES BALANCED REPRESENTATION:
	// When sampling is enabled, we ensure equal representation across all 94 classes.
	// This prevents bias toward classes with more training examples while maintaining
	// statistical validity of the training process.
	//
	// RECOMMENDED SAMPLING WORKFLOW:
	// - 10-50 samples: Initial architecture testing and rapid iteration
	// - 100-500 samples: Hyperparameter tuning and development validation
	// - 1000-5000 samples: Pre-production accuracy assessment and validation
	// - 0 (all data): Final model training for production deployment
	SamplesPerClass int

	// DATA AND MODEL PATHS
	// These specify where to find training data and where to save trained models

	// DataDir: Directory containing training images organized by class
	// data/
	//   upper/           ← Uppercase letters (A-Z)
	//     A/
	//       img1.png
	//       img2.png
	//       ... (potentially 13,000+ images per character)
	//     B/
	//       img1.png
	//       img2.png
	//       ... (potentially 13,000+ images per character)
	//     [continues for all uppercase letters]
	//   lower/           ← Lowercase letters (a-z)
	//     a/
	//       img1.png
	//       ... (potentially 13,000+ images per character)
	//     b/
	//       img1.png
	//       ... (potentially 13,000+ images per character)
	//     [continues for all lowercase letters]
	//   digits/          ← Numbers (0-9)
	//     0/
	//       img1.png
	//       ... (potentially 13,000+ images per digit)
	//     1/
	//       img1.png
	//       ... (potentially 13,000+ images per digit)
	//     [continues for all digits]
	//   punctuation/     ← Punctuation marks (32 different symbols)
	//     asterisk/      ← * symbol (uses descriptive directory name)
	//       img1.png
	//       ... (potentially 13,000+ images per symbol)
	//     dot/           ← . symbol
	//       img1.png
	//       ... (potentially 13,000+ images per symbol)
	//     question/      ← ? symbol
	//       img1.png
	//       ... (potentially 13,000+ images per symbol)
	//     [continues for all punctuation marks]
	//
	// SAMPLING IMPACT ON DIRECTORY USAGE:
	// When sampling is enabled, the system randomly selects N images from each
	// character's directory, ensuring balanced representation across the full
	// 94-class character set while dramatically reducing training time.
	//
	// WHY THIS HIERARCHICAL STRUCTURE?
	// Organizing images by class in separate folders provides several benefits:
	// - Automatic labeling: The folder name becomes the label for all images inside
	// - Easy manual inspection: You can browse and verify your data visually
	// - Simple data management: Adding new examples is just dropping files in folders
	// - Clear organization: Easy to see how much data you have for each character
	// - Sampling compatibility: Random selection works naturally with this structure
	DataDir string

	// ModelPath: Base filename for saving trained models (without extension)
	// The actual saved files will have descriptive suffixes like "_final.json"
	//
	// MODEL PERSISTENCE IMPORTANCE:
	// After investing time and computational resources in training, you want to
	// preserve the results! Saved models enable:
	// - Production deployment without retraining
	// - Sharing models with team members
	// - A/B testing different model versions
	// - Resuming interrupted training sessions
	// - Building model libraries for different use cases
	ModelPath string

	// VISUALIZATION AND ANALYSIS SETTINGS

	// PlottingURL: Address of the plotting service for generating training analysis charts
	//
	// WHY SEPARATE PLOTTING SERVICE?
	// Training generates lots of valuable data (loss curves, accuracy metrics, weight
	// distributions, confusion matrices) that are best understood through visualization.
	// Rather than building complex plotting into our Go program, we use a separate
	// Python service that specializes in creating publication-quality visualizations.
	//
	// SEPARATION OF CONCERNS BENEFITS:
	// - Keep ML code focused on learning algorithms
	// - Leverage Python's superior plotting ecosystem (matplotlib, seaborn)
	// - Enable independent scaling of compute vs visualization services
	// - Allow different teams to work on ML logic vs visualization features
	PlottingURL string

	// TRAINING CONFIGURATION
	// These parameters control the learning process itself

	// TrainingOptions: Detailed settings for how the neural network learns
	// This includes learning rate, batch size, number of epochs, and more.
	// These hyperparameters are absolutely critical for successful training.
	TrainingOptions graymatter.TrainingOptions
}

// NewDefaultConfig creates a configuration with sensible defaults for character recognition.
//
// CAREFULLY CHOSEN DEFAULTS:
// The values here represent a good starting point for character recognition tasks,
// based on common practices in the field and empirical testing. They should work
// reasonably well for most similar problems, but you may need to adjust them based on:
// - The complexity and size of your specific dataset
// - Available computational resources (CPU, memory, time)
// - Desired trade-offs between training time and accuracy
// - Specific performance requirements for your application
//
// UPDATED FOR 94-CLASS RECOGNITION WITH FLEXIBLE SAMPLING:
// The configuration now supports the full 94-character recognition system including
// comprehensive punctuation support, plus flexible data sampling for dramatically
// improved development efficiency.
//
// EXPERIMENTATION IS ENCOURAGED:
// Don't treat these defaults as unchangeable! Machine learning is fundamentally
// an experimental discipline. Try different values systematically and measure
// the results. The sampling feature makes this experimentation much faster
// and more cost-effective than traditional approaches.
func NewDefaultConfig() *Config {
	// Calculate input size from image dimensions
	imageWidth, imageHeight := 28, 28

	return &Config{
		// IMAGE PREPROCESSING SETTINGS
		ImageWidth:  imageWidth,
		ImageHeight: imageHeight,
		InputSize:   imageWidth * imageHeight, // 784 pixels total (28 × 28)

		// NETWORK ARCHITECTURE DESIGN
		// This creates a feed-forward neural network with the following structure:
		// 784 input neurons → 128 hidden neurons → 128 hidden neurons → 94 output neurons
		//
		// ARCHITECTURE RATIONALE:
		// - Input layer (784): One neuron per pixel in flattened 28×28 image
		// - Hidden layers (128 each): Sufficient capacity for character pattern recognition
		// - Output layer (94): One neuron per character class in expanded set
		HiddenSize: 128, // Good balance between learning capacity and efficiency
		OutputSize: 94,  // All possible characters: letters, digits, punctuation

		// DATA SAMPLING CONFIGURATION
		// Default to using all available data (traditional approach)
		// Users can override this for faster development workflows
		SamplesPerClass: 0, // 0 = use all available images per class

		// FILE SYSTEM PATHS
		DataDir:   "data",             // Look for training images in ./data/ subdirectory
		ModelPath: "image_classifier", // Save trained models with this base name

		// EXTERNAL SERVICES
		PlottingURL: "http://localhost:8080", // Plotting service assumed to run locally

		// TRAINING HYPERPARAMETERS
		// These control the learning process and are critical for good results
		TrainingOptions: graymatter.TrainingOptions{
			// LEARNING SCHEDULE
			Iterations:   500,   // Number of complete passes through all training data
			BatchSize:    32,    // Process 32 images at once before updating weights
			LearningRate: 0.001, // How aggressively to adjust weights when correcting errors

			// MODEL PERSISTENCE
			EnableSaving: true,               // Save the model during and after training
			SavePath:     "image_classifier", // Base name for saved model files
			SaveInterval: 100,                // Save checkpoint every 100 epochs

			// VISUALIZATION AND MONITORING
			EnablePlotting: true,                    // Generate training progress charts
			PlottingURL:    "http://localhost:8080", // Where to send plotting requests
			ShowProgress:   true,                    // Display progress during training
		},
	}
}

// UNDERSTANDING THE HYPERPARAMETERS WITH SAMPLING CONTEXT:

// ITERATIONS (500):
// Each iteration processes the entire training dataset once (called an "epoch").
// 500 iterations means the network sees each training image 500 times total.
// With sampling, the effective dataset size changes dramatically:
// - Full dataset (samples=0): Each of ~1.3M images seen 500 times
// - Large sample (samples=1000): Each of ~94K images seen 500 times  
// - Medium sample (samples=100): Each of ~9.4K images seen 500 times
// - Small sample (samples=10): Each of ~940 images seen 500 times
//
// SAMPLING IMPACT: Smaller datasets may need fewer iterations to avoid overfitting,
// while larger datasets can benefit from more iterations for thorough learning.

// BATCH SIZE (32):
// Instead of updating weights after each individual image, we process 32 images
// and then update weights based on the average error across the batch. This
// approach provides several benefits:
// - More stable learning (less noisy weight updates)
// - Better computational efficiency (vectorized operations)
// - More consistent gradients for reliable learning
//
// The batch size remains appropriate regardless of sampling configuration.

// LEARNING RATE (0.001):
// This controls how aggressively the network adjusts its weights when it makes
// mistakes. We use a conservative learning rate because:
// - With 94 output classes, the network needs to make nuanced distinctions
// - Smaller datasets (via sampling) may be more sensitive to learning rate
// - Conservative rates are generally safer for stable, reliable training
// - Too high: Network becomes unstable and fails to converge
// - Too low: Network learns too slowly and may get stuck

// SAMPLES PER CLASS SCALING EXAMPLES:
// This new parameter dramatically changes training characteristics:
// - 0 (all): Use all available images (~13,000+ per class, ~1.3M total)
// - 10: Use 10 images per class (940 total images, ~99% time reduction)
// - 100: Use 100 images per class (9,400 total images, ~95% time reduction)
// - 1000: Use 1000 images per class (94,000 total images, ~85% time reduction)

// HIDDEN SIZE (128):
// This determines the network's capacity to learn complex patterns. 128 neurons
// provides sufficient capacity to distinguish between 94 different character
// classes while avoiding excessive complexity that could lead to overfitting.
// This size works well across different dataset sizes achieved through sampling.

// OUTPUT SIZE (94):
// This matches our expanded character set:
// - 26 uppercase letters (A-Z): Indices 0-25
// - 26 lowercase letters (a-z): Indices 26-51
// - 10 digits (0-9): Indices 52-61
// - 32 punctuation marks (!, @, #, $, etc.): Indices 62-93

// SAMPLING STRATEGY CONSIDERATIONS FOR DIFFERENT USE CASES:

// RAPID PROTOTYPING (10-50 samples per class):
// - Purpose: Test model architecture, activation functions, layer configurations
// - Training time: Seconds to minutes
// - Use case: Initial development, architecture experiments, debugging
// - Expected accuracy: Lower but sufficient for architectural validation
// - Development velocity: Extremely high, enables rapid iteration

// DEVELOPMENT AND TUNING (100-500 samples per class):
// - Purpose: Hyperparameter optimization, feature development
// - Training time: Minutes to tens of minutes
// - Use case: Learning rate tuning, batch size experiments, preprocessing tests
// - Expected accuracy: Moderate, good for comparative evaluation
// - Development velocity: High, enables systematic experimentation

// VALIDATION AND ASSESSMENT (1000-5000 samples per class):
// - Purpose: Performance evaluation, model comparison, pre-production testing
// - Training time: Tens of minutes to hours
// - Use case: Accuracy benchmarking, model selection, validation testing
// - Expected accuracy: High, approaching full dataset performance
// - Development velocity: Medium, balances thoroughness with efficiency

// PRODUCTION TRAINING (0 = all samples):
// - Purpose: Final model for deployment, maximum accuracy
// - Training time: Hours to days
// - Use case: Production model creation, final optimization, deployment prep
// - Expected accuracy: Maximum achievable with current architecture
// - Development velocity: Low, but provides best possible results

// PERFORMANCE SCALING RELATIONSHIPS:

// TRAINING TIME SCALING (approximate):
// Training time scales roughly linearly with dataset size:
// - 10 samples/class: ~1% of full training time (ultra-fast iteration)
// - 100 samples/class: ~10% of full training time (fast development)
// - 1000 samples/class: ~75% of full training time (thorough validation)
// - All samples: 100% baseline (maximum accuracy training)

// ACCURACY SCALING (typical patterns):
// Accuracy generally increases with more data, but with diminishing returns:
// - 10 samples/class: 70-80% of full dataset accuracy (good for architecture testing)
// - 100 samples/class: 85-90% of full dataset accuracy (good for development)
// - 1000 samples/class: 95-98% of full dataset accuracy (good for validation)
// - All samples: 100% maximum achievable accuracy (production quality)

// MEMORY USAGE SCALING:
// Memory usage scales proportionally with dataset size:
// - Smaller samples enable training on laptops and constrained environments
// - Full datasets may require high-memory servers or cloud instances
// - Sampling makes development accessible to broader range of hardware

// RECOMMENDED DEVELOPMENT WORKFLOW:

// PHASE 1: ARCHITECTURE EXPLORATION (samples=10-50)
// Quickly test different network architectures, activation functions, and
// basic hyperparameters. Focus on getting the pipeline working end-to-end.

// PHASE 2: HYPERPARAMETER TUNING (samples=100-500)
// Systematically optimize learning rates, batch sizes, layer sizes, and
// other training parameters using manageable dataset sizes.

// PHASE 3: VALIDATION AND TESTING (samples=1000-5000)
// Validate the optimized architecture on larger datasets to ensure the
// performance gains will scale to production.

// PHASE 4: PRODUCTION TRAINING (samples=0)
// Train the final model on the complete dataset for deployment, using all
// available data for maximum accuracy and robustness.

// These defaults provide a flexible foundation that supports both rapid development
// cycles and production-quality model training. The sampling feature enables
// efficient workflows that scale from quick prototypes to final deployment,
// dramatically improving the developer experience in machine learning projects.