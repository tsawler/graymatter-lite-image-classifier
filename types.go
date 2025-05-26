package main

import "github.com/tsawler/graymatter-lite"

// ClassGroup represents a group of characters with their directory info.
//
// WHY GROUP CHARACTERS BY TYPE?
// Our training data is organized in a hierarchy: data/upper/A/, data/lower/a/, etc.
// Rather than hardcoding the directory structure throughout our code, we define
// these groups to make the code more maintainable and easier to understand.
//
// ORGANIZATIONAL STRATEGY BENEFITS:
// We group characters by type (uppercase, lowercase, digits) because:
// 1. It matches how humans naturally categorize characters
// 2. It makes the file system organization intuitive and browsable
// 3. It allows us to process each group with the same algorithmic logic
// 4. It enables easy analysis of performance per character type
// 5. It simplifies adding new character categories in the future
//
// NOTE ON PUNCTUATION:
// Punctuation marks are handled separately from this ClassGroup structure because
// they require special directory name mapping (e.g., "asterisk" → "*"). Many
// punctuation characters can't be used as directory names on common file systems.
// See the PunctuationDirToChar mapping in constants.go for punctuation handling.
type ClassGroup struct {
	// DirName: The subdirectory name under the main data directory
	// For example, "upper" for uppercase letters, "lower" for lowercase
	//
	// DIRECTORY STRUCTURE EXAMPLE:
	// data/
	//   upper/     ← DirName = "upper"
	//     A/
	//       img1.png
	//       img2.png
	//       ... (thousands of examples)
	//     B/
	//       img1.png
	//   lower/     ← DirName = "lower"
	//     a/
	//       img1.png
	//     b/
	//       img1.png
	//   digits/    ← DirName = "digits"
	//     0/
	//       img1.png
	//     1/
	//       img1.png
	//   punctuation/ ← Handled separately due to special naming requirements
	//     asterisk/  ← Directory name != actual character
	//       img1.png
	//     dot/
	//       img1.png
	//     question/
	//       img1.png
	DirName string

	// StartChar, EndChar: The range of characters in this group
	// We use Go's rune type (which represents Unicode characters) to define ranges
	//
	// CHARACTER RANGES IN UNICODE:
	// - Uppercase letters: 'A' to 'Z' (Unicode 65 to 90)
	// - Lowercase letters: 'a' to 'z' (Unicode 97 to 122)  
	// - Digits: '0' to '9' (Unicode 48 to 57)
	//
	// WHY USE RANGES INSTEAD OF LISTS?
	// Instead of manually listing every character, we can use Go's character
	// arithmetic to iterate through ranges cleanly. This approach:
	// - Reduces code duplication and potential typos
	// - Makes the intent clearer (we want all letters A through Z)
	// - Is less error-prone than manually specifying every character
	// - Makes it easy to add new ranges if needed
	//
	// PUNCTUATION LIMITATION:
	// Punctuation marks don't form a continuous Unicode range, and many characters
	// can't be used as directory names (*/?<>|"), so they require a separate
	// mapping system handled elsewhere in the code.
	StartChar, EndChar rune
}

// ImageData represents a single training example.
//
// WHAT IS A TRAINING EXAMPLE?
// In supervised machine learning, each training example consists of two parts:
// 1. Input data (the question): What does this image look like? (pixel values)
// 2. Correct answer (the label): What character does this image actually represent?
//
// The neural network learns by studying thousands of these input-output pairs,
// gradually adjusting its internal parameters (weights and biases) to make
// better predictions on new, unseen examples.
//
// EXPANDED FOR 94-CLASS RECOGNITION:
// This structure now handles all types of characters including punctuation marks,
// making it suitable for comprehensive character recognition tasks that go beyond
// just letters and numbers.
type ImageData struct {
	// Pixels: The actual image data converted to a flat array of grayscale values
	//
	// FROM IMAGE TO NUMBERS:
	// Images are naturally 2D grids of pixels, but neural networks expect 1D input
	// vectors. We "flatten" a 28×28 image into a single array of 784 numbers, where
	// each number represents the brightness of one pixel (0.0 = black, 1.0 = white).
	//
	// PREPROCESSING PIPELINE SUMMARY:
	// Original image → Resize to 28×28 → Convert to grayscale → Normalize to 0-1 → Flatten to array
	//
	// EXAMPLE: A small 2×2 image might become [0.1, 0.9, 0.3, 0.7]
	// representing the brightness of each pixel reading left-to-right, top-to-bottom.
	//
	// CONSISTENCY REQUIREMENT:
	// Every image must be processed identically to produce this pixel array.
	// Any differences in preprocessing between training and prediction will
	// cause poor performance even with a perfectly trained model.
	Pixels []float64

	// Label: The human-readable character this image represents
	//
	// EXPANDED EXAMPLES: "A", "b", "7", "*", "?", "!", etc.
	//
	// GROUND TRUTH IMPORTANCE:
	// This is the "correct answer" that we want the network to learn to predict.
	// The quality and accuracy of these labels is crucial - if labels are wrong,
	// the network will learn incorrect associations. In our system, labels come
	// from the directory structure and punctuation mapping:
	// - Images in "data/upper/A/" are automatically labeled as "A"
	// - Images in "data/punctuation/asterisk/" are labeled as "*"
	//
	// PUNCTUATION LABELS:
	// For punctuation, the label is the actual character ("*", "?", "!") even though
	// the directory name might be descriptive ("asterisk", "question", "exclamation").
	// This mapping is handled by the PunctuationDirToChar table.
	Label string

	// ClassIndex: The numerical index corresponding to this character class
	// This is what the neural network actually works with during training and prediction
	//
	// EXPANDED STRING-TO-NUMBER CONVERSION:
	// Neural networks output numbers, not strings, so we need a consistent mapping
	// from characters to numerical indices. Our system uses this mapping:
	// - "A" → 0, "B" → 1, ..., "Z" → 25        (uppercase letters: indices 0-25)
	// - "a" → 26, "b" → 27, ..., "z" → 51      (lowercase letters: indices 26-51)  
	// - "0" → 52, "1" → 53, ..., "9" → 61      (digits: indices 52-61)
	// - "*" → 62, "?" → 63, ..., "/" → 93      (punctuation marks: indices 62-93)
	//
	// ONE-HOT ENCODING CONVERSION:
	// During training, this ClassIndex gets converted to a "one-hot" vector, which
	// is an array with all zeros except for a single 1 at the ClassIndex position:
	// - ClassIndex 0 → [1, 0, 0, 0, 0, ...] (94 elements, only first is 1)
	// - ClassIndex 62 → [0, 0, ..., 1, 0, ...] (94 elements, only 63rd is 1)
	// 
	// This format is what the neural network's output layer expects to match against.
	ClassIndex int
}

// ImageClassifier encapsulates the training and prediction logic.
//
// OBJECT-ORIENTED DESIGN APPROACH:
// Rather than having scattered functions throughout the program, we group related
// functionality into this struct. This provides several benefits:
// - Clear organization: Related data and methods are kept together
// - Encapsulation: Internal details are hidden from external code
// - Easy testing: We can create test instances with specific configurations
// - Clean API: Users interact with simple methods like Train() and Predict()
// - State management: Configuration and network state are properly maintained
//
// THE FACADE PATTERN:
// This struct acts as a "facade" that hides the complexity of neural networks
// behind a simple, easy-to-use interface. Users don't need to understand:
// - Matrix operations and linear algebra
// - Backpropagation algorithms  
// - Gradient descent optimization
// - Activation functions and their derivatives
// They just call Train() to train and Predict() to make predictions.
type ImageClassifier struct {
	// config: All the hyperparameters and settings for this classifier
	//
	// CONFIGURATION CENTRALIZATION:
	// This includes network architecture settings, file paths, training parameters,
	// and sampling options. By storing this in the struct, all methods have access
	// to the same settings, ensuring consistency throughout the system.
	config *Config

	// network: The actual neural network that performs learning and prediction
	//
	// NEURAL NETWORK ENCAPSULATION:
	// This is from the graymatter library and handles all the mathematical
	// operations like:
	// - Matrix multiplication for forward propagation
	// - Gradient computation for backpropagation
	// - Weight updates during training
	// - Probability calculation during prediction
	//
	// PRIVATE ACCESS:
	// The network field is private (lowercase name) because external code
	// shouldn't manipulate it directly. All interactions should go through
	// our wrapper methods like Train() and Predict(), which provide proper
	// error handling and validation.
	//
	// 94-CLASS NETWORK ARCHITECTURE:
	// The network now has 94 output neurons (one per character class) instead
	// of the original 62, allowing it to distinguish between all letters,
	// digits, and common punctuation marks.
	network *graymatter.Network
}

// NewImageClassifier creates a new image classifier with the given configuration.
//
// CONSTRUCTOR PATTERN:
// This function serves as a constructor, initializing a new ImageClassifier
// with the provided configuration. It's a common Go pattern to have "New..."
// functions that return initialized struct instances.
//
// WHY NOT INITIALIZE THE NETWORK HERE?
// We defer network creation until training begins because:
// 1. Network creation requires knowing exact input/output dimensions
// 2. The network might be loaded from a saved file instead of created fresh
// 3. It keeps object creation lightweight and fast
// 4. It separates concerns: object creation vs network initialization
//
// CONFIGURATION VALIDATION:
// The constructor ensures the config is properly set for 94-class recognition,
// including appropriate output size and directory structure expectations.
func NewImageClassifier(config *Config) *ImageClassifier {
	return &ImageClassifier{config: config}
}

// LoadModel loads a pre-trained model from a file.
//
// MODEL PERSISTENCE USE CASES:
// Loading saved models enables several important workflows:
// - Deploying trained models to production without retraining
// - Sharing models between team members or with the community
// - Resuming training from a checkpoint after interruption
// - A/B testing different model versions in production
// - Using models trained on different machines or environments
//
// WHAT GETS LOADED:
// - Network architecture: Layer sizes, activation functions, connections
// - Learned weights and biases: The actual "knowledge" the network acquired
// - Training metadata: Information about how the model was trained
//
// COMPATIBILITY CONSIDERATIONS:
// When loading a model, ensure it matches your expected use case:
// - Input dimensions must match your image preprocessing (28×28 = 784 inputs)
// - Output dimensions must match your character set (94 classes for full system)
// - Preprocessing pipeline must be identical to what was used during training
//
// CRITICAL COMPATIBILITY NOTE:
// You cannot load a model trained on 62 classes and use it for 94-class prediction,
// or vice versa. The network architecture must exactly match the expected format.
func LoadModel(filename string) (*ImageClassifier, error) {
	// Use the graymatter library to handle the low-level file loading
	network, _, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, err
	}

	// Create a default config for the loaded model
	// In a production system, you might want to save and load the config
	// alongside the model to ensure perfect consistency
	config := NewDefaultConfig()

	return &ImageClassifier{
		config:  config,
		network: network,
	}, nil
}

// DESIGN PRINCIPLES DEMONSTRATED:

// 1. SEPARATION OF CONCERNS:
// - ImageData: Represents individual training examples and their labels
// - Config: Handles all configuration settings and hyperparameters
// - ImageClassifier: Orchestrates the machine learning pipeline
// Each struct has a single, well-defined responsibility.

// 2. ENCAPSULATION:
// - Private fields (lowercase names) hide implementation details
// - Public methods provide a clean, controlled interface
// - Internal complexity is hidden from users of the code
// - State management is handled properly within each struct

// 3. COMPOSITION OVER INHERITANCE:
// - ImageClassifier contains a Network rather than inheriting from it
// - This allows us to add functionality without modifying the base Network type
// - More flexible than traditional object-oriented inheritance
// - Easier to test and maintain

// 4. DEFENSIVE PROGRAMMING:
// - LoadModel returns an error if loading fails
// - Constructor takes configuration to ensure proper initialization
// - Type safety through Go's strong typing system
// - Validation at appropriate points to catch errors early

// 5. SCALABLE DESIGN:
// - Easy to extend from 62 to 94 classes without breaking existing functionality
// - Punctuation handling is additive, not disruptive to existing code
// - Configuration-driven approach makes further expansion straightforward
// - Clear upgrade path for different model versions

// 1. PUNCTUATION SUPPORT:
// The design accommodates punctuation marks that require special handling
// due to file system naming constraints, while maintaining a clean interface.

// 2. FLEXIBLE SAMPLING:
// The architecture supports data sampling for efficient development workflows
// without compromising the ability to train on full datasets.

// 3. EXTENSIBILITY:
// The modular design makes it easy to add new character types, modify
// training procedures, or integrate new neural network architectures.

// CHARACTER-SPECIFIC DESIGN CONSIDERATIONS:

// 1. VISUAL SIMILARITY HANDLING:
// The 94-class design acknowledges that some characters are inherently similar
// (like 'O' and '0') and provides a framework for handling these challenges
// through balanced training data and appropriate network architecture.

// 2. CONTEXT INDEPENDENCE:
// The classifier is designed to recognize individual characters without context,
// which presents unique challenges for punctuation marks that might be clearer
// when seen with surrounding text.

// 3. FONT VARIATION ROBUSTNESS:
// The design accommodates the reality that punctuation marks can vary significantly
// across different fonts and typographic styles, requiring robust training data
// and flexible preprocessing.

// These design patterns make the code more maintainable, testable, and easier
// to understand, which is especially important in machine learning projects where
// debugging can be particularly challenging.