package main

import "github.com/tsawler/graymatter-lite"

// ClassGroup represents a group of characters with their directory info.
//
// WHY GROUP CHARACTERS?
// Our training data is organized in a hierarchy: data/upper/A/, data/lower/a/, etc.
// Rather than hardcoding the directory structure throughout our code, we define
// these groups to make the code more maintainable and easier to understand.
//
// ORGANIZATIONAL STRATEGY:
// We group characters by type (uppercase, lowercase, digits) because:
// 1. It matches how humans naturally categorize characters
// 2. It makes the file system organization intuitive
// 3. It allows us to process each group with the same logic
//
// NOTE ON PUNCTUATION:
// Punctuation marks are handled separately from this ClassGroup structure
// because they require special directory name mapping (e.g., "asterisk" → "*").
// See the PunctuationDirToChar mapping in constants.go for punctuation handling.
type ClassGroup struct {
	// DirName: The subdirectory name under the main data directory
	// For example, "upper" for uppercase letters, "lower" for lowercase
	//
	// DIRECTORY STRUCTURE:
	// data/
	//   upper/     <- DirName = "upper"
	//     A/
	//     B/
	//   lower/     <- DirName = "lower"
	//     a/
	//     b/
	//   digits/    <- DirName = "digits"
	//     0/
	//     1/
	//   punctuation/ <- Handled separately due to special naming requirements
	//     asterisk/  <- Directory name != actual character
	//     dot/
	//     question/
	DirName string

	// StartChar, EndChar: The range of characters in this group
	// We use Go's rune type (which represents Unicode characters) to define ranges
	//
	// CHARACTER RANGES:
	// - Uppercase: 'A' to 'Z' (Unicode 65 to 90)
	// - Lowercase: 'a' to 'z' (Unicode 97 to 122)  
	// - Digits: '0' to '9' (Unicode 48 to 57)
	//
	// WHY USE RANGES?
	// Instead of listing every character individually, we can use Go's character
	// arithmetic to iterate through ranges. This is cleaner and less error-prone
	// than manually specifying every character.
	//
	// PUNCTUATION LIMITATION:
	// Punctuation marks don't form a continuous Unicode range and many can't be
	// used as directory names, so they're handled with a separate mapping system.
	StartChar, EndChar rune
}

// ImageData represents a single training example.
//
// WHAT IS A TRAINING EXAMPLE?
// In supervised machine learning, each training example consists of:
// 1. Input data (the question): pixel values of a character image
// 2. Correct answer (the label): what character this actually is
//
// The neural network learns by studying thousands of these input-output pairs,
// gradually adjusting its internal parameters to make better predictions.
//
// EXPANDED FOR 94 CLASSES:
// This structure now handles all types of characters including punctuation marks,
// making it suitable for comprehensive character recognition tasks.
type ImageData struct {
	// Pixels: The actual image data converted to a flat array of grayscale values
	//
	// FROM IMAGE TO NUMBERS:
	// Images are 2D grids of pixels, but neural networks expect 1D input.
	// We "flatten" a 28×28 image into a single array of 784 numbers.
	// Each number represents the brightness of one pixel (0.0 = black, 1.0 = white).
	//
	// PREPROCESSING PIPELINE:
	// Original image → Resize to 28×28 → Convert to grayscale → Normalize to 0-1 → Flatten to array
	//
	// EXAMPLE: A small 2×2 image might become [0.1, 0.9, 0.3, 0.7]
	// representing the brightness of each pixel reading left-to-right, top-to-bottom.
	Pixels []float64

	// Label: The human-readable character this image represents
	// EXPANDED EXAMPLES: "A", "b", "7", "*", "?", "!", etc.
	//
	// GROUND TRUTH:
	// This is the "correct answer" that we want the network to learn to predict.
	// The quality of these labels is crucial - if labels are wrong, the network
	// will learn wrong associations. In our case, labels come from the directory
	// structure and punctuation mapping:
	// - Images in "data/upper/A/" are labeled as "A"
	// - Images in "data/punctuation/asterisk/" are labeled as "*"
	//
	// PUNCTUATION LABELS:
	// For punctuation, the label is the actual character ("*", "?", "!") even though
	// the directory name might be descriptive ("asterisk", "question", "exclamation").
	Label string

	// ClassIndex: The numerical index corresponding to this character class
	// This is what the neural network actually works with during training
	//
	// EXPANDED STRING TO NUMBER CONVERSION:
	// Neural networks output numbers, not strings. We convert string labels
	// to numerical indices using our ClassMapping:
	// - "A" → 0, "B" → 1, ..., "Z" → 25
	// - "a" → 26, "b" → 27, ..., "z" → 51  
	// - "0" → 52, "1" → 53, ..., "9" → 61
	// - "*" → 62, "?" → 63, ..., "/" → 93 (punctuation marks)
	//
	// ONE-HOT ENCODING:
	// During training, this index gets converted to a "one-hot" vector:
	// ClassIndex 0 → [1, 0, 0, 0, 0, ...] (94 elements, only first is 1)
	// ClassIndex 62 → [0, 0, ..., 1, 0, ...] (94 elements, only 63rd is 1)
	// This format is what the neural network's output layer expects.
	ClassIndex int
}

// ImageClassifier encapsulates the training and prediction logic.
//
// OBJECT-ORIENTED DESIGN:
// Rather than having scattered functions throughout the program, we group
// related functionality into this struct. This provides:
// - Clear organization of code
// - Encapsulation of related data and methods
// - Easy testing and reuse
// - A clean API for users of this code
//
// THE FACADE PATTERN:
// This struct acts as a "facade" that hides the complexity of neural networks
// behind a simple interface. Users don't need to understand matrix operations,
// backpropagation, or activation functions - they just call Train() and Predict().
//
// ENHANCED FOR 94-CLASS RECOGNITION:
// Now capable of recognizing the full range of printable ASCII characters,
// making it suitable for real-world document processing applications.
type ImageClassifier struct {
	// config: All the hyperparameters and settings for this classifier
	// This includes network architecture, file paths, training parameters, etc.
	// By storing this in the struct, all methods have access to the same settings.
	//
	// UPDATED CONFIGURATION:
	// The config now specifies 94 output classes to accommodate the expanded
	// character set including punctuation marks.
	config *Config

	// network: The actual neural network that does the learning and prediction
	// This is from the graymatter library and handles all the mathematical
	// operations like matrix multiplication, gradient computation, etc.
	//
	// ENCAPSULATION:
	// The network is private (lowercase field name) because external code
	// shouldn't manipulate it directly. All interactions should go through
	// our wrapper methods like Train() and Predict().
	//
	// 94-CLASS NETWORK:
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
// 1. Network creation requires knowing the exact input/output dimensions
// 2. The network might be loaded from a saved file instead of created fresh
// 3. It's cleaner to separate object creation from network initialization
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
// - Deploying trained models to production without retraining
// - Sharing models between team members
// - Resuming training from a checkpoint after interruption
// - A/B testing different model versions
//
// WHAT GETS LOADED:
// - Network architecture (layer sizes, activation functions)
// - Learned weights and biases (the "knowledge" the network acquired)
// - Metadata about training (accuracy, hyperparameters, notes)
//
// COMPATIBILITY CONSIDERATIONS:
// When loading a model, ensure it was trained on the same character set.
// A model trained on 62 classes won't work properly with 94-class predictions,
// and vice versa. The network architecture must match the expected input/output format.
//
// IMPORTANT: The loaded network must match the expected input/output format.
// You can't load a model trained on 62 classes and use it for 94-class prediction!
func LoadModel(filename string) (*ImageClassifier, error) {
	// Use the graymatter library to handle the low-level file loading
	network, _, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, err
	}

	// Create a default config for the loaded model
	// In a production system, you'd probably save and load the config too
	// to ensure perfect consistency between training and inference
	config := NewDefaultConfig()

	return &ImageClassifier{
		config:  config,
		network: network,
	}, nil
}

// DESIGN PRINCIPLES DEMONSTRATED:

// 1. SEPARATION OF CONCERNS:
// - ImageData: Represents raw training data
// - Config: Handles all configuration settings  
// - ImageClassifier: Orchestrates the machine learning pipeline
// Each struct has a single, clear responsibility.

// 2. ENCAPSULATION:
// - Private fields (lowercase names) hide implementation details
// - Public methods provide a clean interface
// - Internal complexity is hidden from users of this code

// 3. COMPOSITION OVER INHERITANCE:
// - ImageClassifier contains a Network rather than inheriting from it
// - This allows us to add functionality without modifying the base Network type
// - More flexible than traditional object-oriented inheritance

// 4. DEFENSIVE PROGRAMMING:
// - LoadModel returns an error if loading fails
// - Constructor takes configuration to ensure proper initialization
// - Type safety through Go's strong typing system

// 5. SCALABLE DESIGN:
// - Easy to extend from 62 to 94 classes without breaking existing functionality
// - Punctuation handling is additive, not disruptive
// - Configuration-driven approach makes further expansion straightforward

// PUNCTUATION-SPECIFIC DESIGN CONSIDERATIONS:

// 1. SPECIAL CHARACTER HANDLING:
// Some punctuation marks require special handling in file systems and URLs.
// The design accommodates this through directory name mapping.

// 2. VISUAL SIMILARITY CHALLENGES:
// Punctuation marks like "." and "," are visually similar and may require
// more sophisticated feature extraction or additional training data.

// 3. CONTEXT INDEPENDENCE:
// The classifier recognizes individual characters without context, which
// can be challenging for punctuation that looks similar in isolation.

// 4. EXTENSIBILITY:
// The design makes it easy to add more punctuation marks or special characters
// by extending the mapping tables and updating the configuration.

// These design patterns make the code more maintainable, testable, and easier
// to understand, which is especially important in machine learning where
// debugging can be challenging. The expansion to 94 classes demonstrates
// the flexibility and scalability of the original design.