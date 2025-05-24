package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// Predict makes a prediction on a single image file using the module's functionality.
//
// WHAT IS PREDICTION/INFERENCE?
// After training is complete, prediction (also called "inference") is how we use
// the trained neural network to make educated guesses about new data it has never
// seen before. This is the practical application of all the training work.
//
// THE PREDICTION PROCESS:
// 1. Load and preprocess the input image (same preprocessing as training data)
// 2. Feed the processed image through the trained neural network
// 3. Get probability scores for each possible character class
// 4. Find the class with the highest probability as our prediction
// 5. Return both the predicted character and confidence level
//
// CONSISTENCY IS CRITICAL:
// The image must be processed EXACTLY the same way as training images:
// - Same dimensions (28×28)
// - Same grayscale conversion
// - Same normalization (0.0-1.0 range)
// - Same flattening (2D image → 1D array)
// Any difference in preprocessing will cause poor predictions even with a perfectly trained model.
func (ic *ImageClassifier) Predict(imagePath string) (string, float64, error) {
	// STEP 1: Load and process the image file
	// This applies the same preprocessing pipeline used during training
	pixels, err := ic.loadAndProcessImage(imagePath)
	if err != nil {
		return "", 0, fmt.Errorf("failed to load image: %w", err)
	}

	// IMAGE PREPROCESSING RECAP:
	// The loadAndProcessImage function:
	// 1. Opens the image file (PNG or JPEG)
	// 2. Decodes the image data
	// 3. Converts color to grayscale using luminance formula
	// 4. Normalizes pixel values to 0.0-1.0 range
	// 5. Flattens 28×28 image to 784-element array
	// 6. Validates that result matches expected dimensions

	// STEP 2: Convert to dataset format for the neural network
	// The graymatter library expects data in DataSet format, even for single predictions
	inputs := [][]float64{pixels}                    // Wrap single image in batch format
	dataset, err := graymatter.NewDataSet(inputs, [][]float64{{0}}) // Dummy output (not used for prediction)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create input dataset: %w", err)
	}

	// WHY DUMMY OUTPUT?
	// The DataSet constructor requires both inputs and outputs, but during prediction
	// we don't know the correct answer (that's what we're trying to predict!).
	// We provide a dummy output that gets ignored during prediction.

	// STEP 3: Run the image through the trained neural network
	// This is where the actual "intelligence" happens - the network uses its
	// learned weights to transform pixel values into class probabilities
	predictions, err := ic.network.Predict(dataset.Inputs)
	if err != nil {
		return "", 0, fmt.Errorf("prediction failed: %w", err)
	}

	// UNDERSTANDING THE NEURAL NETWORK OUTPUT:
	// The network returns a matrix where each row corresponds to one input image
	// and each column corresponds to one character class. Since we're predicting
	// on a single image, we have one row with 62 columns (one per character).
	//
	// EXAMPLE OUTPUT (simplified to 4 classes):
	// [0.05, 0.02, 0.91, 0.02]
	// This means:
	// - 5% confidence it's class 0 (maybe "A")
	// - 2% confidence it's class 1 (maybe "B")  
	// - 91% confidence it's class 2 (maybe "C") ← Our prediction!
	// - 2% confidence it's class 3 (maybe "D")

	// STEP 4: Find the class with highest probability (argmax operation)
	_, cols := predictions.Dims()
	maxProb := 0.0
	maxIndex := 0

	// Search through all class probabilities to find the maximum
	for j := 0; j < cols; j++ {
		prob := predictions.At(0, j) // Row 0 (our single image), column j (class j)
		if prob > maxProb {
			maxProb = prob
			maxIndex = j
		}
	}

	// ARGMAX EXPLANATION:
	// "Argmax" is short for "argument of the maximum" - it finds the INDEX
	// of the maximum value, not the maximum value itself. If the highest
	// probability is 0.91 at position 2, argmax returns 2 (the index).

	// STEP 5: Convert numerical index back to character
	// The network outputs numerical indices, but humans want character names
	className, exists := IndexToClass[maxIndex]
	if !exists {
		return "", 0, fmt.Errorf("unknown class index: %d", maxIndex)
	}

	// REVERSE MAPPING:
	// During training, we mapped "A" → 0, "B" → 1, etc.
	// During prediction, we reverse this: 0 → "A", 1 → "B", etc.
	// The IndexToClass map was built during initialization for this purpose.

	// STEP 6: Return prediction and confidence
	return className, maxProb, nil
}

// INTERPRETING PREDICTION RESULTS:

// CONFIDENCE LEVELS:
// - 0.9+ (90%+): Very confident, likely correct
// - 0.7-0.9 (70-90%): Confident, probably correct
// - 0.5-0.7 (50-70%): Moderately confident, could be wrong
// - 0.3-0.5 (30-50%): Low confidence, likely wrong
// - <0.3 (<30%): Very uncertain, probably wrong

// IN PRODUCTION SYSTEMS:
// You might set confidence thresholds:
// - Above 90%: Auto-accept the prediction
// - 70-90%: Flag for human review
// - Below 70%: Reject and ask for manual input

// COMMON PREDICTION PROBLEMS:

// 1. LOW CONFIDENCE ON GOOD IMAGES:
// - Network wasn't trained enough
// - Training data doesn't match prediction data
// - Network architecture is too simple for the problem

// 2. HIGH CONFIDENCE ON WRONG PREDICTIONS:
// - Network is overconfident (common problem)
// - Training data had systematic biases
// - Need better regularization or more diverse training data

// 3. INCONSISTENT PREDICTIONS:
// - Preprocessing differences between training and prediction
// - Non-deterministic behavior (rare, but possible with some libraries)
// - Model wasn't saved/loaded correctly

// DEBUGGING PREDICTION ISSUES:

// 1. VISUALIZE THE INPUT:
// Save the processed pixel values as an image to verify preprocessing
// is working correctly. The processed image should look like training data.

// 2. CHECK PROBABILITY DISTRIBUTION:
// Print all class probabilities, not just the maximum. Are they reasonable?
// Is the network very uncertain (all probabilities around 0.016 = 1/62)?

// 3. TEST ON TRAINING DATA:
// Try predicting on images from your training set. If these fail,
// the problem is with model saving/loading, not generalization.

// 4. GRADUAL COMPLEXITY:
// Start with very clear, high-quality images before testing on
// challenging or ambiguous examples.

// This prediction function represents the culmination of the entire machine
// learning pipeline - taking a raw image and producing a structured,
// actionable result that can be used in applications.