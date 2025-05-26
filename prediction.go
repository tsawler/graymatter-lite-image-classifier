package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// Predict makes a prediction on a single image file using the module's functionality.
//
// WHAT IS PREDICTION/INFERENCE?
// After training is complete, prediction (also called "inference") is how we use the
// trained neural network to make educated guesses about new data it has never seen
// before. This is the practical application of all the training work - where the
// model demonstrates what it has learned.
//
// THE PREDICTION PROCESS:
// 1. Load and preprocess the input image (using identical preprocessing as training)
// 2. Feed the processed image through the trained neural network
// 3. Get probability scores for each possible character class (0-93)
// 4. Find the class with the highest probability as our best prediction
// 5. Return both the predicted character and the confidence level
//
// CONSISTENCY IS ABSOLUTELY CRITICAL:
// The input image must be processed EXACTLY the same way as training images:
// - Same dimensions (28×28 pixels)
// - Same grayscale conversion method
// - Same normalization range (0.0-1.0)
// - Same flattening process (2D image → 1D array)
// 
// Even tiny differences in preprocessing will cause poor predictions, even with
// a perfectly trained model. This is one of the most common sources of bugs
// in machine learning systems.
func (ic *ImageClassifier) Predict(imagePath string) (string, float64, error) {
	// STEP 1: Load and process the image file
	// This applies exactly the same preprocessing pipeline that was used during training
	pixels, err := ic.loadAndProcessImageForPrediction(imagePath)
	if err != nil {
		return "", 0, fmt.Errorf("failed to load image: %w", err)
	}

	// IMAGE PREPROCESSING RECAP:
	// The loadAndProcessImageForPrediction function performs these steps:
	// 1. Opens the image file (supports PNG and JPEG formats)
	// 2. Decodes the image data into a pixel array
	// 3. Resizes to exactly 28×28 pixels (if not already that size)
	// 4. Converts color to grayscale using perceptual luminance formula
	// 5. Normalizes pixel values to 0.0-1.0 range (0.0=black, 1.0=white)
	// 6. Flattens 28×28 2D image into 784-element 1D array
	// 7. Validates that the result has exactly 784 elements

	// STEP 2: Analyze image polarity and invert if necessary
	// Some images have dark characters on light backgrounds, others have light
	// characters on dark backgrounds. We detect this and standardize to the
	// format the network was trained on.
	isBlackOnWhite := ic.analyzeImagePolarity(pixels)
	if isBlackOnWhite {
		fmt.Println("Detected black character on white background. Inverting image...")
		pixels = ic.invertPixels(pixels)
	} else {
		fmt.Println("Detected white character on black background. No inversion needed.")
	}

	// STEP 3: Save the processed (and potentially inverted) image for debugging
	// This helps verify that preprocessing worked correctly by saving what the
	// network actually "sees" as input
	saveFileName := "prediction_image.png"
	if err := ic.saveProcessedImage(pixels, saveFileName); err != nil {
		fmt.Printf("Warning: Failed to save processed image to %s: %v\n", saveFileName, err)
	} else {
		fmt.Printf("Processed image saved to %s\n", saveFileName)
	}

	// STEP 4: Convert to dataset format for the neural network
	// The graymatter library expects data in DataSet format, even for single predictions
	inputs := [][]float64{pixels}                           // Wrap single image in batch format
	dataset, err := graymatter.NewDataSet(inputs, [][]float64{{0}}) // Dummy output (ignored)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create input dataset: %w", err)
	}

	// WHY DUMMY OUTPUT?
	// The DataSet constructor requires both inputs and outputs, but during prediction
	// we don't know the correct answer (that's what we're trying to predict!).
	// We provide a dummy output value that gets completely ignored during prediction.

	// STEP 5: Run the image through the trained neural network
	// This is where the actual "intelligence" happens - the network uses its learned
	// weights and biases to transform the 784 pixel values into 94 class probabilities
	predictions, err := ic.network.Predict(dataset.Inputs)
	if err != nil {
		return "", 0, fmt.Errorf("prediction failed: %w", err)
	}

	// UNDERSTANDING THE NEURAL NETWORK OUTPUT:
	// The network returns a matrix where each row corresponds to one input image
	// and each column corresponds to one character class. Since we're predicting
	// on a single image, we have one row with 94 columns.
	//
	// EXAMPLE OUTPUT (simplified to 4 classes for illustration):
	// [0.05, 0.02, 0.91, 0.02]
	// This means:
	// - 5% confidence it's class 0 (maybe "A")
	// - 2% confidence it's class 1 (maybe "B")  
	// - 91% confidence it's class 2 (maybe "C") ← Our prediction!
	// - 2% confidence it's class 3 (maybe "D")
	//
	// The softmax activation ensures all probabilities sum to exactly 1.0 (100%).

	// STEP 6: Find the class with highest probability (argmax operation)
	_, cols := predictions.Dims()
	maxProb := 0.0
	maxIndex := 0

	// Search through all 94 class probabilities to find the maximum
	for j := range cols {
		prob := predictions.At(0, j) // Row 0 (our single image), column j (class j)
		if prob > maxProb {
			maxProb = prob
			maxIndex = j
		}
	}

	// ARGMAX EXPLANATION:
	// "Argmax" is short for "argument of the maximum" - it finds the INDEX of the
	// maximum value, not the maximum value itself. If the highest probability is
	// 0.91 at position 2, argmax returns 2 (the index), not 0.91 (the value).

	// STEP 7: Convert numerical index back to human-readable character
	// The network outputs numerical indices (0-93), but humans want character names.
	// We use our reverse mapping to convert the index back to the actual character.
	className, exists := IndexToClass[maxIndex]
	if !exists {
		return "", 0, fmt.Errorf("unknown class index: %d", maxIndex)
	}

	// REVERSE MAPPING EXPLANATION:
	// During training, we mapped characters to indices: "A" → 0, "B" → 1, etc.
	// During prediction, we reverse this mapping: 0 → "A", 1 → "B", etc.
	// The IndexToClass map was built during initialization specifically for this purpose.

	// STEP 8: Return prediction and confidence level
	return className, maxProb, nil
}

// INTERPRETING PREDICTION RESULTS:

// CONFIDENCE LEVELS AND THEIR MEANINGS:
// - 0.9+ (90%+): Very confident prediction, likely to be correct
// - 0.7-0.9 (70-90%): Confident prediction, probably correct but worth double-checking
// - 0.5-0.7 (50-70%): Moderately confident, could be wrong, consider manual review
// - 0.3-0.5 (30-50%): Low confidence, likely wrong, probably needs human intervention
// - <0.3 (<30%): Very uncertain, almost certainly wrong, definitely needs review

// PRODUCTION SYSTEM CONFIDENCE THRESHOLDS:
// In real applications, you might set different confidence thresholds:
// - Above 90%: Auto-accept the prediction and proceed automatically
// - 70-90%: Flag for human review or request additional confirmation
// - Below 70%: Reject the prediction and ask for manual input or better image

// COMMON PREDICTION PROBLEMS AND THEIR CAUSES:

// 1. LOW CONFIDENCE ON GOOD IMAGES:
// Possible causes:
// - Network wasn't trained long enough or with enough data
// - Training data doesn't match the style/format of prediction images
// - Network architecture is too simple for the complexity of the problem
// - Preprocessing differences between training and prediction

// 2. HIGH CONFIDENCE ON WRONG PREDICTIONS:
// Possible causes:
// - Network is overconfident (common problem in neural networks)
// - Training data had systematic biases or labeling errors
// - Need better regularization techniques or more diverse training data
// - Model is overfitting to training data patterns

// 3. INCONSISTENT PREDICTIONS ON SIMILAR IMAGES:
// Possible causes:
// - Preprocessing differences between images
// - Non-deterministic behavior in the network (rare but possible)
// - Model wasn't saved or loaded correctly
// - Images have subtle differences that affect the network

// DEBUGGING PREDICTION ISSUES:

// 1. VISUALIZE THE INPUT:
// Always save the processed pixel values as an image (like we do above) to verify
// that preprocessing is working correctly. The processed image should look similar
// to what the network saw during training.

// 2. CHECK FULL PROBABILITY DISTRIBUTION:
// Don't just look at the maximum probability - examine all class probabilities.
// Are they reasonable? Is the network very uncertain (all probabilities around
// 1/94 ≈ 0.01)? Are there multiple high probabilities suggesting ambiguous input?

// 3. TEST ON TRAINING DATA:
// Try predicting on images from your original training set. If these fail,
// the problem is with model saving/loading or preprocessing, not generalization.

// 4. START WITH CLEAR, HIGH-QUALITY IMAGES:
// Begin testing with very clear, high-contrast images before moving to
// challenging or ambiguous examples. This helps isolate whether problems
// are fundamental or just related to difficult inputs.

// 5. VERIFY PREPROCESSING CONSISTENCY:
// Compare the preprocessing pipeline step-by-step between training and prediction.
// Even small differences can cause major problems.

// CHARACTER-SPECIFIC PREDICTION CHALLENGES:

// 1. VISUALLY SIMILAR CHARACTERS:
// Some character pairs are inherently difficult to distinguish:
// - 'O' vs '0': Often identical in many fonts
// - 'l' vs '1' vs 'I': Can be very similar depending on font
// - '6' vs 'b': Similar shapes, especially in certain orientations
// - '.' vs ',': Very similar punctuation marks

// 2. CONTEXT-DEPENDENT CHARACTERS:
// Some characters are easier to recognize in context:
// - Punctuation marks are often clearer when seen with surrounding text
// - Ambiguous characters like 'O'/'0' can be disambiguated by context
// - Our model works on individual characters without context

// 3. FONT AND STYLE VARIATIONS:
// Different fonts and writing styles can dramatically affect recognition:
// - Serif vs sans-serif fonts
// - Bold vs regular weight
// - Italics vs normal orientation
// - Handwritten vs printed text

// This prediction function represents the culmination of the entire machine learning
// pipeline - taking a raw image file and producing a structured, actionable result
// that can be used in real applications. The comprehensive error handling and
// debugging features make it suitable for production use while remaining
// educational for understanding how neural network inference works.