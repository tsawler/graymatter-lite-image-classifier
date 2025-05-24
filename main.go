package main

import (
	"fmt"
	"log"
)

// main is the entry point for our image classification neural network training program.
//
// WHAT THIS PROGRAM DOES:
// This program trains a neural network to recognize handwritten characters (A-Z, a-z, 0-9).
// It's similar to how postal services automatically read ZIP codes or how banks
// process handwritten checks. The network learns from thousands of example images
// and their correct labels, then can make predictions on new images it's never seen.
//
// THE MACHINE LEARNING PIPELINE:
// 1. Load and preprocess training images (convert to numbers the network can understand)
// 2. Create a neural network with the right architecture for this problem
// 3. Train the network using backpropagation (show it examples and correct mistakes)
// 4. Test the trained network on a new image to verify it learned correctly
//
// WHY NEURAL NETWORKS FOR IMAGE RECOGNITION?
// Traditional programming can't handle the variability in handwriting - everyone writes
// differently, images have noise, lighting varies, etc. Neural networks excel at
// finding patterns in messy, high-dimensional data like images. They can learn to
// ignore irrelevant variations while focusing on the essential features that
// distinguish an 'A' from a 'B'.
func main() {
	fmt.Println("Starting Image Classification Training...")

	// STEP 1: Create configuration for our neural network
	// This defines the architecture and training parameters
	config := NewDefaultConfig()
	
	// STEP 2: Create an ImageClassifier instance
	// This encapsulates all the logic for loading data, training, and making predictions
	classifier := NewImageClassifier(config)

	// STEP 3: Train the network with validation
	// This is where the actual machine learning happens. The network will:
	// - Load thousands of character images from the data directory
	// - Learn to associate pixel patterns with character labels
	// - Continuously improve its predictions through many iterations
	// - Track its progress on both training and validation data
	//
	// WHAT IS VALIDATION?
	// We hold back some data that the network doesn't train on, then test it
	// on this "validation" data to see how well it generalizes to new examples.
	// This helps us detect overfitting (memorizing training data without learning
	// general patterns).
	if err := classifier.TrainWithValidation(); err != nil {
		log.Fatal("Training failed:", err)
	}

	// STEP 4: Test the trained network on a single image
	// Now that training is complete, let's see if our network can correctly
	// identify a character in a new image. This simulates real-world usage
	// where you'd feed the trained model new data for prediction.
	fmt.Println("\nMaking prediction on 'a.png'...")
	
	// The Predict method will:
	// 1. Load the image file and convert it to the same format used during training
	// 2. Run it through the trained neural network
	// 3. Return the predicted character and confidence level
	prediction, confidence, err := classifier.Predict("a.png")
	if err != nil {
		log.Printf("Failed to predict image: %v", err)
	} else {
		// Display the results in a human-readable format
		// Confidence is returned as a decimal (0.0 to 1.0), so we multiply by 100 for percentage
		fmt.Printf("Prediction: '%s' (confidence: %.2f%%)\n", prediction, confidence*100)
		
		// INTERPRETING THE RESULTS:
		// - High confidence (>90%): Network is very sure of its prediction
		// - Medium confidence (70-90%): Network thinks this is likely correct
		// - Low confidence (<50%): Network is uncertain, prediction may be wrong
		//
		// In production systems, you might set confidence thresholds and
		// flag low-confidence predictions for human review.
	}

	fmt.Println("Program completed successfully!")
	
	// AT THIS POINT:
	// - The neural network has been trained on your dataset
	// - Training metrics and plots have been generated (if plotting was enabled)
	// - The trained model has been saved to disk (if saving was enabled)
	// - You've verified the network can make predictions on new data
	//
	// NEXT STEPS:
	// - Evaluate the model on a larger test set to get robust performance metrics
	// - Experiment with different network architectures or hyperparameters
	// - Deploy the trained model to a production environment
	// - Collect more training data if performance isn't satisfactory
}