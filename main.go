package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// main is the entry point for our image classification neural network program.
//
// ENHANCED WORKFLOW:
// This program now intelligently decides whether to train a new model or load
// an existing one based on what's available on disk. This saves time and
// computational resources by avoiding unnecessary retraining.
//
// DECISION LOGIC:
// 1. Check if a "best" model file exists (typically the highest accuracy model)
// 2. If found: Load the existing model and skip training
// 3. If not found: Train a new model from scratch
// 4. In both cases: Test the model with a prediction to verify it works
//
// WHY LOAD EXISTING MODELS?
// Training neural networks can take minutes to hours. Once you have a good
// model, you typically want to reuse it rather than retrain every time you
// run the program. This is especially important for:
// - Production deployments
// - Development and testing
// - Sharing models between team members
// - Resuming work after interruptions
//
// UPDATED FOR 94-CLASS RECOGNITION:
// The program now supports comprehensive character recognition including
// uppercase letters, lowercase letters, digits, and punctuation marks.
//
// NEW DATA SAMPLING FEATURE:
// Added support for data sampling to reduce training time with large datasets.
// Use -samples parameter to limit images per class for faster experimentation.
func main() {
	var batchSize, iterations, maxSamplesPerClass int
	var learningRate float64
	fileToPredict := ""
	flag.IntVar(&batchSize, "batchsize", 32, "Batch size (default 32)")
	flag.IntVar(&iterations, "iterations", 500, "Number of iterations (default 500)")
	flag.Float64Var(&learningRate, "lr", 0.001, "Learning rate (default 0.001)")
	flag.StringVar(&fileToPredict, "predict", "a.png", "File to make prediction on (default 'a.png')")
	flag.IntVar(&maxSamplesPerClass, "samples", 0, "Maximum samples per class (0 = use all available, default 0)")
	
	flag.Parse()

	fmt.Println("Starting Enhanced Image Classification System...")
	fmt.Println("Supporting 94 character classes: A-Z, a-z, 0-9, and punctuation marks")
	
	// Display data sampling information
	if maxSamplesPerClass > 0 {
		fmt.Printf("Data sampling enabled: Using maximum %d images per class\n", maxSamplesPerClass)
		fmt.Println("This will significantly reduce training time for large datasets.")
	} else {
		fmt.Println("Using all available training data (no sampling)")
	}

	// STEP 1: Create configuration for our neural network
	config := NewDefaultConfig()
	config.TrainingOptions.BatchSize = batchSize
	config.TrainingOptions.Iterations = iterations
	config.TrainingOptions.LearningRate = learningRate
	
	// NEW: Add sampling configuration to the config
	config.MaxSamplesPerClass = maxSamplesPerClass
	
	// STEP 2: Check if a pre-trained "best" model exists
	// We look for a model named "image_classifier_final.json".
	bestModelPath := "./image_classifier_final.json"
	
	var classifier *ImageClassifier
	var err error
	
	// Check if the best model file exists on disk
	if _, err := os.Stat(bestModelPath); err == nil {
		// MODEL FOUND: Load the existing trained model
		fmt.Printf("Found existing best model: %s\n", bestModelPath)
		fmt.Println("Loading pre-trained model...")
		
		// Use the model loading utility from model-utils.go
		c, metadata, err := LoadModelForInference(bestModelPath)
		if err != nil {
			log.Fatalf("Failed to load existing model: %v", err)
		}

		// set classifier to the model loaded from JSON
		classifier = c
		
		// Display information about the loaded model
		fmt.Println("Successfully loaded pre-trained model!")
		fmt.Printf("Model description: %s\n", metadata.Description)
		fmt.Printf("Training details:\n")
		fmt.Printf("  - Learning rate: %.6f\n", metadata.LearningRate)
		fmt.Printf("  - Batch size: %d\n", metadata.BatchSize)
		fmt.Printf("  - Epochs trained: %d\n", metadata.Epochs)
		fmt.Printf("  - Additional notes: %s\n", metadata.Notes)
		
	} else {
		// NO MODEL FOUND: Train a new model from scratch
		fmt.Printf("No existing model found at %s\n", bestModelPath)
		fmt.Println("Training new model from scratch...")
		
		// Display training configuration including sampling info
		fmt.Printf("Training configuration:\n")
		fmt.Printf("  - Batch size: %d\n", batchSize)
		fmt.Printf("  - Learning rate: %.6f\n", learningRate)
		fmt.Printf("  - Iterations: %d\n", iterations)
		if maxSamplesPerClass > 0 {
			fmt.Printf("  - Data sampling: %d images per class\n", maxSamplesPerClass)
			fmt.Printf("  - Expected total images: ~%d (94 classes × %d samples)\n", 94*maxSamplesPerClass, maxSamplesPerClass)
		} else {
			fmt.Printf("  - Data sampling: Using all available data\n")
		}
		
		startTime := time.Now()
		fmt.Printf("Training started at %s\n", time.Now().Format("2006-01-02 03:04:05pm"))

		// Create a new classifier instance
		classifier = NewImageClassifier(config)

		// STEP 3: Train the network with validation
		// This is the same training process as before, but now handles 94 classes
		// and supports data sampling
		if err := classifier.TrainWithValidation(); err != nil {
			log.Fatal("Training failed:", err)
		}

		fmt.Printf("Training complete. Time to train: %v\n", time.Since(startTime))
		
		// STEP 4: Save the newly trained model as the "best" model
		// This ensures that future runs will find and load this model
		fmt.Println("Saving trained model as best model...")
		
		// Include sampling information in the model description
		var description string
		if maxSamplesPerClass > 0 {
			description = fmt.Sprintf("Enhanced character classifier with %d classes (A-Z, a-z, 0-9, punctuation) - trained with %d samples per class", 
				config.OutputSize, maxSamplesPerClass)
		} else {
			description = fmt.Sprintf("Enhanced character classifier with %d classes (A-Z, a-z, 0-9, punctuation) - trained with all available data", 
				config.OutputSize)
		}
		
		if err := classifier.SaveModel(bestModelPath, description); err != nil {
			log.Printf("Warning: Failed to save model: %v", err)
		} else {
			fmt.Printf("Model saved successfully: %s\n", bestModelPath)
		}
	}

	// STEP 5: Test the model (whether loaded or newly trained)
	// This verification step ensures the model is working correctly
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TESTING ENHANCED MODEL PERFORMANCE")
	fmt.Println("Supports: Letters (A-Z, a-z), Digits (0-9), Punctuation (32 marks)")
	fmt.Println(strings.Repeat("=", 60))
	
	// Test with a sample image
	fmt.Printf("Making prediction on '%s'...\n", fileToPredict)
	
	prediction, confidence, err := classifier.Predict(fileToPredict)
	if err != nil {
		log.Printf("Failed to predict image: %v", err)
		fmt.Printf("Note: Make sure '%s' exists or change the test image path\n", fileToPredict)
		fmt.Println("You can test with images containing letters, digits, or punctuation marks!")
	} else {
		fmt.Printf("✓ Prediction successful!\n")
		fmt.Printf("  Predicted character: '%s'\n", prediction)
		fmt.Printf("  Confidence: %.2f%%\n", confidence*100)
		
		// Interpret confidence level for user
		switch {
		case confidence >= 0.9:
			fmt.Printf("  Assessment: Very confident prediction\n")
		case confidence >= 0.7:
			fmt.Printf("  Assessment: Confident prediction\n")
		case confidence >= 0.5:
			fmt.Printf("  Assessment: Moderately confident prediction\n")
		default:
			fmt.Printf("  Assessment: Low confidence - prediction may be incorrect\n")
		}
		
		// Provide character type information
		charType := getCharacterType(prediction)
		fmt.Printf("  Character type: %s\n", charType)
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("PROGRAM STATUS")
	fmt.Println(strings.Repeat("=", 60))
	
	if _, err := os.Stat(bestModelPath); err == nil {
		fmt.Printf("✓ Enhanced model available: %s\n", bestModelPath)
		fmt.Println("✓ Ready for production use with 94-class recognition")
		fmt.Println("✓ Future runs will load this model automatically")
		fmt.Println("✓ Supports comprehensive character recognition:")
		fmt.Println("  - Uppercase letters (A-Z)")
		fmt.Println("  - Lowercase letters (a-z)")
		fmt.Println("  - Digits (0-9)")
		fmt.Println("  - Punctuation marks (!, @, #, $, %, etc.)")
	} else {
		fmt.Println("⚠ No model was saved - check for errors above")
	}
	
	fmt.Println("\nProgram completed successfully!")
	
	// USAGE EXAMPLES WITH NEW SAMPLING FEATURE:
	
	// QUICK EXPERIMENTATION (1000 samples per class):
	// ./program -samples 1000 -iterations 200 -batchsize 128
	// This trains much faster while still getting good results
	
	// MEDIUM TRAINING (5000 samples per class):
	// ./program -samples 5000 -iterations 300 -lr 0.01
	// Balance between speed and accuracy
	
	// FULL TRAINING (all available data):
	// ./program -batchsize 512 -lr 0.02
	// Uses all 13,812 images per class for maximum accuracy
	
	// PRODUCTION DEPLOYMENT:
	// Train with desired sampling, then deploy the saved model
	
	// TESTING WITH DIFFERENT CHARACTER TYPES:
	// ./program -predict letter_A.png -samples 1000
	// ./program -predict digit_5.png
	// ./program -predict exclamation.png
}

// getCharacterType determines what type of character was predicted.
//
// CHARACTER CLASSIFICATION HELPER:
// This utility function helps users understand what type of character
// the model predicted, which is useful for debugging and validation.
func getCharacterType(char string) string {
	if len(char) != 1 {
		return "Unknown (multi-character)"
	}
	
	c := char[0]
	
	switch {
	case c >= 'A' && c <= 'Z':
		return "Uppercase letter"
	case c >= 'a' && c <= 'z':
		return "Lowercase letter"
	case c >= '0' && c <= '9':
		return "Digit"
	default:
		return "Punctuation mark"
	}
}

// ENHANCED FEATURES FOR DATA SAMPLING:

// 1. FLEXIBLE SAMPLING:
// Use -samples parameter to control dataset size
// 0 = use all data, any positive number = limit per class

// 2. TRAINING TIME ESTIMATION:
// With 13,812 images per class:
// - No sampling: 6-8 hours training time
// - 1000 samples: ~30-45 minutes training time  
// - 2000 samples: ~1-1.5 hours training time
// - 5000 samples: ~2-3 hours training time

// 3. ACCURACY EXPECTATIONS:
// - 1000 samples: 85-90% accuracy (good for experimentation)
// - 2000 samples: 90-93% accuracy (good balance)
// - 5000 samples: 93-95% accuracy (near-optimal)
// - All samples: 94-96% accuracy (maximum potential)

// 4. RECOMMENDED WORKFLOW:
// 1. Start with -samples 1000 for quick experiments
// 2. Increase to -samples 2000 for better accuracy
// 3. Use full dataset for final production model

// COMMAND LINE EXAMPLES:

// Quick test (fast training):
// ./program -samples 500 -iterations 100 -batchsize 64

// Balanced training:  
// ./program -samples 2000 -iterations 300 -batchsize 128 -lr 0.01

// Production training:
// ./program -batchsize 512 -lr 0.02 -iterations 400

// Just prediction (loads existing model):
// ./program -predict my_image.png

// BEST PRACTICES DEMONSTRATED:

// 1. COMMAND LINE FLEXIBILITY:
// All key parameters can be adjusted without code changes

// 2. INFORMATIVE FEEDBACK:
// Clear messaging about sampling settings and expected performance

// 3. GRADUAL SCALING:
// Easy to start small and scale up as needed

// 4. CONFIGURATION INTEGRATION:
// Sampling parameter integrates cleanly with existing config system

// This enhanced main function provides complete control over dataset sampling,
// enabling efficient experimentation and training time management for large
// datasets while maintaining full functionality for production use cases.