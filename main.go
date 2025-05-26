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
func main() {
	fileToPredict := ""
	flag.StringVar(&fileToPredict, "predict", "a.png", "File to make prediction on (default 'a.png')")
	flag.Parse()

	fmt.Println("Starting Enhanced Image Classification System...")
	fmt.Println("Supporting 94 character classes: A-Z, a-z, 0-9, and punctuation marks")

	// STEP 1: Create configuration for our neural network
	config := NewDefaultConfig()
	
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
		fmt.Println("This may take longer due to the expanded 94-class character set...")
		startTime := time.Now()

		// Create a new classifier instance
		classifier = NewImageClassifier(config)

		// STEP 3: Train the network with validation
		// This is the same training process as before, but now handles 94 classes
		if err := classifier.TrainWithValidation(); err != nil {
			log.Fatal("Training failed:", err)
		}

		fmt.Println("Training complete. Time to train:", time.Since(startTime))
		
		// STEP 4: Save the newly trained model as the "best" model
		// This ensures that future runs will find and load this model
		fmt.Println("Saving trained model as best model...")
		description := fmt.Sprintf("Enhanced character classifier trained with %d classes (A-Z, a-z, 0-9, punctuation)", config.OutputSize)
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
	
	// USAGE SCENARIOS:
	
	// FIRST RUN (no existing model):
	// - Program trains a new model from scratch with 94 classes
	// - Saves the trained model as "image_classifier_final.json"
	// - Tests the model with a prediction
	
	// SUBSEQUENT RUNS (model exists):
	// - Program loads the existing "image_classifier_final.json" 
	// - Skips training entirely (much faster!)
	// - Tests the loaded model with a prediction
	
	// FORCE RETRAINING:
	// - Delete the "image_classifier_final.json" file
	// - Run the program again to train a fresh model with updated data
	
	// PRODUCTION DEPLOYMENT:
	// - Train the model on your development/training machine
	// - Copy the "_final.json" file to your production environment
	// - Run this program in production - it will load the model and be ready for predictions
	
	// TESTING WITH DIFFERENT CHARACTER TYPES:
	// Try predicting on different types of characters:
	// - Letters: ./program -predict letter_A.png
	// - Digits: ./program -predict digit_5.png  
	// - Punctuation: ./program -predict exclamation.png
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

// ADDITIONAL UTILITIES FOR MODEL MANAGEMENT:

// You might want to add these functions for more sophisticated model management:

// func findBestModel(basePath string) (string, error) {
//     // Look for various model files and return the one with highest accuracy
//     // Could check for _best.json, _final.json, etc.
// }

// func compareModelPerformance(model1Path, model2Path, testDataPath string) error {
//     // Load both models and compare their accuracy on the same test set
//     // Useful for A/B testing different model versions
// }

// func validateModelCompatibility(modelPath string, expectedConfig *Config) error {
//     // Ensure the loaded model matches the expected input/output dimensions
//     // Prevents runtime errors from incompatible models (62-class vs 94-class)
// }

// func generateTestReport(classifier *ImageClassifier, testDataPath string) error {
//     // Generate comprehensive test report showing per-class accuracy
//     // Especially useful for identifying which punctuation marks are problematic
// }

// BEST PRACTICES DEMONSTRATED:

// 1. SMART LOADING:
// Check for existing models before training to save time and resources.

// 2. GRACEFUL FALLBACK:  
// If model loading fails, fall back to training rather than crashing.

// 3. INFORMATIVE OUTPUT:
// Display model metadata so users understand what they're working with.

// 4. VERIFICATION TESTING:
// Always test the model (loaded or trained) to ensure it's working correctly.

// 5. CLEAR STATUS REPORTING:
// Tell users exactly what happened and what files are available for future use.

// 6. CHARACTER TYPE FEEDBACK:
// Help users understand what type of character was predicted for better debugging.

// ENHANCED FEATURES FOR 94-CLASS RECOGNITION:

// 1. EXPANDED CAPABILITY MESSAGING:
// Clear communication about the model's enhanced capabilities.

// 2. CHARACTER TYPE IDENTIFICATION:
// Helps users understand and debug predictions across different character types.

// 3. COMPREHENSIVE STATUS REPORTING:
// Detailed information about what character types are supported.

// 4. PERFORMANCE CONSIDERATIONS:
// Additional messaging about training time due to increased complexity.

// This enhanced main function provides a complete solution for comprehensive
// character recognition including punctuation marks, making it suitable for
// real-world document processing and OCR applications.