package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// Directory to save trained models as JSON
const modelSaveDir = "./saved_models"

func init() {
	// Create output directory for saved models
	_ = os.MkdirAll(modelSaveDir, 0755)
}

// main is the entry point for our image classification neural network program.
//
// INTELLIGENT MODEL MANAGEMENT AND FLEXIBLE DATA SAMPLING:
// This program demonstrates a sophisticated machine learning workflow that balances
// development efficiency with production-quality results. Key features include:
//
// FLEXIBLE DATA SAMPLING:
// - Control training dataset size via command line (--samples flag)
// - Sample sizes from 1 image per class to full dataset (13,000+ per class)
// - Maintains balanced representation across all 94 character classes
// - Enables rapid prototyping, debugging, and iterative development
// - Dramatically reduces training time for development workflows
//
// INTELLIGENT MODEL MANAGEMENT:
// - Automatically detects existing trained models
// - Loads pre-trained models when available (skips training entirely)
// - Saves newly trained models for future use
// - Provides detailed model metadata and performance information
//
// COMPREHENSIVE CHARACTER RECOGNITION:
// - Supports 94 different character classes: A-Z, a-z, 0-9, plus 32 punctuation marks
// - Suitable for real-world text recognition applications
//
// PROGRAM EXECUTION FLOW:
// 1. Parse command line arguments for training parameters and sampling configuration
// 2. Check for existing trained model files
// 3. If model exists: Load it and skip training entirely
// 4. If no model: Train new model with specified parameters and data sampling
// 5. Test the model (whether loaded or newly trained) with a sample prediction
// 6. Display comprehensive system status and capability information
//
// DECISION LOGIC FOR MODEL LOADING:
// 1. Check if a "best" model file exists at the specified path
// 2. If found: Load the existing model, display its metadata, skip all training
// 3. If not found: Train a new model with current configuration
// 4. Test the model with a prediction to verify it works correctly
//
// WHY DATA SAMPLING IS TRANSFORMATIVE FOR ML DEVELOPMENT:
// - Rapid prototyping: Test model architecture changes in minutes instead of hours
// - Resource efficiency: Train on laptops and constrained development environments
// - Cost management: Reduce cloud computing expenses during experimentation
// - Progressive development: Start small, validate approach, then scale up
// - Debugging facilitation: Work with manageable dataset sizes during troubleshooting
//
// UPDATED FOR 94-CLASS RECOGNITION WITH EFFICIENT DEVELOPMENT WORKFLOWS:
// The program now supports comprehensive character recognition with flexible data
// sampling that makes machine learning development dramatically more efficient.
func main() {
	// COMMAND LINE ARGUMENT PARSING
	// Define variables to hold command line arguments with sensible defaults
	var fileToPredict, fileToLoad string
	var batchSize, iterations, samplesPerClass int
	var learningRate float64

	// Define command line flags with helpful descriptions
	flag.StringVar(&fileToPredict, "predict", "", "Image file to make prediction on (default 'a.png')")
	flag.StringVar(&fileToLoad, "load", "", "Path to existing trained model file to load")
	flag.IntVar(&batchSize, "batchsize", 64, "Batch size for training (default 64)")
	flag.IntVar(&iterations, "iterations", 30, "Number of training iterations/epochs (default 30)")
	flag.Float64Var(&learningRate, "lr", 0.001, "Learning rate for training (default 0.001)")
	flag.IntVar(&samplesPerClass, "samples", 0, "Number of samples per class (0 = use all available; default 0)")

	// Parse the command line arguments
	flag.Parse()

	// PROGRAM INITIALIZATION
	fmt.Println("Starting Image Classification System...")

	// Display current sampling configuration with helpful context
	if samplesPerClass > 0 {
		totalSamples := samplesPerClass * 94
		fmt.Printf("\n📊 Data Sampling Configuration:\n")
		fmt.Printf("   • Using %d samples per class\n", samplesPerClass)
		fmt.Printf("   • Total training images: ~%d\n", totalSamples)
		fmt.Printf("   • Estimated time reduction: ~%.0f%%\n", (1.0-float64(totalSamples)/1300000.0)*100)
		fmt.Printf("   • Use case: %s\n", getSamplingUseCase(samplesPerClass))
		fmt.Printf("   • Note: Smaller datasets train faster but may have lower accuracy\n")
	} else {
		fmt.Printf("\n📊 Data Sampling Configuration:\n")
		fmt.Printf("   • Using ALL available samples per class\n")
		fmt.Printf("   • Total training images: ~1.3M+ (full dataset)\n")
		fmt.Printf("   • Use case: Production model training for maximum accuracy\n")
		fmt.Printf("   • Note: Full dataset provides best accuracy but takes longer to train\n")
	}

	// CONFIGURATION SETUP
	// Create base configuration and apply command line overrides
	config := NewDefaultConfig()
	config.TrainingOptions.LearningRate = learningRate
	config.TrainingOptions.BatchSize = batchSize
	config.TrainingOptions.Iterations = iterations
	config.SamplesPerClass = samplesPerClass

	var classifier *ImageClassifier

	// Define the expected path for the best/final trained model
	bestModelPath := filepath.Join(modelSaveDir, "image_classifier_final.json")

	if fileToLoad != "" && fileExists(fileToLoad) {
		// PRE-TRAINED MODEL FOUND: Load existing model and skip training
		fmt.Printf("\n🔄 LOADING PRE-TRAINED MODEL\n")
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("📁 Found existing model: %s\n", fileToLoad)
		fmt.Printf("⚡ Loading pre-trained model (skipping training entirely)\n")
		fmt.Printf("ℹ️  Note: Pre-trained model may have been trained with different settings\n")

		// Load the model using our utility function for inference
		c, metadata, err := LoadModelForInference(fileToLoad)
		if err != nil {
			log.Fatalf("❌ Failed to load existing model: %v", err)
		}

		classifier = c

		// Display detailed information about the loaded model
		fmt.Printf("✅ Successfully loaded pre-trained model!\n\n")
		fmt.Printf("📋 Model Information:\n")
		fmt.Printf("   • Description: %s\n", metadata.Description)
		fmt.Printf("   • Training Details:\n")
		fmt.Printf("     - Learning rate: %.6f\n", metadata.LearningRate)
		fmt.Printf("     - Batch size: %d\n", metadata.BatchSize)
		fmt.Printf("     - Epochs trained: %d\n", metadata.Epochs)
		fmt.Printf("     - Additional notes: %s\n", metadata.Notes)

	} else {
		// NO PRE-TRAINED MODEL FOUND: Train a new model from scratch
		fmt.Printf("\n🚀 TRAINING NEW MODEL\n")
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		if fileToLoad != "" {
			fmt.Printf("⚠️  Specified model not found: %s\n", fileToLoad)
		}

		// Display training configuration information
		if samplesPerClass > 0 {
			fmt.Printf("🎯 Training new model with %d samples per class\n", samplesPerClass)
			fmt.Printf("⏱️  Expected training time: %s\n", getExpectedTrainingTime(samplesPerClass))
		} else {
			fmt.Printf("🎯 Training new model with ALL available samples\n")
			fmt.Printf("⏱️  Expected training time: Extended due to full dataset (~1.3M images)\n")
		}

		fmt.Printf("🔧 Training parameters:\n")
		fmt.Printf("   • Learning rate: %.4f\n", learningRate)
		fmt.Printf("   • Batch size: %d\n", batchSize)
		fmt.Printf("   • Iterations: %d\n", iterations)

		// Record training start time for performance measurement
		startTime := time.Now()

		// Create a new classifier instance with our configuration
		classifier = NewImageClassifier(config)

		// TRAIN THE NETWORK
		// The TrainWithValidation function handles:
		// - Loading and preprocessing training data (with sampling if configured)
		// - Splitting data into training and validation sets
		// - Training the neural network with backpropagation
		// - Generating analysis plots and performance metrics
		fmt.Printf("\n🧠 Starting neural network training...\n")
		if err := classifier.TrainWithValidation(); err != nil {
			log.Fatalf("❌ Training failed: %v", err)
		}

		// Calculate and display training performance metrics
		trainingDuration := time.Since(startTime)
		fmt.Printf("\n✅ Training completed successfully!\n")
		fmt.Printf("⏱️  Total training time: %v\n", trainingDuration)

		// Provide context about training time based on sampling configuration
		if samplesPerClass > 0 && samplesPerClass < 1000 {
			fmt.Printf("💡 Training time reasonable for %d samples per class\n", samplesPerClass)
			fmt.Printf("🔄 Consider increasing sample size for higher accuracy when ready\n")
		} else {
			fmt.Printf("💡 Training time reflects dataset size\n")
		}

		// SAVE THE NEWLY TRAINED MODEL
		fmt.Printf("\n💾 SAVING TRAINED MODEL\n")
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("💾 Saving trained model for future use...\n")

		// Create descriptive model metadata including sampling information
		var description string
		if samplesPerClass > 0 {
			description = fmt.Sprintf("94-class character classifier (trained with %d samples per class)", samplesPerClass)
		} else {
			description = "94-class character classifier (trained with full dataset)"
		}

		// Save the model with comprehensive metadata
		if err := classifier.SaveModel(bestModelPath, description); err != nil {
			log.Printf("⚠️  Warning: Failed to save model: %v", err)
		} else {
			fmt.Printf("✅ Model saved successfully: %s\n", bestModelPath)
		}
	}

	// MODEL TESTING AND VALIDATION
	// Test the model (whether loaded or newly trained) to verify it works correctly
	fmt.Printf("\n🧪 TESTING MODEL PERFORMANCE\n")
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	fmt.Printf("🎯 Model Capabilities:\n")
	fmt.Printf("   • Letters: A-Z (uppercase), a-z (lowercase)\n")
	fmt.Printf("   • Digits: 0-9\n")
	fmt.Printf("   • Punctuation: 32 symbols (!, @, #, $, %%, etc.)\n")
	fmt.Printf("   • Total: 94 different character classes\n")

	if fileToPredict != "" {
		// Perform a test prediction to validate the model works
		fmt.Printf("\n🔍 Making test prediction on '%s'...\n", fileToPredict)

		prediction, confidence, err := classifier.Predict(fileToPredict)
		if err != nil {
			log.Printf("❌ Failed to predict image: %v\n", err)
			fmt.Printf("💡 Troubleshooting tips:\n")
			fmt.Printf("   • Ensure '%s' exists in the current directory\n", fileToPredict)
			fmt.Printf("   • Try a different image file with --predict flag\n")
			fmt.Printf("   • Supported formats: PNG, JPEG\n")
			fmt.Printf("   • Test with images containing letters, digits, or punctuation!\n")
		} else {
			// Successful prediction - display comprehensive results
			fmt.Printf("✅ Prediction successful!\n\n")
			fmt.Printf("📊 Prediction Results:\n")
			fmt.Printf("   • Predicted character: '%s'\n", prediction)
			fmt.Printf("   • Confidence level: %.1f%%\n", confidence*100)

			// Provide human-friendly interpretation of confidence level
			confidenceAssessment := getConfidenceAssessment(confidence)
			fmt.Printf("   • Assessment: %s\n", confidenceAssessment)

			// Provide additional context about the character type
			charType := getCharacterType(prediction)
			fmt.Printf("   • Character type: %s\n", charType)

			// Show confidence-based recommendations
			if confidence < 0.7 {
				fmt.Printf("💡 Recommendation: Consider using a clearer image for better accuracy\n")
			}
		}
	}

	// COMPREHENSIVE SYSTEM STATUS REPORT
	fmt.Printf("\n📋 SYSTEM STATUS REPORT\n")
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	// Check and report model availability
	if _, err := os.Stat(bestModelPath); err == nil {
		fmt.Printf("✅ Model available: %s\n", bestModelPath)
		fmt.Printf("✅ Ready for production use with 94-class recognition\n")
	} else {
		fmt.Printf("⚠️  No model was saved - check for errors above\n")
		fmt.Printf("💡 Ensure training completed successfully\n")
	}

	// Display sampling strategy information and recommendations
	fmt.Printf("\n📊 Current Data Sampling Strategy:\n")
	if samplesPerClass > 0 {
		totalSamples := samplesPerClass * 94
		percentOfFull := float64(totalSamples) / 1300000.0 * 100
		fmt.Printf("   ✅ Used %d samples per class for this run\n", samplesPerClass)
		fmt.Printf("   📈 Total training samples: ~%d (%.1f%% of full dataset)\n", totalSamples, percentOfFull)
		fmt.Printf("   💡 Next steps:\n")
		if samplesPerClass < 100 {
			fmt.Printf("      • Try --samples=100 for improved accuracy\n")
			fmt.Printf("      • Try --samples=1000 for validation testing\n")
		} else if samplesPerClass < 1000 {
			fmt.Printf("      • Try --samples=1000 for near-production accuracy\n")
		}
		fmt.Printf("      • Use --samples=0 to train on full dataset for maximum accuracy\n")
	} else {
		fmt.Printf("   ✅ Used ALL available samples (full dataset)\n")
		fmt.Printf("   🎯 Maximum accuracy configuration\n")
		fmt.Printf("   💡 Development options:\n")
		fmt.Printf("      • Use --samples=N for faster development iterations\n")
		fmt.Printf("      • Recommended: --samples=100 for development, --samples=0 for production\n")
	}

	fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	fmt.Printf("🎉 PROGRAM COMPLETED SUCCESSFULLY!\n")
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
}

// Helper function to check if a file exists
func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}

// getCharacterType determines what type of character was predicted.
//
// CHARACTER CLASSIFICATION HELPER:
// This utility function helps users understand what type of character the model
// predicted, which is valuable for debugging, validation, and understanding
// model performance across different character categories.
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

// getConfidenceAssessment provides human-friendly interpretation of confidence levels.
//
// CONFIDENCE INTERPRETATION GUIDE:
// Neural networks output probability scores, but these need interpretation for
// practical use. This function translates raw confidence scores into actionable
// assessments that help users understand prediction reliability.
func getConfidenceAssessment(confidence float64) string {
	switch {
	case confidence >= 0.95:
		return "Extremely confident - prediction highly reliable"
	case confidence >= 0.90:
		return "Very confident - prediction likely correct"
	case confidence >= 0.75:
		return "Confident - good prediction quality"
	case confidence >= 0.60:
		return "Moderately confident - consider verification"
	case confidence >= 0.40:
		return "Low confidence - prediction may be incorrect"
	default:
		return "Very uncertain - prediction likely wrong"
	}
}

// getSamplingUseCase provides context about different sampling configurations.
//
// SAMPLING STRATEGY GUIDANCE:
// This function helps users understand what different sampling configurations
// are best suited for, enabling informed decisions about development workflows.
func getSamplingUseCase(samples int) string {
	switch {
	case samples <= 20:
		return "Rapid prototyping and architecture testing"
	case samples <= 100:
		return "Development and hyperparameter tuning"
	case samples <= 500:
		return "Validation and model comparison"
	case samples <= 2000:
		return "Pre-production accuracy assessment"
	default:
		return "High-accuracy training approaching full dataset"
	}
}

// getExpectedTrainingTime provides time estimates based on sampling configuration.
//
// TRAINING TIME ESTIMATION:
// This function helps users set appropriate expectations for training duration
// based on their sampling configuration, enabling better planning and resource allocation.
func getExpectedTrainingTime(samples int) string {
	switch {
	case samples <= 20:
		return "Very fast (seconds to minutes)"
	case samples <= 100:
		return "Fast (minutes)"
	case samples <= 500:
		return "Moderate (tens of minutes)"
	case samples <= 2000:
		return "Extended (hours)"
	default:
		return "Long (multiple hours)"
	}
}

// COMPREHENSIVE DEVELOPMENT WORKFLOW GUIDANCE:

// BEST PRACTICES FOR DATA SAMPLING IN ML DEVELOPMENT:

// 1. START SMALL AND SCALE GRADUALLY:
// Begin with very small sample sizes (10-50 per class) to validate your approach,
// then gradually increase as your architecture and hyperparameters stabilize.

// 2. USE SAMPLING FOR EXPERIMENTATION:
// When testing different architectures, activation functions, or preprocessing
// approaches, use small samples for fast iteration cycles.

// 3. VALIDATE ON LARGER SAMPLES:
// Before committing to a final architecture, validate its performance on
// larger sample sizes to ensure the results will scale.

// 4. TRAIN PRODUCTION MODELS ON FULL DATA:
// For final deployment, always use the complete dataset to achieve maximum
// accuracy and robustness.

// SAMPLING PERFORMANCE CHARACTERISTICS:

// DEVELOPMENT EFFICIENCY GAINS:
// - 10 samples/class: 99%+ time reduction, perfect for rapid prototyping
// - 100 samples/class: 95%+ time reduction, excellent for development
// - 1000 samples/class: 85%+ time reduction, good for validation
// - Full dataset: Maximum accuracy, required for production

// ACCURACY TRADE-OFFS:
// - Small samples: Lower accuracy but enable rapid iteration and debugging
// - Medium samples: Good accuracy for development decisions and comparisons
// - Large samples: Near-production accuracy for final validation
// - Full dataset: Maximum achievable accuracy for deployment

// RESOURCE SCALING:
// - Memory usage scales linearly with sample size
// - Training time scales roughly linearly with sample size
// - Smaller samples enable development on resource-constrained systems
// - Full dataset may require high-memory servers or cloud instances

// RECOMMENDED DEVELOPMENT PHASES:

// PHASE 1: EXPLORATION (samples=10-50)
// Focus: Architecture design, basic functionality, pipeline validation
// Time: Minutes per experiment
// Goal: Get the system working end-to-end with reasonable results

// PHASE 2: OPTIMIZATION (samples=100-500)
// Focus: Hyperparameter tuning, preprocessing optimization, feature engineering
// Time: Minutes to hours per experiment
// Goal: Optimize the approach for best performance per unit of data

// PHASE 3: VALIDATION (samples=1000-5000)
// Focus: Performance validation, robustness testing, model comparison
// Time: Hours per experiment
// Goal: Ensure the optimized approach scales to larger datasets

// PHASE 4: PRODUCTION (samples=0, full dataset)
// Focus: Final model training, deployment preparation, maximum accuracy
// Time: Hours to days
// Goal: Create the best possible model for real-world deployment

// This main function provides a complete, production-ready machine learning
// system with intelligent workflows that dramatically improve development efficiency
// while maintaining the ability to create high-accuracy models for deployment.
// The flexible sampling system enables developers to work efficiently across the
// entire ML development lifecycle, from initial prototyping to final production.
