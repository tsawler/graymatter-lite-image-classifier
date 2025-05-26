package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// SaveModel saves the trained model with metadata.
//
// WHY SAVE MODELS?
// Training neural networks can take anywhere from minutes to days depending on the
// dataset size and model complexity. Once you've invested that time and computational
// resources, you want to preserve the results! Saving models enables several important
// use cases:
//
// 1. PRODUCTION DEPLOYMENT: Use the trained model in applications without retraining
// 2. EXPERIMENTATION: Compare different model versions and hyperparameter settings
// 3. COLLABORATION: Share models with team members or the broader community
// 4. CHECKPOINT RECOVERY: Resume training if the process gets interrupted
// 5. VERSION CONTROL: Track model evolution and performance over time
//
// WHAT GETS SAVED?
// - Network architecture: Layer sizes, activation functions, connections
// - Learned weights and biases: The actual "knowledge" acquired during training
// - Training metadata: Accuracy metrics, hyperparameters, notes for future reference
//
// WHAT DOESN'T GET SAVED?
// - Training data: Usually too large and often confidential/proprietary
// - Training history: Loss curves and intermediate checkpoints (unless specifically saved)
// - Temporary training state: Gradients, momentum values, optimizer internal state
func (ic *ImageClassifier) SaveModel(filename string, description string) error {
	// Validate that we have a trained model to save
	if ic.network == nil {
		return fmt.Errorf("no trained network to save")
	}

	// METADATA CREATION:
	// We package important information about the model and training process.
	// This metadata is invaluable for understanding model performance and
	// reproducing successful experiments later.
	metadata := graymatter.NetworkMetadata{
		// Human-readable description of what this model does and its purpose
		Description: description,
		
		// HYPERPARAMETERS USED DURING TRAINING:
		// These settings are critical for reproducing results or understanding
		// why a particular model succeeded or failed
		LearningRate: ic.config.TrainingOptions.LearningRate,
		BatchSize:    ic.config.TrainingOptions.BatchSize,
		Epochs:       ic.config.TrainingOptions.Iterations,
		
		// MODEL CHARACTERISTICS:
		// Information that helps users understand the model's capabilities and limitations
		Notes: fmt.Sprintf("Character classifier with %d classes (A-Z, a-z, 0-9, punctuation)", ic.config.OutputSize),
	}

	// SAVE OPERATION:
	// The graymatter library handles the low-level details of serialization,
	// file I/O, and format conversion. We just provide the filename and metadata.
	return ic.network.Save(filename, metadata)
}

// LoadModelForInference loads a model specifically for making predictions.
//
// INFERENCE vs TRAINING:
// When loading a model for inference (making predictions), we don't need all the
// training infrastructure. We create a minimal configuration that includes only
// what's necessary for prediction:
// - Network architecture must match the saved model exactly
// - Input/output dimensions must be correct for the data format
// - We don't need training options, plotting settings, or other training-specific config
//
// COMMON USE CASES:
// - Loading a model in a production web service for real-time predictions
// - Batch processing of images for classification in data pipelines
// - Interactive applications where users upload images for character recognition
// - A/B testing different model versions to compare performance
//
// COMPATIBILITY NOTE:
// Always ensure the loaded model matches your expected character set. A model trained
// on the original 62 classes (A-Z, a-z, 0-9) won't work properly for 94-class recognition
// that includes punctuation, and vice versa. The input/output dimensions must match exactly.
func LoadModelForInference(filename string) (*ImageClassifier, *graymatter.NetworkMetadata, error) {
	// STEP 1: Load the saved network and its metadata from disk
	network, metadata, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load network: %w", err)
	}

	// STEP 2: Create minimal configuration for inference
	// We hard-code the image dimensions and class count because they're fundamental
	// properties of our character recognition system. In a more flexible system,
	// these values might be saved with the model metadata.
	config := &Config{
		ImageWidth:  28,  // All input images must be exactly 28×28 pixels
		ImageHeight: 28,
		InputSize:   784, // 28 × 28 = 784 pixels (flattened to 1D array)
		OutputSize:  94,  // 26 + 26 + 10 + 32 = 94 total character classes
	}

	// STEP 3: Create classifier instance with the loaded network
	classifier := &ImageClassifier{
		config:  config,
		network: network,
	}

	// Return both the classifier and metadata
	// The metadata contains useful information like training accuracy,
	// hyperparameters used, and notes about the model's capabilities
	return classifier, metadata, nil
}

// EvaluateModel evaluates the model on a test dataset.
//
// EVALUATION vs VALIDATION:
// It's important to understand the difference between these concepts:
// - Validation: Used during training to tune hyperparameters and detect overfitting
// - Evaluation: Final assessment on completely unseen test data
//
// The test dataset should be kept completely separate from both training and validation data.
// It provides an unbiased estimate of how the model will perform in the real world on
// data it has never seen before.
//
// EVALUATION METRICS:
// We calculate overall accuracy (percentage of correct predictions), but other metrics
// might be relevant depending on your specific use case:
// - Precision: Of all positive predictions for a class, how many were actually correct?
// - Recall: Of all actual instances of a class, how many did we correctly identify?
// - F1-score: Balanced combination of precision and recall
// - Per-class accuracy: Performance on each individual character type
func (ic *ImageClassifier) EvaluateModel(testDataPath string) (float64, error) {
	// STEP 1: Temporarily switch data directory to load test data
	// We need to load test data using the same pipeline as training data,
	// but from a different directory. We temporarily change the config to point
	// to the test data location.
	originalDataDir := ic.config.DataDir
	ic.config.DataDir = testDataPath
	
	// STEP 2: Load test data using existing data loading infrastructure
	// This ensures test data gets exactly the same preprocessing as training data,
	// which is critical for fair evaluation
	testData, err := ic.loadTrainingData()
	if err != nil {
		ic.config.DataDir = originalDataDir // Restore original setting
		return 0, fmt.Errorf("failed to load test data: %w", err)
	}
	
	// STEP 3: Restore original data directory setting
	ic.config.DataDir = originalDataDir

	// STEP 4: Convert test data to neural network format
	// Use the same conversion process as training to ensure consistency
	inputs, outputs, err := ic.prepareDataForTraining(testData)
	if err != nil {
		return 0, fmt.Errorf("failed to prepare test data: %w", err)
	}

	// STEP 5: Create dataset object for evaluation
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return 0, fmt.Errorf("failed to create test dataset: %w", err)
	}

	// STEP 6: Calculate accuracy using the library's evaluation method
	// This runs all test examples through the network and computes the percentage
	// of correct predictions across all 94 character classes
	accuracy, err := ic.network.CalculateAccuracy(dataset, 0.5)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate accuracy: %w", err)
	}

	return accuracy, nil
}

// MODEL MANAGEMENT BEST PRACTICES:

// 1. DESCRIPTIVE FILENAMES:
// Use names that include key information for easy identification:
// - "character_classifier_94class_acc_92.5_lr_0.001.json"
// - "model_with_punctuation_v3.json"
// - "production_model_2024_05_25.json"
// Include accuracy, learning rate, date, or version information in the filename.

// 2. VERSION CONTROL FOR MODELS:
// Consider using Git LFS (Large File Storage) or specialized model versioning
// tools like DVC (Data Version Control) to track model changes alongside code changes.
// This helps maintain reproducibility and enables rollbacks if needed.

// 3. COMPREHENSIVE MODEL METADATA:
// Always include detailed metadata with saved models:
// - Training date and duration
// - Dataset characteristics (size, source, preprocessing steps)
// - All hyperparameters used (learning rate, batch size, epochs, etc.)
// - Performance metrics (overall accuracy and per-class if available)
// - Known limitations, issues, or special considerations
// - Character set supported (94-class)

// 4. TESTING STRATEGY:
// Implement multiple levels of testing for robust model deployment:
// - Unit tests: Verify individual functions work correctly with expected inputs
// - Integration tests: Test the entire pipeline end-to-end with sample data
// - Performance tests: Verify accuracy meets requirements on known test sets
// - Regression tests: Ensure new changes don't break existing functionality
// - Character-type-specific tests: Verify punctuation recognition works properly

// 5. DEPLOYMENT CONSIDERATIONS:
// Plan for production deployment early in the development process:
// - Model file size (affects loading time and storage requirements)
// - Inference speed (predictions per second under expected load)
// - Resource requirements (CPU, memory, potentially GPU)
// - Backward compatibility (can old client code use new model versions?)
// - Character set compatibility (62-class vs 94-class models)

// COMMON PITFALLS TO AVOID:

// 1. INCONSISTENT PREPROCESSING:
// The exact same preprocessing pipeline MUST be used for training, validation,
// testing, and production inference. Even small differences (like different
// normalization ranges or image resize algorithms) will cause poor performance
// even with a perfectly trained model.

// 2. DATA LEAKAGE:
// Test data must be completely separate from training data. If any test examples
// were seen during training (even indirectly), evaluation results will be
// overly optimistic and won't reflect real-world performance.

// 3. TEMPORAL ISSUES:
// If your data has a time component (like documents from different time periods),
// split chronologically rather than randomly. Don't use future data to predict
// past events, as this creates unrealistic performance expectations.

// 4. CLASS IMBALANCE IN EVALUATION:
// If your test set has unequal representation of different character classes,
// overall accuracy might be misleading. Consider calculating per-class metrics
// and using balanced test sets. This is especially important with punctuation
// marks, which may be less common than letters in typical text.

// 5. MODEL VERSION MISMATCH:
// Ensure complete compatibility between saved models and inference code. A model
// trained on 62 classes won't work with 94-class prediction code, and vice versa.
// Always verify that input/output dimensions match expectations.

// 6. PUNCTUATION-SPECIFIC CHALLENGES:
// Punctuation marks present unique recognition challenges:
// - Visual similarity between some marks (period vs comma vs semicolon)
// - Size variations (punctuation is often smaller than letters)
// - Font dependencies (punctuation varies more dramatically across fonts)
// - Lower frequency in typical text (may need more training examples)

// 7. PERFORMANCE MONITORING:
// Implement detailed performance monitoring for production systems:
// - Monitor per-character-type accuracy (letters vs digits vs punctuation)
// - Identify which specific punctuation marks are most problematic
// - Track performance over different text sources and fonts
// - Consider separate metrics for different character categories

// 8. DATASET CONSIDERATIONS:
// When building datasets for 94-class recognition:
// - Ensure balanced representation across all 94 classes
// - Collect extra examples for visually similar characters
// - Include variety in punctuation mark sizes and styles
// - Test with multiple fonts to ensure generalization