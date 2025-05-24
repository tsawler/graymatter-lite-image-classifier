package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// SaveModel saves the trained model with metadata.
//
// WHY SAVE MODELS?
// Training neural networks can take minutes, hours, or even days depending on
// the dataset size and model complexity. Once you've invested that time and
// computational resources, you want to preserve the results! Saving models enables:
//
// 1. PRODUCTION DEPLOYMENT: Use the trained model in applications
// 2. EXPERIMENTATION: Compare different model versions and hyperparameters
// 3. COLLABORATION: Share models with team members
// 4. CHECKPOINT RECOVERY: Resume training if interrupted
// 5. VERSION CONTROL: Track model evolution over time
//
// WHAT GETS SAVED?
// - Network architecture (layer sizes, activation functions)
// - Learned weights and biases (the "knowledge" acquired during training)
// - Metadata (training accuracy, hyperparameters, notes for future reference)
//
// WHAT DOESN'T GET SAVED?
// - Training data (usually too large and often confidential)
// - Training history (loss curves, intermediate checkpoints)
// - Temporary training state (gradients, momentum, optimizer state)
func (ic *ImageClassifier) SaveModel(filename string, description string) error {
	// Validate that we have a trained model to save
	if ic.network == nil {
		return fmt.Errorf("no trained network to save")
	}

	// METADATA CREATION:
	// We package important information about the model and training process.
	// This metadata is invaluable for:
	// - Remembering which models worked best
	// - Reproducing successful experiments
	// - Understanding model performance characteristics
	// - Debugging prediction issues
	metadata := graymatter.NetworkMetadata{
		// Human-readable description of what this model does
		Description: description,
		
		// HYPERPARAMETERS USED DURING TRAINING:
		// These are critical for reproducing results or understanding
		// why a particular model succeeded or failed
		LearningRate: ic.config.TrainingOptions.LearningRate,
		BatchSize:    ic.config.TrainingOptions.BatchSize,
		Epochs:       ic.config.TrainingOptions.Iterations,
		
		// MODEL CHARACTERISTICS:
		// Information that helps users understand the model's capabilities
		Notes: fmt.Sprintf("Image classifier with %d classes", ic.config.OutputSize),
	}

	// SAVE OPERATION:
	// The graymatter library handles the low-level details of serialization,
	// file I/O, and format conversion. We just provide the filename and metadata.
	return ic.network.Save(filename, metadata)
}

// LoadModelForInference loads a model specifically for making predictions.
//
// INFERENCE vs TRAINING:
// When loading a model for inference (making predictions), we don't need
// all the training infrastructure. We create a minimal configuration that
// includes only what's necessary for prediction:
// - Network architecture must match the saved model
// - Input/output dimensions must be correct
// - We don't need training options, plotting settings, etc.
//
// USE CASES:
// - Loading a model in a production web service
// - Batch processing of images for classification
// - Interactive applications where users upload images
// - A/B testing different model versions
func LoadModelForInference(filename string) (*ImageClassifier, *graymatter.NetworkMetadata, error) {
	// STEP 1: Load the saved network and its metadata
	network, metadata, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load network: %w", err)
	}

	// STEP 2: Create minimal configuration for inference
	// We hard-code the image dimensions and class count because they're
	// fundamental properties of our character recognition system.
	// In a more flexible system, these might be saved with the model.
	config := &Config{
		ImageWidth:  28,  // All images must be 28×28 pixels
		ImageHeight: 28,
		InputSize:   784, // 28 × 28 = 784 pixels
		OutputSize:  62,  // 26 + 26 + 10 = 62 character classes
	}

	// STEP 3: Create classifier instance with loaded network
	classifier := &ImageClassifier{
		config:  config,
		network: network,
	}

	// Return both the classifier and metadata
	// Metadata contains useful information like training accuracy,
	// hyperparameters used, and notes about the model
	return classifier, metadata, nil
}

// EvaluateModel evaluates the model on a test dataset.
//
// EVALUATION vs VALIDATION:
// - Validation: Used during training to tune hyperparameters and detect overfitting
// - Evaluation: Final assessment on completely unseen test data
//
// The test dataset should be kept separate from both training and validation data.
// It provides an unbiased estimate of how the model will perform in the real world.
//
// EVALUATION METRICS:
// We calculate accuracy (percentage of correct predictions), but other metrics
// might be relevant depending on your use case:
// - Precision: Of all positive predictions, how many were actually correct?
// - Recall: Of all actual positives, how many did we correctly identify?
// - F1-score: Balanced combination of precision and recall
// - Per-class accuracy: Performance on each individual character
func (ic *ImageClassifier) EvaluateModel(testDataPath string) (float64, error) {
	// STEP 1: Temporarily switch data directory to load test data
	// We need to load test data the same way we loaded training data,
	// but from a different directory. We temporarily change the config.
	originalDataDir := ic.config.DataDir
	ic.config.DataDir = testDataPath
	
	// STEP 2: Load test data using existing data loading infrastructure
	// This ensures test data gets the same preprocessing as training data
	testData, err := ic.loadTrainingData()
	if err != nil {
		ic.config.DataDir = originalDataDir // Restore original setting
		return 0, fmt.Errorf("failed to load test data: %w", err)
	}
	
	// STEP 3: Restore original data directory setting
	ic.config.DataDir = originalDataDir

	// STEP 4: Convert test data to neural network format
	inputs, outputs, err := ic.prepareDataForTraining(testData)
	if err != nil {
		return 0, fmt.Errorf("failed to prepare test data: %w", err)
	}

	// STEP 5: Create dataset object for evaluation
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return 0, fmt.Errorf("failed to create test dataset: %w", err)
	}

	// STEP 6: Calculate accuracy using the library's method
	// This runs all test examples through the network and computes
	// the percentage of correct predictions
	accuracy, err := ic.network.CalculateAccuracy(dataset, 0.5)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate accuracy: %w", err)
	}

	return accuracy, nil
}

// MODEL MANAGEMENT BEST PRACTICES:

// 1. DESCRIPTIVE FILENAMES:
// Use names that include key information:
// - "character_classifier_acc_94.5_lr_0.001.json"
// - "handwriting_model_v2_epochs_500.json"
// - "production_model_2024_01_15.json"

// 2. VERSION CONTROL FOR MODELS:
// Consider using Git LFS or specialized model versioning tools like DVC
// to track model changes alongside code changes.

// 3. MODEL METADATA:
// Always include comprehensive metadata:
// - Training date and duration
// - Dataset characteristics (size, source, preprocessing)
// - Hyperparameters used
// - Performance metrics
// - Known limitations or issues

// 4. TESTING STRATEGY:
// - Unit tests: Test individual functions work correctly
// - Integration tests: Test entire pipeline end-to-end
// - Performance tests: Verify accuracy on known test sets
// - Regression tests: Ensure new changes don't break existing functionality

// 5. DEPLOYMENT CONSIDERATIONS:
// - Model size (affects loading time and memory usage)
// - Inference speed (predictions per second)
// - Resource requirements (CPU, memory, GPU)
// - Backward compatibility (can old clients use new models?)

// COMMON PITFALLS TO AVOID:

// 1. INCONSISTENT PREPROCESSING:
// The same preprocessing pipeline MUST be used for training, validation,
// testing, and production inference. Any difference will cause poor performance.

// 2. DATA LEAKAGE:
// Test data must be completely separate from training data. If any test
// examples were seen during training, evaluation results will be overly optimistic.

// 3. TEMPORAL ISSUES:
// If your data has a time component, split chronologically rather than randomly.
// Don't use future data to predict past events.

// 4. CLASS IMBALANCE IN EVALUATION:
// If your test set has unequal class representation, overall accuracy might
// be misleading. Consider per-class metrics and balanced test sets.

// This model utilities module provides the foundation for a complete machine
// learning lifecycle: train, save, load, and evaluate models in a consistent,
// reliable manner.