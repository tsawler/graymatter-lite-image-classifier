package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// SaveModel saves the trained model with metadata
func (ic *ImageClassifier) SaveModel(filename string, description string) error {
	if ic.network == nil {
		return fmt.Errorf("no trained network to save")
	}

	// Calculate final metrics for metadata
	metadata := graymatter.NetworkMetadata{
		Description:  description,
		LearningRate: ic.config.TrainingOptions.LearningRate,
		BatchSize:    ic.config.TrainingOptions.BatchSize,
		Epochs:       ic.config.TrainingOptions.Iterations,
		Notes:        fmt.Sprintf("Image classifier with %d classes", ic.config.OutputSize),
	}

	return ic.network.Save(filename, metadata)
}

// LoadModelForInference loads a model specifically for making predictions
func LoadModelForInference(filename string) (*ImageClassifier, *graymatter.NetworkMetadata, error) {
	network, metadata, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load network: %w", err)
	}

	// Create a minimal config for inference
	config := &Config{
		ImageWidth:  28,
		ImageHeight: 28,
		InputSize:   784,
		OutputSize:  62,
	}

	classifier := &ImageClassifier{
		config:  config,
		network: network,
	}

	return classifier, metadata, nil
}

// EvaluateModel evaluates the model on a test dataset
func (ic *ImageClassifier) EvaluateModel(testDataPath string) (float64, error) {
	// Load test data (reuse existing loading logic)
	originalDataDir := ic.config.DataDir
	ic.config.DataDir = testDataPath
	
	testData, err := ic.loadTrainingData()
	if err != nil {
		ic.config.DataDir = originalDataDir
		return 0, fmt.Errorf("failed to load test data: %w", err)
	}
	
	ic.config.DataDir = originalDataDir

	// Convert to neural network format
	inputs, outputs, err := ic.prepareDataForTraining(testData)
	if err != nil {
		return 0, fmt.Errorf("failed to prepare test data: %w", err)
	}

	// Create dataset
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return 0, fmt.Errorf("failed to create test dataset: %w", err)
	}

	// Calculate accuracy using the module's method
	accuracy, err := ic.network.CalculateAccuracy(dataset, 0.5)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate accuracy: %w", err)
	}

	return accuracy, nil
}