package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// Predict makes a prediction on a single image file using the module's functionality
func (ic *ImageClassifier) Predict(imagePath string) (string, float64, error) {
	// Load and process the image
	pixels, err := ic.loadAndProcessImage(imagePath)
	if err != nil {
		return "", 0, fmt.Errorf("failed to load image: %w", err)
	}

	// Convert to dataset format using module's constructor
	inputs := [][]float64{pixels}
	dataset, err := graymatter.NewDataSet(inputs, [][]float64{{0}}) // Dummy output
	if err != nil {
		return "", 0, fmt.Errorf("failed to create input dataset: %w", err)
	}

	// Use the module's Predict method directly
	predictions, err := ic.network.Predict(dataset.Inputs)
	if err != nil {
		return "", 0, fmt.Errorf("prediction failed: %w", err)
	}

	// Find the class with highest probability
	_, cols := predictions.Dims()
	maxProb := 0.0
	maxIndex := 0

	for j := 0; j < cols; j++ {
		prob := predictions.At(0, j)
		if prob > maxProb {
			maxProb = prob
			maxIndex = j
		}
	}

	// Convert index back to class name
	className, exists := IndexToClass[maxIndex]
	if !exists {
		return "", 0, fmt.Errorf("unknown class index: %d", maxIndex)
	}

	return className, maxProb, nil
}