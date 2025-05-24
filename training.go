package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// TrainWithValidation handles the complete training process using the module's capabilities
func (ic *ImageClassifier) TrainWithValidation() error {
	// Load training data
	fmt.Println("Loading training data...")
	trainingData, err := ic.loadTrainingData()
	if err != nil {
		return fmt.Errorf("failed to load training data: %w", err)
	}

	fmt.Printf("Loaded %d training samples\n", len(trainingData))

	// Convert to neural network format
	inputs, outputs, err := ic.prepareDataForTraining(trainingData)
	if err != nil {
		return fmt.Errorf("failed to prepare training data: %w", err)
	}

	// Create dataset using the module's constructor
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %w", err)
	}

	// Split into training and validation sets using module's function
	trainData, validData, err := graymatter.SplitDataSet(dataset, graymatter.SplitOptions{
		TrainRatio: 0.8,
		Shuffle:    true,
	})
	if err != nil {
		return fmt.Errorf("failed to split dataset: %w", err)
	}

	fmt.Printf("Training samples: %d\n", ic.getDatasetSize(trainData))
	fmt.Printf("Validation samples: %d\n", ic.getDatasetSize(validData))

	// Create neural network using module's constructor
	fmt.Println("Creating neural network...")
	if err := ic.createNetwork(); err != nil {
		return fmt.Errorf("failed to create network: %w", err)
	}

	// Use the module's TrainWithValidation method directly
	fmt.Println("Starting training...")
	finalCost, err := ic.network.TrainWithValidation(trainData, validData, ic.config.TrainingOptions)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	fmt.Printf("Training completed! Final cost: %.6f\n", finalCost)

	// Generate comprehensive analysis using module's plotting utilities
	if ic.config.TrainingOptions.EnablePlotting {
		fmt.Println("\nGenerating comprehensive analysis...")
		if err := ic.generateAnalysisPlots(trainData, validData); err != nil {
			fmt.Printf("Warning: Failed to generate analysis plots: %v\n", err)
		}
	}

	return nil
}

// createNetwork initializes the neural network using module's NetworkConfig
func (ic *ImageClassifier) createNetwork() error {
	config := graymatter.NetworkConfig{
		LayerSizes:               []int{ic.config.InputSize, ic.config.HiddenSize, ic.config.HiddenSize, ic.config.OutputSize},
		HiddenActivationFunction: "relu",
		OutputActivationFunction: "softmax",
		CostFunction:             "categorical",
		Seed:                     42,
	}

	network, err := graymatter.NewNetwork(config)
	if err != nil {
		return err
	}

	ic.network = network
	return nil
}

// generateAnalysisPlots uses the module's plotting utilities for comprehensive analysis
func (ic *ImageClassifier) generateAnalysisPlots(trainData, validData *graymatter.DataSet) error {
	plotClient := graymatter.NewPlottingClient(ic.config.PlottingURL)

	// Check if plotting service is available
	healthResp, err := plotClient.CheckHealth()
	if err != nil {
		return fmt.Errorf("plotting service unavailable: %w", err)
	}

	fmt.Printf("Plotting service health: %s (version: %s)\n", healthResp.Status, healthResp.Version)

	// Generate network architecture visualization
	fmt.Println("Generating network architecture diagram...")
	archReq := graymatter.CreateNetworkArchitectureVisualization(
		ic.network,
		"Image Classification Network Architecture",
		"network_architecture.png",
	)

	archResp, err := plotClient.PlotNetworkArchitecture(archReq)
	if err != nil {
		fmt.Printf("Warning: Failed to generate architecture plot: %v\n", err)
	} else if archResp.Success {
		fmt.Printf("Architecture diagram saved: %s\n", archResp.FilePath)
	}

	// Generate weight distribution analysis
	fmt.Println("Generating weight distribution analysis...")
	weightStats := graymatter.ExtractWeightStatistics(ic.network)

	weightReq := &graymatter.WeightDistributionRequest{
		PlotType:   "weight_distribution",
		Title:      "Weight Distributions by Layer",
		Filename:   "weight_distributions.png",
		WeightData: weightStats,
		Width:      1200,
		Height:     800,
		ShowStats:  true,
	}

	weightResp, err := plotClient.PlotWeightDistribution(weightReq)
	if err != nil {
		fmt.Printf("Warning: Failed to generate weight distribution plot: %v\n", err)
	} else if weightResp.Success {
		fmt.Printf("Weight distribution plot saved: %s\n", weightResp.FilePath)
	}

	// Generate confusion matrix for validation set
	fmt.Println("Generating confusion matrix...")
	if err := ic.generateConfusionMatrix(plotClient, validData); err != nil {
		fmt.Printf("Warning: Failed to generate confusion matrix: %v\n", err)
	}

	return nil
}

// generateConfusionMatrix creates a confusion matrix plot for the validation set
func (ic *ImageClassifier) generateConfusionMatrix(plotClient *graymatter.PlottingClient, validData *graymatter.DataSet) error {
	// Get predictions for validation set
	predictions, err := ic.network.Predict(validData.Inputs)
	if err != nil {
		return fmt.Errorf("failed to get predictions: %w", err)
	}

	// Generate confusion matrix using module's utility
	matrix, classLabels, err := graymatter.GenerateConfusionMatrix(predictions, validData.Outputs, 0.5)
	if err != nil {
		return fmt.Errorf("failed to generate confusion matrix: %w", err)
	}

	// Limit to a subset of classes for readability
	if len(classLabels) > 10 {
		fmt.Println("Note: Limiting confusion matrix to first 10 classes for readability")
		truncatedMatrix := make([][]int, 10)
		truncatedLabels := make([]string, 10)
		for i := 0; i < 10; i++ {
			truncatedMatrix[i] = matrix[i][:10]
			truncatedLabels[i] = classLabels[i]
		}
		matrix = truncatedMatrix
		classLabels = truncatedLabels
	}

	confusionReq := &graymatter.ConfusionMatrixRequest{
		PlotType:    "confusion_matrix",
		Title:       "Validation Set Confusion Matrix (Sample)",
		Filename:    "confusion_matrix.png",
		Matrix:      matrix,
		ClassLabels: classLabels,
		Width:       800,
		Height:      800,
		Colormap:    "Blues",
		ShowValues:  true,
		Normalize:   false,
	}

	confusionResp, err := plotClient.PlotConfusionMatrix(confusionReq)
	if err != nil {
		return fmt.Errorf("failed to plot confusion matrix: %w", err)
	}

	if confusionResp.Success {
		fmt.Printf("Confusion matrix saved: %s\n", confusionResp.FilePath)
	} else {
		return fmt.Errorf("confusion matrix plotting failed: %s", confusionResp.Error)
	}

	return nil
}

// getDatasetSize returns the number of samples in a dataset
func (ic *ImageClassifier) getDatasetSize(dataset *graymatter.DataSet) int {
	rows, _ := dataset.Inputs.Dims()
	return rows
}