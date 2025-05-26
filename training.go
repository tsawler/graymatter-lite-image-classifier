package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// TrainWithValidation handles the complete training process using the module's capabilities.
//
// WHAT IS NEURAL NETWORK TRAINING?
// Training is the process where a neural network learns from examples. Imagine teaching
// a child to recognize letters by showing them thousands of letter images with the
// correct answers. The network starts knowing nothing (random weights) and gradually
// adjusts its internal parameters to make better predictions.
//
// THE LEARNING PROCESS:
// 1. Show the network a batch of training images
// 2. Let it make predictions (probably wrong at first)
// 3. Measure how wrong the predictions are (calculate loss/cost)
// 4. Use backpropagation to figure out how to adjust weights to reduce errors
// 5. Make small adjustments to the weights
// 6. Repeat thousands of times until the network learns the patterns
//
// WHY "WITH VALIDATION"?
// We split our data into two parts:
// - Training data: Used to teach the network (adjust weights)
// - Validation data: Used to test how well the network generalizes to new examples
// This helps detect "overfitting" where the network memorizes training data but
// can't handle new examples it hasn't seen before.
func (ic *ImageClassifier) TrainWithValidation() error {
	// STEP 1: Load all training data from disk
	fmt.Println("Loading training data...")
	trainingData, err := ic.loadTrainingData()
	if err != nil {
		return fmt.Errorf("failed to load training data: %w", err)
	}

	fmt.Printf("Loaded %d training samples\n", len(trainingData))
	
	// DATA SUFFICIENCY CHECK:
	// Machine learning requires substantial amounts of data. A few dozen examples
	// per class is usually insufficient. Hundreds or thousands per class is better.
	// For character recognition, you typically want at least 100-500 examples
	// per character for decent performance.

	// STEP 2: Convert data to neural network format
	// Neural networks work with matrices of numbers, not our ImageData structs
	inputs, outputs, err := ic.prepareDataForTraining(trainingData)
	if err != nil {
		return fmt.Errorf("failed to prepare training data: %w", err)
	}

	// STEP 3: Create a DataSet object that the neural network library understands
	// This wraps our inputs and outputs in a format that includes validation
	// and utility methods needed during training
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %w", err)
	}

	// STEP 4: Split data into training and validation sets
	// This is crucial for detecting overfitting and selecting the best model
	trainData, validData, err := graymatter.SplitDataSet(dataset, graymatter.SplitOptions{
		TrainRatio: 0.8,   // 80% for training, 20% for validation
		Shuffle:    true,  // Randomize order to avoid bias from data organization
	})
	if err != nil {
		return fmt.Errorf("failed to split dataset: %w", err)
	}

	// UNDERSTANDING THE SPLIT:
	// - Training set (80%): The network learns from these examples
	// - Validation set (20%): We test the network on these to measure generalization
	// The validation examples are never used to adjust weights - they're like
	// a "final exam" that the network hasn't studied for.
	
	fmt.Printf("Training samples: %d\n", ic.getDatasetSize(trainData))
	fmt.Printf("Validation samples: %d\n", ic.getDatasetSize(validData))

	// STEP 5: Create the neural network architecture
	fmt.Println("Creating neural network...")
	if err := ic.createNetwork(); err != nil {
		return fmt.Errorf("failed to create network: %w", err)
	}

	// STEP 6: Train the network using the graymatter library
	// This is where the actual machine learning happens!
	fmt.Println("Starting training...")
	finalCost, err := ic.network.TrainWithValidation(trainData, validData, ic.config.TrainingOptions)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// INTERPRETING FINAL COST:
	// - Lower cost = better performance
	// - Cost near 0 = almost perfect predictions
	// - High cost = network is making many mistakes
	// - Cost should generally decrease during training
	fmt.Printf("Training completed! Final cost: %.6f\n", finalCost)

	// STEP 7: Generate comprehensive analysis plots
	// Visualization is crucial for understanding what happened during training
	if ic.config.TrainingOptions.EnablePlotting {
		fmt.Println("\nGenerating comprehensive analysis...")
		if err := ic.generateAnalysisPlots(trainData, validData); err != nil {
			fmt.Printf("Warning: Failed to generate analysis plots: %v\n", err)
		}
	}

	return nil
}

// createNetwork initializes the neural network using module's NetworkConfig.
//
// NETWORK ARCHITECTURE DESIGN:
// The architecture defines how information flows through the network. Our design:
// 784 inputs → 128 hidden → 128 hidden → 62 outputs
//
// INPUT LAYER (784 neurons):
// One neuron for each pixel in a 28×28 image. Each neuron receives the
// brightness value of one pixel (0.0 = black, 1.0 = white).
//
// HIDDEN LAYERS (128 neurons each):
// These layers learn to detect patterns and features in the images.
// First hidden layer might learn to detect edges and curves.
// Second hidden layer might learn to combine edges into letter parts.
// More layers = more capacity to learn complex patterns, but also more
// risk of overfitting and slower training.
//
// OUTPUT LAYER (62 neurons):
// One neuron for each possible character (A-Z, a-z, 0-9).
// Uses softmax activation so outputs form a probability distribution
// that sums to 1.0. The neuron with the highest value is the prediction.
//
// ACTIVATION FUNCTIONS:
// - Hidden layers use ReLU: Simple, fast, avoids vanishing gradient problems
// - Output layer uses softmax: Converts raw scores to probabilities
//
// COST FUNCTION:
// Categorical cross-entropy measures how far the predicted probabilities
// are from the true labels. Works perfectly with softmax outputs.
func (ic *ImageClassifier) createNetwork() error {
	// Define the network configuration
	config := graymatter.NetworkConfig{
		// ARCHITECTURE: Define layer sizes from input to output
		LayerSizes: []int{
			ic.config.InputSize,  // 784 (28×28 pixels)
			ic.config.HiddenSize, // 128 (first hidden layer)
			ic.config.HiddenSize, // 128 (second hidden layer)  
			ic.config.OutputSize, // 62 (one per character class)
		},
		
		// ACTIVATION FUNCTIONS:
		// Choose functions that work well together and suit our problem type
		HiddenActivationFunction: "relu",     // Fast, simple, effective for hidden layers
		OutputActivationFunction: "softmax",  // Perfect for multi-class classification
		
		// COST FUNCTION:
		// Must match the output activation and problem type
		CostFunction: "categorical", // Categorical cross-entropy for multi-class
		
		// REPRODUCIBILITY:
		// Fixed seed ensures consistent results for debugging and comparison
		Seed: 42, // Any fixed number works; 42 is a popular choice
	}

	// ADD THIS DEBUG OUTPUT
    fmt.Printf("Network config: %+v\n", config)

	// Create the network using the graymatter library
	network, err := graymatter.NewNetwork(config)
	if err != nil {
		return err
	}

	// Store the network in our classifier for later use
	ic.network = network

	// ADD THIS TOO
    fmt.Printf("Network created with layers: %v\n", network.LayerSizes)
    fmt.Printf("Hidden activation: %d, Output activation: %d, Cost function: %d\n", 
        network.HiddenActivationFunctionType, 
        network.OutputActivationFunctionType, 
        network.CostFunctionType)

		
	return nil
}

// generateAnalysisPlots uses the module's plotting utilities for comprehensive analysis.
//
// WHY VISUALIZE TRAINING?
// Machine learning can be opaque - it's hard to know what's happening inside
// the network during training. Plots provide crucial insights:
// - Are we learning? (loss should decrease)
// - Are we overfitting? (validation loss increases while training loss decreases)
// - Is our architecture appropriate? (weight distributions should be healthy)
// - Which classes are problematic? (confusion matrix shows misclassifications)
//
// PLOTS WE GENERATE:
// 1. Network architecture diagram: Shows the structure visually
// 2. Weight distributions: Reveals training health issues
// 3. Confusion matrix: Shows which characters are confused with each other
func (ic *ImageClassifier) generateAnalysisPlots(trainData, validData *graymatter.DataSet) error {
	// Create a client to communicate with the plotting service
	plotClient := graymatter.NewPlottingClient(ic.config.PlottingURL)

	// STEP 1: Verify plotting service is available
	// Training can take hours - we want to catch plotting issues early
	healthResp, err := plotClient.CheckHealth()
	if err != nil {
		return fmt.Errorf("plotting service unavailable: %w", err)
	}

	fmt.Printf("Plotting service health: %s (version: %s)\n", healthResp.Status, healthResp.Version)

	// STEP 2: Generate network architecture visualization
	fmt.Println("Generating network architecture diagram...")
	archReq := graymatter.CreateNetworkArchitectureVisualization(
		ic.network,
		"Alphanumeric Character Recognition Network Architecture",
		"network_architecture.png",
	)

	archResp, err := plotClient.PlotNetworkArchitecture(archReq)
	if err != nil {
		fmt.Printf("Warning: Failed to generate architecture plot: %v\n", err)
	} else if archResp.Success {
		fmt.Printf("Architecture diagram saved: %s\n", archResp.FilePath)
	}

	// STEP 3: Generate weight distribution analysis
	// Weight distributions reveal important information about training health
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
		fmt.Printf("Weight distribution plo: %s\n", weightResp.FilePath)
	}

	// STEP 4: Generate confusion matrix for validation set
	// This shows which characters the network confuses with each other
	fmt.Println("Generating confusion matrix...")
	if err := ic.generateConfusionMatrix(plotClient, validData); err != nil {
		fmt.Printf("Warning: Failed to generate confusion matrix: %v\n", err)
	}

	return nil
}

// generateConfusionMatrix creates a confusion matrix plot for the validation set.
//
// WHAT IS A CONFUSION MATRIX?
// A confusion matrix is a table that shows how often each true class is
// predicted as each possible class. It's called "confusion" because it
// reveals which classes the model confuses with each other.
//
// EXAMPLE 3×3 CONFUSION MATRIX:
//           Predicted
//         A   B   C
// True A [90   5   5]  <- 90% correct, 5% confused with B, 5% with C
//      B [ 3  85  12]  <- 3% confused with A, 85% correct, 12% with C
//      C [ 2   8  90]  <- 2% confused with A, 8% with B, 90% correct
//
// DIAGONAL = CORRECT PREDICTIONS:
// Perfect predictions would show high numbers on the diagonal and zeros elsewhere.
// Off-diagonal elements show misclassifications.
//
// WHY THIS MATTERS:
// - Identifies systematic problems (is 'O' always confused with '0'?)
// - Reveals which classes need more training data
// - Helps understand model limitations
// - Guides data collection and model improvement efforts
func (ic *ImageClassifier) generateConfusionMatrix(plotClient *graymatter.PlottingClient, validData *graymatter.DataSet) error {
	// STEP 1: Get predictions from the trained network
	// Run all validation examples through the network to see what it predicts
	predictions, err := ic.network.Predict(validData.Inputs)
	if err != nil {
		return fmt.Errorf("failed to get predictions: %w", err)
	}

	// STEP 2: Generate confusion matrix using the library's utility
	// This compares predictions against true labels and counts mismatches
	matrix, classLabels, err := graymatter.GenerateConfusionMatrix(predictions, validData.Outputs, 0.5)
	if err != nil {
		return fmt.Errorf("failed to generate confusion matrix: %w", err)
	}

	// STEP 3: Limit matrix size for readability
	// With 62 classes, a full confusion matrix would be quite large.
	// For visualization purposes, we show a subset that's easier to interpret.
	if len(classLabels) > 15 {
		fmt.Println("Note: Limiting confusion matrix to first 15 classes for readability")
		truncatedMatrix := make([][]int, 15)
		truncatedLabels := make([]string, 15)
		for i := 0; i < 15; i++ {
			truncatedMatrix[i] = matrix[i][:15]
			truncatedLabels[i] = classLabels[i]
		}
		matrix = truncatedMatrix
		classLabels = truncatedLabels
	}

	// STEP 4: Create the plotting request
	confusionReq := &graymatter.ConfusionMatrixRequest{
		PlotType:    "confusion_matrix",
		Title:       "Validation Set Confusion Matrix (Sample)",
		Filename:    "confusion_matrix.png",
		Matrix:      matrix,
		ClassLabels: classLabels,
		Width:       800,
		Height:      800,
		Colormap:    "Blues",    // Color scheme: darker blue = more confusions
		ShowValues:  true,       // Display numbers in each cell
		Normalize:   false,      // Show raw counts, not percentages
	}

	// STEP 5: Send request to plotting service
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

// getDatasetSize returns the number of samples in a dataset.
//
// UTILITY FUNCTION:
// Simple helper to get dataset size for progress reporting.
// The graymatter library stores data as matrices, so we need to
// check the number of rows to get the sample count.
func (ic *ImageClassifier) getDatasetSize(dataset *graymatter.DataSet) int {
	rows, _ := dataset.Inputs.Dims()
	return rows
}

// TRAINING BEST PRACTICES DEMONSTRATED:

// 1. COMPREHENSIVE ERROR HANDLING:
// Every step can fail, and we provide meaningful error messages to help
// debug issues. We wrap errors with context about what operation failed.

// 2. PROGRESS REPORTING:
// Long-running operations provide regular updates so users know the
// system is working and can estimate completion time.

// 3. GRACEFUL DEGRADATION:
// If plotting fails, we warn the user but don't stop training.
// The core functionality (training the model) is more important than
// auxiliary features (generating plots).

// 4. SEPARATION OF CONCERNS:
// - TrainWithValidation: Orchestrates the overall process
// - createNetwork: Handles architecture setup
// - generateAnalysisPlots: Manages visualization
// Each function has a single, clear responsibility.

// 5. CONFIGURATION-DRIVEN BEHAVIOR:
// All settings come from the configuration struct, making it easy to
// experiment with different parameters without changing code.

// COMMON TRAINING PROBLEMS AND SOLUTIONS:

// 1. OVERFITTING:
// Problem: High training accuracy but low validation accuracy
// Solutions: More training data, simpler model, regularization, early stopping

// 2. UNDERFITTING:
// Problem: Both training and validation accuracy are low
// Solutions: More complex model, more training iterations, better features

// 3. VANISHING GRADIENTS:
// Problem: Network stops learning (weights don't change)
// Solutions: Better weight initialization, different activation functions (ReLU)

// 4. EXPLODING GRADIENTS:
// Problem: Network becomes unstable (weights grow extremely large)
// Solutions: Lower learning rate, gradient clipping, better normalization

// 5. CLASS IMBALANCE:
// Problem: Some classes have much more training data than others
// Solutions: Balanced sampling, weighted loss functions, data augmentation

// ALPHANUMERIC-SPECIFIC CONSIDERATIONS:

// 1. SIMILAR CHARACTERS:
// Some characters look very similar and may be confused:
// - 'O' (letter) vs '0' (zero)
// - 'l' (lowercase L) vs '1' (one) vs 'I' (uppercase i)
// - 'S' vs '5' in certain fonts

// 2. CASE SENSITIVITY:
// The model learns both uppercase and lowercase versions of letters.
// Make sure training data includes good examples of both cases.

// 3. FONT VARIATIONS:
// Different fonts can make characters look quite different.
// Include variety in training data for better generalization.

// This training pipeline provides a solid foundation for alphanumeric character
// recognition and demonstrates best practices for neural network training.
// The 62-class architecture (A-Z, a-z, 0-9) is well-suited for most text
// recognition applications.