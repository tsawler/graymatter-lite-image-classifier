package main

import (
	"fmt"

	"github.com/tsawler/graymatter-lite"
)

// TrainWithValidation handles the complete training process using the module's capabilities.
//
// WHAT IS NEURAL NETWORK TRAINING?
// Training is the process where a neural network learns from examples. Think of it like
// teaching someone to recognize handwriting by showing them thousands of letter samples
// with the correct answers. The network starts with random internal settings (called
// "weights") and gradually adjusts them to make better predictions.
//
// THE LEARNING PROCESS:
// 1. Show the network a batch of training images with their correct labels
// 2. Let it make predictions (which will be mostly wrong at first)
// 3. Calculate how wrong the predictions are (this error is called "loss" or "cost")
// 4. Use a mathematical technique called "backpropagation" to figure out how to
//    adjust the weights to reduce the errors
// 5. Make small adjustments to the weights in the right direction
// 6. Repeat this process thousands of times until the network learns the patterns
//
// WHY "WITH VALIDATION"?
// We split our data into two separate groups:
// - Training data: Used to teach the network (the network adjusts its weights based on this)
// - Validation data: Used to test how well the network performs on new, unseen examples
// 
// This split helps us detect "overfitting" - a problem where the network memorizes the
// training examples perfectly but fails on new data it hasn't seen before. It's like
// a student who memorizes textbook examples but can't solve similar problems on an exam.
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
	// per character class is usually insufficient for good performance. For character
	// recognition, you typically want at least 100-500 examples per character, though
	// more is better. With too little data, the network can't learn the full variety
	// of ways each character can be written.

	// STEP 2: Convert data to neural network format
	// Neural networks work with matrices of floating-point numbers, not our custom
	// ImageData structs. This step converts our organized data into the mathematical
	// format the network expects.
	inputs, outputs, err := ic.prepareDataForTraining(trainingData)
	if err != nil {
		return fmt.Errorf("failed to prepare training data: %w", err)
	}

	// STEP 3: Create a DataSet object that the neural network library understands
	// This wraps our inputs and outputs in a container that includes validation
	// methods and other utilities needed during the training process
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %w", err)
	}

	// STEP 4: Split data into training and validation sets
	// This is crucial for detecting overfitting and measuring how well the model
	// will perform on completely new data
	trainData, validData, err := graymatter.SplitDataSet(dataset, graymatter.SplitOptions{
		TrainRatio: 0.8,   // 80% for training, 20% for validation
		Shuffle:    true,  // Randomize order to avoid bias from data organization
	})
	if err != nil {
		return fmt.Errorf("failed to split dataset: %w", err)
	}

	// UNDERSTANDING THE SPLIT:
	// - Training set (80%): The network learns from these examples and adjusts its weights
	// - Validation set (20%): We test the network on these to measure generalization
	// 
	// The validation examples are never used to adjust weights - they're like a "final exam"
	// that the network hasn't studied for. This gives us an honest assessment of performance.
	
	fmt.Printf("Training samples: %d\n", ic.getDatasetSize(trainData))
	fmt.Printf("Validation samples: %d\n", ic.getDatasetSize(validData))

	// STEP 5: Create the neural network architecture
	fmt.Println("Creating neural network...")
	if err := ic.createNetwork(); err != nil {
		return fmt.Errorf("failed to create network: %w", err)
	}

	// STEP 6: Train the network using the graymatter library
	// This is where the actual machine learning happens! The network will repeatedly:
	// - Look at batches of training images
	// - Make predictions
	// - Calculate errors
	// - Adjust its internal weights to reduce those errors
	fmt.Println("Starting training...")
	finalCost, err := ic.network.TrainWithValidation(trainData, validData, ic.config.TrainingOptions)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// INTERPRETING FINAL COST:
	// "Cost" (also called "loss") measures how wrong the network's predictions are:
	// - Lower cost = better performance (fewer mistakes)
	// - Cost approaching 0 = nearly perfect predictions
	// - High cost = network is making many mistakes
	// - Cost should generally decrease during training as the network improves
	fmt.Printf("Training completed! Final cost: %.6f\n", finalCost)

	// STEP 7: Generate comprehensive analysis plots
	// Visualization is crucial for understanding what happened during training and
	// diagnosing potential problems. We create several types of plots to analyze
	// the trained network.
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
// The architecture defines how information flows through the network. Think of it as
// designing the structure of a decision-making system. Our design follows this pattern:
// 784 inputs → 128 hidden neurons → 128 hidden neurons → 94 outputs
//
// INPUT LAYER (784 neurons):
// One neuron for each pixel in a 28×28 image. Each neuron receives the brightness
// value of one pixel (0.0 = completely black, 1.0 = completely white). The network
// examines all 784 pixels simultaneously to recognize the character.
//
// HIDDEN LAYERS (128 neurons each):
// These intermediate layers learn to detect patterns and features in the images:
// - First hidden layer: Might learn to detect basic edges, curves, and lines
// - Second hidden layer: Might learn to combine these basic features into character parts
// 
// More hidden layers = more capacity to learn complex patterns, but also increased
// risk of overfitting and slower training. Two layers of 128 neurons each provides
// a good balance for character recognition.
//
// OUTPUT LAYER (94 neurons):
// One neuron for each possible character (A-Z, a-z, 0-9, plus 32 punctuation marks).
// Uses "softmax" activation so outputs form a probability distribution that sums to 1.0.
// The neuron with the highest value indicates the network's best guess.
//
// ACTIVATION FUNCTIONS:
// These are mathematical functions that determine how neurons respond to their inputs:
// - Hidden layers use ReLU (Rectified Linear Unit): Simple, fast, and avoids training problems
// - Output layer uses softmax: Converts raw scores to probabilities that sum to 100%
//
// COST FUNCTION:
// "Categorical cross-entropy" measures how far the predicted probabilities are from
// the true labels. It works perfectly with softmax outputs and multi-class problems.
func (ic *ImageClassifier) createNetwork() error {
	// Define the network configuration
	config := graymatter.NetworkConfig{
		// ARCHITECTURE: Define layer sizes from input to output
		LayerSizes: []int{
			ic.config.InputSize,  // 784 (28×28 pixels flattened to 1D)
			ic.config.HiddenSize, // 128 (first hidden layer)
			ic.config.HiddenSize, // 128 (second hidden layer)  
			ic.config.OutputSize, // 94 (one per character class)
		},
		
		// ACTIVATION FUNCTIONS:
		// Choose functions that work well together for this type of problem
		HiddenActivationFunction: "relu",     // Fast and effective for hidden layers
		OutputActivationFunction: "softmax",  // Perfect for multi-class classification
		
		// COST FUNCTION:
		// Must match the output activation and problem type
		CostFunction: "categorical", // Categorical cross-entropy for multi-class problems
		
		// REPRODUCIBILITY:
		// Fixed seed ensures consistent results across runs for debugging and comparison
		Seed: 42, // Any fixed number works; 42 is a popular choice in programming
	}

	// Debug output to verify configuration
	fmt.Printf("Network config: %+v\n", config)

	// Create the network using the graymatter library
	network, err := graymatter.NewNetwork(config)
	if err != nil {
		return err
	}

	// Store the network in our classifier for later use
	ic.network = network

	// Debug output to verify network creation
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
// Machine learning can be opaque - it's hard to understand what's happening inside
// the network during training. Plots provide crucial insights that help us:
// - Verify we're learning: loss should decrease over time
// - Detect overfitting: validation loss increases while training loss decreases
// - Check architecture health: weight distributions should look reasonable
// - Identify problem classes: confusion matrix shows which characters are mixed up
//
// PLOTS WE GENERATE:
// 1. Network architecture diagram: Visual representation of the network structure
// 2. Weight distributions: Shows the health of the learning process
// 3. Confusion matrix: Reveals which characters are confused with each other
func (ic *ImageClassifier) generateAnalysisPlots(trainData, validData *graymatter.DataSet) error {
	// Create a client to communicate with the plotting service
	// (The plotting service is a separate Python application that creates charts)
	plotClient := graymatter.NewPlottingClient(ic.config.PlottingURL)

	// STEP 1: Verify plotting service is available
	// Training can take hours - we want to catch plotting issues early rather than
	// discovering them at the end
	healthResp, err := plotClient.CheckHealth()
	if err != nil {
		return fmt.Errorf("plotting service unavailable: %w", err)
	}

	fmt.Printf("Plotting service health: %s (version: %s)\n", healthResp.Status, healthResp.Version)

	// STEP 2: Generate network architecture visualization
	// This creates a diagram showing how the layers connect to each other
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
	// Weight distributions reveal important information about training health:
	// - Well-trained networks typically have weights distributed around zero
	// - Very large or very small weights can indicate training problems
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
		fmt.Printf("Weight distribution plot: %s\n", weightResp.FilePath)
	}

	// STEP 4: Generate confusion matrix for validation set
	// This shows which characters the network tends to confuse with each other
	fmt.Println("Generating confusion matrix...")
	if err := ic.generateConfusionMatrix(plotClient, validData); err != nil {
		fmt.Printf("Warning: Failed to generate confusion matrix: %v\n", err)
	}

	return nil
}

// generateConfusionMatrix creates a confusion matrix plot for the validation set.
//
// WHAT IS A CONFUSION MATRIX?
// A confusion matrix is a table that shows how often each true character is
// predicted as each possible character. It's called "confusion" because it
// reveals which classes the model confuses with each other.
//
// EXAMPLE 3×3 CONFUSION MATRIX (for 3 characters):
//           Predicted
//         A   B   C
// True A [90   5   5]  ← 90% correct, 5% confused with B, 5% with C
//      B [ 3  85  12]  ← 3% confused with A, 85% correct, 12% with C
//      C [ 2   8  90]  ← 2% confused with A, 8% with B, 90% correct
//
// READING THE MATRIX:
// - Diagonal elements (top-left to bottom-right) = correct predictions
// - Off-diagonal elements = misclassifications
// - Perfect predictions would show high numbers on diagonal, zeros elsewhere
//
// WHY THIS MATTERS:
// - Identifies systematic problems (e.g., is 'O' always confused with '0'?)
// - Reveals which classes need more training data or better features
// - Helps understand model limitations and guides improvements
// - Shows which character pairs are inherently difficult to distinguish
func (ic *ImageClassifier) generateConfusionMatrix(plotClient *graymatter.PlottingClient, validData *graymatter.DataSet) error {
	// STEP 1: Get predictions from the trained network
	// Run all validation examples through the network to see what it predicts
	predictions, err := ic.network.Predict(validData.Inputs)
	if err != nil {
		return fmt.Errorf("failed to get predictions: %w", err)
	}

	// STEP 2: Generate confusion matrix using the library's utility
	// This compares predictions against true labels and counts matches/mismatches
	matrix, classLabels, err := graymatter.GenerateConfusionMatrix(predictions, validData.Outputs, 0.5)
	if err != nil {
		return fmt.Errorf("failed to generate confusion matrix: %w", err)
	}

	// STEP 3: Limit matrix size for readability
	// With 94 classes, a full confusion matrix would be quite large and hard to read.
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
// The graymatter library stores data as matrices, so we check the number of rows
// to get the sample count.
func (ic *ImageClassifier) getDatasetSize(dataset *graymatter.DataSet) int {
	rows, _ := dataset.Inputs.Dims()
	return rows
}

// TRAINING BEST PRACTICES DEMONSTRATED:

// 1. COMPREHENSIVE ERROR HANDLING:
// Every step can fail for various reasons (file I/O, memory issues, configuration errors).
// We provide meaningful error messages that help identify and fix problems quickly.

// 2. PROGRESS REPORTING:
// Long-running operations provide regular updates so users know the system is working
// and can estimate completion time. This is especially important for training that
// can take hours.

// 3. GRACEFUL DEGRADATION:
// If plotting fails, we warn the user but don't stop training. The core functionality
// (training the model) is more important than auxiliary features (generating plots).

// 4. SEPARATION OF CONCERNS:
// - TrainWithValidation: Orchestrates the overall training process
// - createNetwork: Handles neural network architecture setup
// - generateAnalysisPlots: Manages visualization and analysis
// Each function has a single, clear responsibility making code easier to maintain.

// 5. CONFIGURATION-DRIVEN BEHAVIOR:
// All settings come from the configuration struct, making it easy to experiment
// with different parameters without changing code.

// COMMON TRAINING PROBLEMS AND SOLUTIONS:

// 1. OVERFITTING:
// Problem: High training accuracy but low validation accuracy (memorizing vs learning)
// Solutions: More training data, simpler model, regularization techniques, early stopping

// 2. UNDERFITTING:
// Problem: Both training and validation accuracy are low (model too simple)
// Solutions: More complex model, more training iterations, better feature engineering

// 3. VANISHING GRADIENTS:
// Problem: Network stops learning because gradients become too small
// Solutions: Better weight initialization, different activation functions (like ReLU)

// 4. EXPLODING GRADIENTS:
// Problem: Network becomes unstable because gradients become too large
// Solutions: Lower learning rate, gradient clipping, better normalization

// 5. CLASS IMBALANCE:
// Problem: Some character classes have much more training data than others
// Solutions: Balanced sampling, weighted loss functions, data augmentation

// ALPHANUMERIC-SPECIFIC CONSIDERATIONS:

// 1. SIMILAR CHARACTERS:
// Some characters look very similar and are commonly confused:
// - 'O' (letter) vs '0' (zero) - often identical in many fonts
// - 'l' (lowercase L) vs '1' (one) vs 'I' (uppercase i) - can be very similar
// - 'S' vs '5' - particularly in handwritten or stylized fonts

// 2. CASE SENSITIVITY:
// The model learns both uppercase and lowercase versions of letters. Make sure
// training data includes good examples of both cases for each letter.

// 3. FONT VARIATIONS:
// Different fonts can make characters look quite different. Include variety in
// training data for better generalization across different text sources.

// 4. PUNCTUATION CHALLENGES:
// Punctuation marks are often smaller and more varied than letters, which can
// make them harder to recognize consistently.

// This training pipeline provides a solid foundation for alphanumeric character
// recognition and demonstrates best practices for neural network training.
// The 94-class architecture (A-Z, a-z, 0-9, plus punctuation) is well-suited
// for comprehensive text recognition applications.