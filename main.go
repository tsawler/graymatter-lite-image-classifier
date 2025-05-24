package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/tsawler/graymatter-lite"
)

const (
	// Image dimensions - assuming square images
	ImageWidth  = 28
	ImageHeight = 28

	// Network configuration
	InputSize  = ImageWidth * ImageHeight // 784 pixels
	HiddenSize = 128
	OutputSize = 62 // 26 uppercase + 26 lowercase + 10 digits
)

// ClassMapping maps class names to output indices
var ClassMapping = map[string]int{
	// Uppercase letters (A-Z): indices 0-25
	"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
	"K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19,
	"U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25,

	// Lowercase letters (a-z): indices 26-51
	"a": 26, "b": 27, "c": 28, "d": 29, "e": 30, "f": 31, "g": 32, "h": 33, "i": 34, "j": 35,
	"k": 36, "l": 37, "m": 38, "n": 39, "o": 40, "p": 41, "q": 42, "r": 43, "s": 44, "t": 45,
	"u": 46, "v": 47, "w": 48, "x": 49, "y": 50, "z": 51,

	// Digits (0-9): indices 52-61
	"0": 52, "1": 53, "2": 54, "3": 55, "4": 56, "5": 57, "6": 58, "7": 59, "8": 60, "9": 61,
}

// Reverse mapping for predictions
var IndexToClass = make(map[int]string)

func init() {
	// Create reverse mapping
	for class, index := range ClassMapping {
		IndexToClass[index] = class
	}
}

// ImageData represents a single training example
type ImageData struct {
	Pixels     []float64
	Label      string
	ClassIndex int
}

func main() {
	fmt.Println("Starting Image Classification Training...")

	// Load training data
	fmt.Println("Loading training data...")
	trainingData, err := loadTrainingData("data")
	if err != nil {
		log.Fatal("Failed to load training data:", err)
	}

	fmt.Printf("Loaded %d training samples\n", len(trainingData))

	// Convert to neural network format
	inputs, outputs, err := prepareDataForTraining(trainingData)
	if err != nil {
		log.Fatal("Failed to prepare training data:", err)
	}

	// Create dataset
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		log.Fatal("Failed to create dataset:", err)
	}

	// Split into training and validation sets
	trainData, validData, err := graymatter.SplitDataSet(dataset, graymatter.SplitOptions{
		TrainRatio: 0.8,
		Shuffle:    true,
	})
	if err != nil {
		log.Fatal("Failed to split dataset:", err)
	}

	fmt.Printf("Training samples: %d\n", getDatasetSize(trainData))
	fmt.Printf("Validation samples: %d\n", getDatasetSize(validData))

	// Create neural network
	fmt.Println("Creating neural network...")
	config := graymatter.NetworkConfig{
		LayerSizes:               []int{InputSize, HiddenSize, HiddenSize, OutputSize},
		HiddenActivationFunction: "relu",
		OutputActivationFunction: "softmax",
		CostFunction:             "categorical",
		Seed:                     42, // For reproducible results
	}

	network, err := graymatter.NewNetwork(config)
	if err != nil {
		log.Fatal("Failed to create network:", err)
	}

	// Train the network with updated plotting functionality
	fmt.Println("Starting training...")
	trainingOptions := graymatter.TrainingOptions{
		Iterations:     500,
		BatchSize:      32,
		LearningRate:   0.001,
		EnableSaving:   true,
		SavePath:       "image_classifier",
		SaveInterval:   100,                     // Save every 100 epochs
		EnablePlotting: true,                    // Enable the new plotting functionality
		PlottingURL:    "http://localhost:8080", // Updated to use the new API port
	}

	finalCost, err := network.TrainWithValidation(trainData, validData, trainingOptions)
	if err != nil {
		log.Fatal("Training failed:", err)
	}

	fmt.Printf("Training completed! Final cost: %.6f\n", finalCost)

	// Evaluate final performance
	trainAccuracy, err := network.CalculateAccuracy(trainData, 0.5)
	if err != nil {
		fmt.Printf("Warning: Could not calculate training accuracy: %v\n", err)
	} else {
		fmt.Printf("Final training accuracy: %.2f%%\n", trainAccuracy*100)
	}

	validAccuracy, err := network.CalculateAccuracy(validData, 0.5)
	if err != nil {
		fmt.Printf("Warning: Could not calculate validation accuracy: %v\n", err)
	} else {
		fmt.Printf("Final validation accuracy: %.2f%%\n", validAccuracy*100)
	}

	// Generate additional analysis plots using the new plotting capabilities
	fmt.Println("\nGenerating additional analysis plots...")
	err = generateComprehensiveAnalysis(network, trainData, validData)
	if err != nil {
		fmt.Printf("Warning: Failed to generate comprehensive analysis: %v\n", err)
	}

	// Make prediction on "a.png"
	fmt.Println("\nMaking prediction on 'a.png'...")
	prediction, confidence, err := predictImage(network, "a.png")
	if err != nil {
		log.Printf("Failed to predict image: %v", err)
	} else {
		fmt.Printf("Prediction: '%s' (confidence: %.2f%%)\n", prediction, confidence*100)
	}

	fmt.Println("Program completed successfully!")
}

// generateComprehensiveAnalysis creates additional plots using the new plotting API
func generateComprehensiveAnalysis(network *graymatter.Network, trainData, validData *graymatter.DataSet) error {
	// Create plotting client
	plotClient := graymatter.NewPlottingClient("http://localhost:8080")

	// Check if plotting service is available
	healthResp, err := plotClient.CheckHealth()
	if err != nil {
		return fmt.Errorf("plotting service unavailable: %w", err)
	}

	fmt.Printf("Plotting service health: %s (version: %s)\n", healthResp.Status, healthResp.Version)

	// Generate network architecture visualization
	fmt.Println("Generating network architecture diagram...")
	archReq := graymatter.CreateNetworkArchitectureVisualization(
		network,
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
	weightStats := graymatter.ExtractWeightStatistics(network)

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
	err = generateConfusionMatrix(plotClient, network, validData)
	if err != nil {
		fmt.Printf("Warning: Failed to generate confusion matrix: %v\n", err)
	}

	return nil
}

// generateConfusionMatrix creates a confusion matrix plot for the validation set
func generateConfusionMatrix(plotClient *graymatter.PlottingClient, network *graymatter.Network, validData *graymatter.DataSet) error {
	// Get predictions for validation set
	predictions, err := network.Predict(validData.Inputs)
	if err != nil {
		return fmt.Errorf("failed to get predictions: %w", err)
	}

	// Generate confusion matrix
	matrix, classLabels, err := graymatter.GenerateConfusionMatrix(predictions, validData.Outputs, 0.5)
	if err != nil {
		return fmt.Errorf("failed to generate confusion matrix: %w", err)
	}

	// For display purposes, limit to a subset of classes if there are too many
	// (62 classes would make a very large confusion matrix)
	if len(classLabels) > 10 {
		fmt.Println("Note: Limiting confusion matrix to first 10 classes for readability")
		// Truncate matrix and labels to first 10 classes
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

// loadTrainingData loads all images from the data directory structure
func loadTrainingData(dataDir string) ([]ImageData, error) {
	var allData []ImageData

	// Define the subdirectories and their patterns
	subdirs := map[string]string{
		"upper":  "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
		"lower":  "abcdefghijklmnopqrstuvwxyz",
		"digits": "0123456789",
	}

	for subdir, chars := range subdirs {
		subdirPath := filepath.Join(dataDir, subdir)

		// Check if subdirectory exists
		if _, err := os.Stat(subdirPath); os.IsNotExist(err) {
			fmt.Printf("Warning: Directory %s does not exist, skipping...\n", subdirPath)
			continue
		}

		// Process each character directory
		for _, char := range chars {
			charDir := filepath.Join(subdirPath, string(char))

			if _, err := os.Stat(charDir); os.IsNotExist(err) {
				fmt.Printf("Warning: Directory %s does not exist, skipping...\n", charDir)
				continue
			}

			// Load all images from this character directory
			charData, err := loadImagesFromDirectory(charDir, string(char))
			if err != nil {
				return nil, fmt.Errorf("failed to load images from %s: %w", charDir, err)
			}

			allData = append(allData, charData...)
			fmt.Printf("Loaded %d images for class '%s'\n", len(charData), string(char))
		}
	}

	if len(allData) == 0 {
		return nil, fmt.Errorf("no training data found in %s", dataDir)
	}

	return allData, nil
}

// loadImagesFromDirectory loads all images from a specific directory
func loadImagesFromDirectory(dir, label string) ([]ImageData, error) {
	var images []ImageData

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Check if file is an image
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
			return nil
		}

		// Load and process the image
		pixels, err := loadAndProcessImage(path)
		if err != nil {
			fmt.Printf("Warning: Failed to load image %s: %v\n", path, err)
			return nil // Continue with other images
		}

		classIndex, exists := ClassMapping[label]
		if !exists {
			return fmt.Errorf("unknown class label: %s", label)
		}

		images = append(images, ImageData{
			Pixels:     pixels,
			Label:      label,
			ClassIndex: classIndex,
		})

		return nil
	})

	return images, err
}

// loadAndProcessImage loads an image file and converts it to normalized pixel values
func loadAndProcessImage(imagePath string) ([]float64, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	// Decode image based on file extension
	var img image.Image
	ext := strings.ToLower(filepath.Ext(imagePath))

	switch ext {
	case ".png":
		img, err = png.Decode(file)
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(file)
	default:
		return nil, fmt.Errorf("unsupported image format: %s", ext)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Convert to grayscale and resize if necessary
	pixels := imageToPixels(img)

	if len(pixels) != InputSize {
		return nil, fmt.Errorf("image has %d pixels, expected %d (ensure images are %dx%d)",
			len(pixels), InputSize, ImageWidth, ImageHeight)
	}

	return pixels, nil
}

// imageToPixels converts an image to normalized pixel values
func imageToPixels(img image.Image) []float64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// For simplicity, assume images are already the correct size
	// In a production system, you'd want to resize images here
	pixels := make([]float64, width*height)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// Get pixel color
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert to grayscale using luminance formula
			// RGBA values are in range 0-65535, so divide by 65535 to get 0-1
			gray := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 65535.0

			// Normalize to 0-1 range (already done above) and store
			pixelIndex := (y-bounds.Min.Y)*width + (x - bounds.Min.X)
			pixels[pixelIndex] = gray
		}
	}

	return pixels
}

// prepareDataForTraining converts ImageData to the format needed by the neural network
func prepareDataForTraining(data []ImageData) ([][]float64, [][]float64, error) {
	numSamples := len(data)

	inputs := make([][]float64, numSamples)
	outputs := make([][]float64, numSamples)

	for i, sample := range data {
		// Input is the pixel data
		inputs[i] = sample.Pixels

		// Output is one-hot encoded class
		oneHot := make([]float64, OutputSize)
		oneHot[sample.ClassIndex] = 1.0
		outputs[i] = oneHot
	}

	return inputs, outputs, nil
}

// getDatasetSize returns the number of samples in a dataset
func getDatasetSize(dataset *graymatter.DataSet) int {
	rows, _ := dataset.Inputs.Dims()
	return rows
}

// predictImage makes a prediction on a single image file
func predictImage(network *graymatter.Network, imagePath string) (string, float64, error) {
	// Load and process the image
	pixels, err := loadAndProcessImage(imagePath)
	if err != nil {
		return "", 0, fmt.Errorf("failed to load image: %w", err)
	}

	// Convert to matrix format (1 sample)
	inputs := [][]float64{pixels}
	inputMatrix, err := graymatter.NewDataSet(inputs, [][]float64{{0}}) // Dummy output
	if err != nil {
		return "", 0, fmt.Errorf("failed to create input matrix: %w", err)
	}

	// Make prediction
	predictions, err := network.Predict(inputMatrix.Inputs)
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
