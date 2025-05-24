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
	// Image processing constants
	RedLuminance   = 0.299
	GreenLuminance = 0.587
	BlueLuminance  = 0.114
	RGBAMax        = 65535.0
)

// Config holds all configuration parameters
type Config struct {
	ImageWidth, ImageHeight           int
	InputSize, HiddenSize, OutputSize int
	DataDir                           string
	ModelPath                         string
	PlottingURL                       string
	TrainingOptions                   graymatter.TrainingOptions
}

// NewDefaultConfig creates a configuration with sensible defaults
func NewDefaultConfig() *Config {
	imageWidth, imageHeight := 28, 28
	return &Config{
		ImageWidth:  imageWidth,
		ImageHeight: imageHeight,
		InputSize:   imageWidth * imageHeight, // 784 pixels
		HiddenSize:  128,
		OutputSize:  62, // 26 uppercase + 26 lowercase + 10 digits
		DataDir:     "data",
		ModelPath:   "image_classifier",
		PlottingURL: "http://localhost:8080",
		TrainingOptions: graymatter.TrainingOptions{
			Iterations:     500,
			BatchSize:      32,
			LearningRate:   0.001,
			EnableSaving:   true,
			SaveInterval:   100,
			EnablePlotting: true,
		},
	}
}

// ClassMapping maps class names to output indices
var ClassMapping = generateClassMapping()
var IndexToClass = make(map[int]string)

// generateClassMapping creates the mapping programmatically
func generateClassMapping() map[string]int {
	mapping := make(map[string]int)
	index := 0

	// Uppercase A-Z (indices 0-25)
	for i := 'A'; i <= 'Z'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	// Lowercase a-z (indices 26-51)
	for i := 'a'; i <= 'z'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	// Digits 0-9 (indices 52-61)
	for i := '0'; i <= '9'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	return mapping
}

// ClassGroup represents a group of characters with their directory info
type ClassGroup struct {
	DirName              string
	StartChar, EndChar   rune
}

// ImageData represents a single training example
type ImageData struct {
	Pixels     []float64
	Label      string
	ClassIndex int
}

// ImageClassifier encapsulates the training and prediction logic
type ImageClassifier struct {
	config  *Config
	network *graymatter.Network
}

// NewImageClassifier creates a new image classifier
func NewImageClassifier(config *Config) *ImageClassifier {
	return &ImageClassifier{config: config}
}

func main() {
	fmt.Println("Starting Image Classification Training...")

	config := NewDefaultConfig()
	classifier := NewImageClassifier(config)

	if err := classifier.Train(); err != nil {
		log.Fatal("Training failed:", err)
	}

	// Make prediction on "a.png"
	fmt.Println("\nMaking prediction on 'a.png'...")
	prediction, confidence, err := classifier.Predict("a.png")
	if err != nil {
		log.Printf("Failed to predict image: %v", err)
	} else {
		fmt.Printf("Prediction: '%s' (confidence: %.2f%%)\n", prediction, confidence*100)
	}

	fmt.Println("Program completed successfully!")
}

// Train handles the complete training process
func (ic *ImageClassifier) Train() error {
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

	// Create dataset
	dataset, err := graymatter.NewDataSet(inputs, outputs)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %w", err)
	}

	// Split into training and validation sets
	trainData, validData, err := graymatter.SplitDataSet(dataset, graymatter.SplitOptions{
		TrainRatio: 0.8,
		Shuffle:    true,
	})
	if err != nil {
		return fmt.Errorf("failed to split dataset: %w", err)
	}

	fmt.Printf("Training samples: %d\n", ic.getDatasetSize(trainData))
	fmt.Printf("Validation samples: %d\n", ic.getDatasetSize(validData))

	// Create neural network
	fmt.Println("Creating neural network...")
	if err := ic.createNetwork(); err != nil {
		return fmt.Errorf("failed to create network: %w", err)
	}

	// Train the network
	fmt.Println("Starting training...")
	ic.config.TrainingOptions.SavePath = ic.config.ModelPath
	ic.config.TrainingOptions.PlottingURL = ic.config.PlottingURL

	finalCost, err := ic.network.TrainWithValidation(trainData, validData, ic.config.TrainingOptions)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	fmt.Printf("Training completed! Final cost: %.6f\n", finalCost)

	// Evaluate final performance
	if err := ic.evaluatePerformance(trainData, validData); err != nil {
		fmt.Printf("Warning: Performance evaluation failed: %v\n", err)
	}

	// Generate additional analysis plots
	fmt.Println("\nGenerating additional analysis plots...")
	if err := ic.generateComprehensiveAnalysis(trainData, validData); err != nil {
		fmt.Printf("Warning: Failed to generate comprehensive analysis: %v\n", err)
	}

	return nil
}

// createNetwork initializes the neural network
func (ic *ImageClassifier) createNetwork() error {
	config := graymatter.NetworkConfig{
		LayerSizes:               []int{ic.config.InputSize, ic.config.HiddenSize, ic.config.HiddenSize, ic.config.OutputSize},
		HiddenActivationFunction: "relu",
		OutputActivationFunction: "softmax",
		CostFunction:             "categorical",
		Seed:                     42, // For reproducible results
	}

	network, err := graymatter.NewNetwork(config)
	if err != nil {
		return err
	}

	ic.network = network
	return nil
}

// evaluatePerformance calculates and reports training and validation accuracy
func (ic *ImageClassifier) evaluatePerformance(trainData, validData *graymatter.DataSet) error {
	trainAccuracy, err := ic.network.CalculateAccuracy(trainData, 0.5)
	if err != nil {
		fmt.Printf("Warning: Could not calculate training accuracy: %v\n", err)
	} else {
		fmt.Printf("Final training accuracy: %.2f%%\n", trainAccuracy*100)
	}

	validAccuracy, err := ic.network.CalculateAccuracy(validData, 0.5)
	if err != nil {
		fmt.Printf("Warning: Could not calculate validation accuracy: %v\n", err)
	} else {
		fmt.Printf("Final validation accuracy: %.2f%%\n", validAccuracy*100)
	}

	return nil
}

// loadTrainingData loads all images from the data directory structure
func (ic *ImageClassifier) loadTrainingData() ([]ImageData, error) {
	var allData []ImageData

	classGroups := []ClassGroup{
		{"upper", 'A', 'Z'},
		{"lower", 'a', 'z'},
		{"digits", '0', '9'},
	}

	for _, group := range classGroups {
		subdirPath := filepath.Join(ic.config.DataDir, group.DirName)

		if _, err := os.Stat(subdirPath); os.IsNotExist(err) {
			fmt.Printf("Warning: Directory %s does not exist, skipping...\n", subdirPath)
			continue
		}

		for char := group.StartChar; char <= group.EndChar; char++ {
			charDir := filepath.Join(subdirPath, string(char))

			if _, err := os.Stat(charDir); os.IsNotExist(err) {
				fmt.Printf("Warning: Directory %s does not exist, skipping...\n", charDir)
				continue
			}

			charData, err := ic.loadImagesFromDirectory(charDir, string(char))
			if err != nil {
				return nil, fmt.Errorf("failed to load images from %s: %w", charDir, err)
			}

			allData = append(allData, charData...)
			fmt.Printf("Loaded %d images for class '%s'\n", len(charData), string(char))
		}
	}

	if len(allData) == 0 {
		return nil, fmt.Errorf("no training data found in %s", ic.config.DataDir)
	}

	return allData, nil
}

// loadImagesFromDirectory loads all images from a specific directory
func (ic *ImageClassifier) loadImagesFromDirectory(dir, label string) ([]ImageData, error) {
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
		pixels, err := ic.loadAndProcessImage(path)
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
func (ic *ImageClassifier) loadAndProcessImage(imagePath string) ([]float64, error) {
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
	pixels := ic.imageToPixels(img)

	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("image has %d pixels, expected %d (ensure images are %dx%d)",
			len(pixels), ic.config.InputSize, ic.config.ImageWidth, ic.config.ImageHeight)
	}

	return pixels, nil
}

// imageToPixels converts an image to normalized pixel values
func (ic *ImageClassifier) imageToPixels(img image.Image) []float64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	pixels := make([]float64, width*height)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert to grayscale using standard luminance formula
			gray := (RedLuminance*float64(r) + GreenLuminance*float64(g) + BlueLuminance*float64(b)) / RGBAMax

			pixelIndex := (y-bounds.Min.Y)*width + (x-bounds.Min.X)
			pixels[pixelIndex] = gray
		}
	}

	return pixels
}

// prepareDataForTraining converts ImageData to the format needed by the neural network
func (ic *ImageClassifier) prepareDataForTraining(data []ImageData) ([][]float64, [][]float64, error) {
	numSamples := len(data)

	inputs := make([][]float64, numSamples)
	outputs := make([][]float64, numSamples)

	for i, sample := range data {
		// Input is the pixel data
		inputs[i] = sample.Pixels

		// Output is one-hot encoded class
		oneHot := make([]float64, ic.config.OutputSize)
		oneHot[sample.ClassIndex] = 1.0
		outputs[i] = oneHot
	}

	return inputs, outputs, nil
}

// getDatasetSize returns the number of samples in a dataset
func (ic *ImageClassifier) getDatasetSize(dataset *graymatter.DataSet) int {
	rows, _ := dataset.Inputs.Dims()
	return rows
}

// Predict makes a prediction on a single image file
func (ic *ImageClassifier) Predict(imagePath string) (string, float64, error) {
	// Load and process the image
	pixels, err := ic.loadAndProcessImage(imagePath)
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
	predictions, err := ic.network.Predict(inputMatrix.Inputs)
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

// generateComprehensiveAnalysis creates additional plots using the new plotting API
func (ic *ImageClassifier) generateComprehensiveAnalysis(trainData, validData *graymatter.DataSet) error {
	// Create plotting client
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