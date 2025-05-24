package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

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