package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// loadTrainingData loads all images from the data directory structure.
//
// DATA ORGANIZATION PHILOSOPHY:
// This function implements a common machine learning pattern where training data
// is organized by class in separate directories. The directory structure becomes
// the labeling system - all images in "data/upper/A/" are automatically labeled
// as the character "A". This approach is:
// - Self-documenting (easy to see what data you have)
// - Less error-prone than separate label files
// - Easy to organize and manage by humans
//
// EXPECTED DIRECTORY STRUCTURE:
// data/
//   upper/          <- Uppercase letters
//     A/
//       img1.png
//       img2.png
//     B/
//       img1.png
//   lower/          <- Lowercase letters  
//     a/
//       img1.png
//     b/
//       img1.png
//   digits/         <- Numbers
//     0/
//       img1.png
//     1/
//       img1.png
//
// WHY GROUP BY CHARACTER TYPE?
// Separating uppercase, lowercase, and digits makes the data organization
// cleaner and allows for easier dataset analysis. You can quickly see how
// many examples you have of each type and ensure balanced representation.
func (ic *ImageClassifier) loadTrainingData() ([]ImageData, error) {
	var allData []ImageData

	// Define the character groups we want to process
	// Each group corresponds to a subdirectory and a range of characters
	classGroups := []ClassGroup{
		{"upper", 'A', 'Z'},   // Uppercase A through Z
		{"lower", 'a', 'z'},   // Lowercase a through z  
		{"digits", '0', '9'},  // Digits 0 through 9
	}

	// Process each character group
	for _, group := range classGroups {
		// Construct the path to this group's subdirectory
		subdirPath := filepath.Join(ic.config.DataDir, group.DirName)

		// CHECK IF SUBDIRECTORY EXISTS:
		// Not all datasets might have all three groups. For example, you might
		// only have uppercase letters. We gracefully skip missing directories
		// rather than failing completely.
		if _, err := os.Stat(subdirPath); os.IsNotExist(err) {
			fmt.Printf("Warning: Directory %s does not exist, skipping...\n", subdirPath)
			continue
		}

		// Process each character in this group's range
		// Go's character arithmetic lets us iterate through character ranges
		// cleanly without hardcoding every character
		for char := group.StartChar; char <= group.EndChar; char++ {
			// Construct path to this specific character's directory
			charDir := filepath.Join(subdirPath, string(char))

			// Check if this character's directory exists
			// Some datasets might be incomplete (missing certain characters)
			if _, err := os.Stat(charDir); os.IsNotExist(err) {
				fmt.Printf("Warning: Directory %s does not exist, skipping...\n", charDir)
				continue
			}

			// Load all images for this character
			charData, err := ic.loadImagesFromDirectory(charDir, string(char))
			if err != nil {
				return nil, fmt.Errorf("failed to load images from %s: %w", charDir, err)
			}

			// Add this character's data to our growing collection
			allData = append(allData, charData...)
			fmt.Printf("Loaded %d images for class '%s'\n", len(charData), string(char))
		}
	}

	// VALIDATION: Ensure we actually loaded some data
	// Training with zero examples would be impossible, so we catch this error early
	if len(allData) == 0 {
		return nil, fmt.Errorf("no training data found in %s", ic.config.DataDir)
	}

	return allData, nil
}

// loadImagesFromDirectory loads all images from a specific directory.
//
// SINGLE RESPONSIBILITY:
// This function has one job: given a directory path and a label, load all
// image files from that directory and associate them with that label.
// This separation of concerns makes the code easier to test and debug.
//
// ERROR HANDLING PHILOSOPHY:
// We use a "resilient" approach to individual file errors - if one image
// fails to load (corrupted file, wrong format, etc.), we log a warning
// but continue processing other images. This prevents a single bad file
// from destroying an entire training run.
func (ic *ImageClassifier) loadImagesFromDirectory(dir, label string) ([]ImageData, error) {
	var images []ImageData

	// RECURSIVE DIRECTORY WALKING:
	// filepath.Walk recursively visits every file and subdirectory.
	// This is more robust than manual directory reading because it handles
	// nested subdirectories and provides consistent cross-platform behavior.
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		// Handle walk errors (permissions, broken symlinks, etc.)
		if err != nil {
			return err
		}

		// Skip directories - we only want files
		if info.IsDir() {
			return nil
		}

		// FILTER BY FILE EXTENSION:
		// Only process files that are likely to be images
		// This prevents us from trying to process README files, .DS_Store, etc.
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
			return nil // Skip non-image files
		}

		// ATTEMPT TO LOAD AND PROCESS THE IMAGE:
		// This is where the image gets converted from a file to numerical data
		pixels, err := ic.loadAndProcessImage(path)
		if err != nil {
			// RESILIENT ERROR HANDLING:
			// Log the problem but don't stop processing. A few bad images
			// shouldn't prevent training on thousands of good images.
			fmt.Printf("Warning: Failed to load image %s: %v\n", path, err)
			return nil // Continue with other images
		}

		// CONVERT LABEL TO CLASS INDEX:
		// Neural networks work with numbers, not strings. We convert the
		// string label (like "A") to a numerical index (like 0) using
		// our predefined mapping.
		classIndex, exists := ClassMapping[label]
		if !exists {
			return fmt.Errorf("unknown class label: %s", label)
		}

		// CREATE THE TRAINING EXAMPLE:
		// Combine the numerical pixel data with the label information
		images = append(images, ImageData{
			Pixels:     pixels,     // The actual image data as numbers
			Label:      label,      // Human-readable label ("A", "b", "7")
			ClassIndex: classIndex, // Numerical index for the neural network
		})

		return nil
	})

	return images, err
}

// prepareDataForTraining converts ImageData to the format needed by the neural network.
//
// FROM HUMAN FORMAT TO NEURAL NETWORK FORMAT:
// This function handles the final conversion step - taking our collected
// ImageData structs and converting them into the matrix format that the
// neural network library expects. This is where individual examples become
// part of a batch that can be processed efficiently.
//
// MATRIX FORMAT REQUIREMENTS:
// Neural networks work with matrices where:
// - Each row represents one training example (one image)
// - Each column represents one feature (one pixel value)
// - Labels become "one-hot encoded" vectors
//
// ONE-HOT ENCODING EXPLAINED:
// Instead of representing the label "A" as the number 0, we represent it as
// a vector like [1, 0, 0, 0, 0, ...] where only the position corresponding
// to "A" contains a 1. This format works better with the softmax output
// layer that produces probability distributions.
func (ic *ImageClassifier) prepareDataForTraining(data []ImageData) ([][]float64, [][]float64, error) {
	numSamples := len(data)

	// ALLOCATE OUTPUT MATRICES:
	// Pre-allocate the slices to avoid repeated memory allocations during the loop
	inputs := make([][]float64, numSamples)   // Each row is pixel values for one image
	outputs := make([][]float64, numSamples)  // Each row is one-hot encoded label

	// CONVERT EACH TRAINING EXAMPLE:
	for i, sample := range data {
		// INPUT PREPARATION:
		// The pixel data is already in the right format - just assign it
		inputs[i] = sample.Pixels

		// OUTPUT PREPARATION (ONE-HOT ENCODING):
		// Create a vector of all zeros, then set the correct class position to 1
		oneHot := make([]float64, ic.config.OutputSize) // All zeros initially
		oneHot[sample.ClassIndex] = 1.0                 // Set correct class to 1
		outputs[i] = oneHot

		// EXAMPLE OF ONE-HOT ENCODING:
		// If we have 4 classes and the sample is class 2:
		// ClassIndex = 2
		// oneHot = [0, 0, 1, 0]  <- Only position 2 is 1
		//
		// For our 62-class problem:
		// If sample is "A" (ClassIndex = 0): [1, 0, 0, 0, ..., 0] (62 elements)
		// If sample is "B" (ClassIndex = 1): [0, 1, 0, 0, ..., 0] (62 elements)
		// If sample is "a" (ClassIndex = 26): [0, 0, ..., 1, 0, ..., 0] (1 at position 26)
	}

	return inputs, outputs, nil
}

// DATA LOADING BEST PRACTICES DEMONSTRATED:

// 1. ROBUST FILE SYSTEM HANDLING:
// - Check if directories exist before trying to read them
// - Handle file system errors gracefully
// - Use cross-platform path manipulation (filepath.Join)

// 2. RESILIENT ERROR HANDLING:
// - Individual file failures don't stop the entire process
// - Clear warning messages help identify data quality issues
// - Early validation prevents training with empty datasets

// 3. EFFICIENT MEMORY USAGE:
// - Pre-allocate slices when final size is known
// - Process files one at a time rather than loading everything into memory first
// - Use appropriate data structures for the task

// 4. CLEAR SEPARATION OF CONCERNS:
// - loadTrainingData: Orchestrates the overall process
// - loadImagesFromDirectory: Handles a single directory
// - prepareDataForTraining: Converts to neural network format
// Each function has a single, clear responsibility

// 5. USER-FRIENDLY FEEDBACK:
// - Progress messages show what's happening during long operations
// - Warning messages help identify data issues
// - Meaningful error messages aid in debugging

// DATA QUALITY CONSIDERATIONS:

// 1. BALANCED DATASETS:
// Ideally, you want roughly the same number of examples for each character.
// If you have 1000 examples of "A" but only 10 examples of "Z", the network
// will be biased toward predicting "A" more often.

// 2. DATA AUGMENTATION:
// For small datasets, you might want to artificially increase variety by
// rotating, scaling, or adding noise to existing images. This helps the
// network generalize better to new examples.

// 3. QUALITY CONTROL:
// Bad training data leads to poor models. Consider adding validation to
// detect obviously corrupted images, incorrectly labeled examples, or
// images that don't match the expected format.

// This data loading pipeline forms the foundation of successful machine learning.
// High-quality, well-organized training data is often more important than
// sophisticated algorithms or architectures.