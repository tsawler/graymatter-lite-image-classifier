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
//   punctuation/    <- Punctuation marks
//     asterisk/     <- * symbol
//       img1.png
//     dot/          <- . symbol
//       img1.png
//     exclamation/  <- ! symbol
//       img1.png
//
// WHY GROUP BY CHARACTER TYPE?
// Separating uppercase, lowercase, digits, and punctuation makes the data 
// organization cleaner and allows for easier dataset analysis. You can quickly 
// see how many examples you have of each type and ensure balanced representation.
func (ic *ImageClassifier) loadTrainingData() ([]ImageData, error) {
	var allData []ImageData

	// Define the character groups we want to process
	// Each group corresponds to a subdirectory and processing method
	classGroups := []ClassGroup{
		{"upper", 'A', 'Z'},   // Uppercase A through Z
		{"lower", 'a', 'z'},   // Lowercase a through z  
		{"digits", '0', '9'},  // Digits 0 through 9
	}

	// Process traditional character groups (letters and digits)
	for _, group := range classGroups {
		// Construct the path to this group's subdirectory
		subdirPath := filepath.Join(ic.config.DataDir, group.DirName)

		// CHECK IF SUBDIRECTORY EXISTS:
		// Not all datasets might have all groups. We gracefully skip missing directories
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

	// PUNCTUATION HANDLING:
	// Punctuation marks require special handling because:
	// 1. Some characters can't be used as directory names (/, ?, *, etc.)
	// 2. We use descriptive names like "asterisk" instead of "*"
	// 3. We need to map directory names back to actual characters
	punctuationDir := filepath.Join(ic.config.DataDir, "punctuation")
	if _, err := os.Stat(punctuationDir); err == nil {
		fmt.Println("Processing punctuation characters...")
		
		// Get list of punctuation subdirectories
		punctDirs, err := os.ReadDir(punctuationDir)
		if err != nil {
			return nil, fmt.Errorf("failed to read punctuation directory: %w", err)
		}

		// Process each punctuation subdirectory
		for _, dir := range punctDirs {
			if !dir.IsDir() {
				continue // Skip files, only process directories
			}

			dirName := dir.Name()
			
			// DIRECTORY NAME TO CHARACTER MAPPING:
			// Convert directory name (like "asterisk") to actual character ("*")
			actualChar, exists := PunctuationDirToChar[dirName]
			if !exists {
				fmt.Printf("Warning: Unknown punctuation directory '%s', skipping...\n", dirName)
				continue
			}

			// Construct path to this punctuation character's directory
			punctCharDir := filepath.Join(punctuationDir, dirName)

			// Load all images for this punctuation character
			punctData, err := ic.loadImagesFromDirectory(punctCharDir, actualChar)
			if err != nil {
				fmt.Printf("Warning: Failed to load images from %s: %v\n", punctCharDir, err)
				continue // Continue with other punctuation marks
			}

			// Add this punctuation character's data to our collection
			allData = append(allData, punctData...)
			fmt.Printf("Loaded %d images for punctuation class '%s' (from dir '%s')\n", 
				len(punctData), actualChar, dirName)
		}
	} else {
		fmt.Printf("Warning: Punctuation directory %s does not exist, skipping...\n", punctuationDir)
	}

	// VALIDATION: Ensure we actually loaded some data
	// Training with zero examples would be impossible, so we catch this error early
	if len(allData) == 0 {
		return nil, fmt.Errorf("no training data found in %s", ic.config.DataDir)
	}

	fmt.Printf("Total training samples loaded: %d\n", len(allData))
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
//
// UPDATED FOR PUNCTUATION:
// This function now handles any character label, including punctuation marks.
// The label parameter can be a letter ("A"), digit ("7"), or punctuation ("*").
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
		// string label (like "A", "*", "?") to a numerical index using
		// our predefined mapping.
		classIndex, exists := ClassMapping[label]
		if !exists {
			return fmt.Errorf("unknown class label: %s", label)
		}

		// CREATE THE TRAINING EXAMPLE:
		// Combine the numerical pixel data with the label information
		images = append(images, ImageData{
			Pixels:     pixels,     // The actual image data as numbers
			Label:      label,      // Human-readable label ("A", "b", "7", "*")
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
//
// UPDATED FOR 94 CLASSES:
// Now handles 94 different character classes instead of 62, including
// all punctuation marks in addition to letters and digits.
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
		oneHot := make([]float64, ic.config.OutputSize) // All zeros initially (94 elements)
		oneHot[sample.ClassIndex] = 1.0                 // Set correct class to 1
		outputs[i] = oneHot

		// EXAMPLE OF ONE-HOT ENCODING FOR 94 CLASSES:
		// If we have 94 classes and the sample is class 62 (first punctuation):
		// ClassIndex = 62
		// oneHot = [0, 0, ..., 1, 0, ..., 0]  <- Only position 62 is 1 (94 elements total)
		//
		// For our expanded character set:
		// If sample is "A" (ClassIndex = 0): [1, 0, 0, 0, ..., 0] (94 elements)
		// If sample is "a" (ClassIndex = 26): [0, 0, ..., 1, 0, ..., 0] (1 at position 26)
		// If sample is "*" (ClassIndex = 62): [0, 0, ..., 1, 0, ..., 0] (1 at position 62)
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

// 6. EXTENSIBLE DESIGN:
// - Easy to add new character types by extending the directory structure
// - Punctuation handling demonstrates how to handle special cases
// - Maintains backward compatibility with existing data organization

// DATA QUALITY CONSIDERATIONS FOR PUNCTUATION:

// 1. BALANCED DATASETS:
// Punctuation marks might be less common in typical text, so you may need
// more training examples per punctuation mark to achieve balanced representation.

// 2. SIMILAR-LOOKING CHARACTERS:
// Some punctuation marks are visually similar (like "." and ",") and may be
// more challenging to distinguish. Consider collecting extra examples of
// these characters.

// 3. FONT VARIATIONS:
// Punctuation marks can look very different across fonts. Ensure your
// training data includes sufficient variety to generalize well.

// This enhanced data loading pipeline now supports the full range of
// printable ASCII characters, making it suitable for comprehensive
// character recognition tasks including document digitization and
// optical character recognition (OCR) applications.