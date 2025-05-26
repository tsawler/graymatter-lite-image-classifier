package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// loadTrainingData loads all images from the data directory structure.
//
// DATA ORGANIZATION PHILOSOPHY:
// This function implements a common machine learning pattern where training data
// is organized by class in separate directories. The directory structure becomes
// the labeling system - all images in "data/upper/A/" are automatically labeled
// as the character "A". This approach has several advantages:
// - Self-documenting: Easy to see what data you have at a glance
// - Less error-prone than maintaining separate label files
// - Easy to organize and manage manually
// - Natural way to add new examples (just drop files in the right folder)
//
// EXPECTED DIRECTORY STRUCTURE:
// data/
//   upper/          ← Uppercase letters (A-Z)
//     A/
//       img1.png
//       img2.png
//       ... (thousands of examples)
//     B/
//       img1.png
//       img2.png
//   lower/          ← Lowercase letters (a-z)
//     a/
//       img1.png
//     b/
//       img1.png
//   digits/         ← Numbers (0-9)
//     0/
//       img1.png
//     1/
//       img1.png
//   punctuation/    ← Punctuation marks (32 different symbols)
//     asterisk/     ← * symbol (uses descriptive name)
//       img1.png
//     dot/          ← . symbol
//       img1.png
//     exclamation/  ← ! symbol
//       img1.png
//
// WHY GROUP BY CHARACTER TYPE?
// Separating uppercase, lowercase, digits, and punctuation makes the data 
// organization cleaner and allows for easier dataset analysis. You can quickly 
// see how many examples you have of each type and ensure balanced representation
// across different character categories.
//
// DATA SAMPLING:
// This function supports flexible data sampling controlled by config.SamplesPerClass:
// - 0: Use all available images (full dataset, potentially 1M+ images)
// - N > 0: Use exactly N randomly selected images per class (balanced sampling)
// This dramatically speeds up development and experimentation while maintaining
// statistical validity.
func (ic *ImageClassifier) loadTrainingData() ([]ImageData, error) {
	var allData []ImageData

	// Initialize random seed for sampling consistency within a session
	// but variability across different program runs
	rand.Seed(time.Now().UnixNano())

	// Define the character groups we want to process
	// Each group corresponds to a subdirectory and a range of characters
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
		// Not all datasets might have all groups. We gracefully skip missing
		// directories rather than failing completely, which allows for partial
		// datasets during development.
		if _, err := os.Stat(subdirPath); os.IsNotExist(err) {
			fmt.Printf("Warning: Directory %s does not exist, skipping...\n", subdirPath)
			continue
		}

		// Process each character in this group's range
		// Go's character arithmetic lets us iterate through character ranges
		// cleanly without hardcoding every single character
		for char := group.StartChar; char <= group.EndChar; char++ {
			// Construct path to this specific character's directory
			charDir := filepath.Join(subdirPath, string(char))

			// Check if this character's directory exists
			// Some datasets might be incomplete (missing certain characters)
			if _, err := os.Stat(charDir); os.IsNotExist(err) {
				fmt.Printf("Warning: Directory %s does not exist, skipping...\n", charDir)
				continue
			}

			// Load images for this character (with sampling if configured)
			charData, err := ic.loadImagesFromDirectory(charDir, string(char))
			if err != nil {
				return nil, fmt.Errorf("failed to load images from %s: %w", charDir, err)
			}

			// Add this character's data to our growing collection
			allData = append(allData, charData...)
			
			// Display sampling information to keep user informed
			if ic.config.SamplesPerClass > 0 {
				fmt.Printf("Loaded %d images for class '%s' (sampled from available images)\n", 
					len(charData), string(char))
			} else {
				fmt.Printf("Loaded %d images for class '%s' (all available)\n", 
					len(charData), string(char))
			}
		}
	}

	// PUNCTUATION HANDLING:
	// Punctuation marks require special handling because:
	// 1. Some characters can't be used as directory names (/, ?, *, etc.)
	// 2. We use descriptive names like "asterisk" instead of "*"
	// 3. We need to map directory names back to actual characters
	// 4. Punctuation doesn't form a continuous range like letters or digits
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
			// Convert descriptive directory name (like "asterisk") to actual character ("*")
			actualChar, exists := PunctuationDirToChar[dirName]
			if !exists {
				fmt.Printf("Warning: Unknown punctuation directory '%s', skipping...\n", dirName)
				continue
			}

			// Construct path to this punctuation character's directory
			punctCharDir := filepath.Join(punctuationDir, dirName)

			// Load images for this punctuation character (with sampling if configured)
			punctData, err := ic.loadImagesFromDirectory(punctCharDir, actualChar)
			if err != nil {
				fmt.Printf("Warning: Failed to load images from %s: %v\n", punctCharDir, err)
				continue // Continue with other punctuation marks instead of failing completely
			}

			// Add this punctuation character's data to our collection
			allData = append(allData, punctData...)
			
			// Display sampling information for punctuation
			if ic.config.SamplesPerClass > 0 {
				fmt.Printf("Loaded %d images for punctuation class '%s' (from dir '%s', sampled)\n", 
					len(punctData), actualChar, dirName)
			} else {
				fmt.Printf("Loaded %d images for punctuation class '%s' (from dir '%s', all available)\n", 
					len(punctData), actualChar, dirName)
			}
		}
	} else {
		fmt.Printf("Warning: Punctuation directory %s does not exist, skipping...\n", punctuationDir)
	}

	// VALIDATION: Ensure we actually loaded some data
	// Training with zero examples would be impossible, so we catch this error early
	// and provide a helpful error message
	if len(allData) == 0 {
		return nil, fmt.Errorf("no training data found in %s", ic.config.DataDir)
	}

	// Display comprehensive dataset statistics
	fmt.Printf("\n=== DATASET STATISTICS ===\n")
	fmt.Printf("Total training samples loaded: %d\n", len(allData))
	
	if ic.config.SamplesPerClass > 0 {
		expectedTotal := ic.config.SamplesPerClass * ic.config.OutputSize
		fmt.Printf("Sampling configuration: %d samples per class\n", ic.config.SamplesPerClass)
		fmt.Printf("Expected total samples: %d (for %d classes)\n", expectedTotal, ic.config.OutputSize)
		
		if len(allData) < expectedTotal {
			fmt.Printf("⚠ Warning: Loaded fewer samples than expected\n")
			fmt.Printf("  This may indicate missing directories or insufficient images per class\n")
		} else {
			fmt.Printf("✓ Successfully loaded balanced dataset with sampling\n")
		}
	} else {
		fmt.Printf("Using full dataset (no sampling applied)\n")
		fmt.Printf("✓ Successfully loaded complete dataset\n")
	}
	fmt.Printf("=============================\n\n")
	
	return allData, nil
}

// loadImagesFromDirectory loads images from a specific directory with optional sampling.
//
// INTELLIGENT SAMPLING:
// This function supports configurable data sampling that dramatically improves
// development efficiency:
// - When SamplesPerClass = 0: Load all available images (original behavior)
// - When SamplesPerClass > 0: Randomly sample exactly N images per class
//
// SAMPLING STRATEGY:
// 1. First, discover all valid image files in the directory
// 2. If sampling is disabled or we have fewer images than requested, use all available
// 3. If sampling is enabled and we have more images than needed, randomly select subset
// 4. Random selection ensures diversity within the sampled subset
//
// BENEFITS OF SAMPLING:
// - Faster training during development and experimentation (10x-100x speedup)
// - Balanced datasets (equal number of examples per class)
// - Ability to work with subsets for resource-constrained environments
// - Progressive development workflow (start small, scale up gradually)
//
// ERROR HANDLING PHILOSOPHY:
// We use a "resilient" approach to individual file errors - if one image fails to
// load (corrupted file, wrong format, permission issues), we log a warning but
// continue processing other images. This prevents single bad files from destroying
// entire training runs.
//
// UPDATED FOR PUNCTUATION:
// This function now handles any character label including punctuation marks.
// The label parameter can be a letter ("A"), digit ("7"), or punctuation ("*").
func (ic *ImageClassifier) loadImagesFromDirectory(dir, label string) ([]ImageData, error) {
	// STEP 1: Discover all image files in the directory
	var imagePaths []string

	// RECURSIVE DIRECTORY WALKING:
	// filepath.Walk recursively visits every file and subdirectory, which is more
	// robust than manual directory reading because it handles nested subdirectories
	// and provides consistent cross-platform behavior.
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		// Handle walk errors (permissions, broken symlinks, etc.)
		if err != nil {
			return err
		}

		// Skip directories - we only want actual files
		if info.IsDir() {
			return nil
		}

		// FILTER BY FILE EXTENSION:
		// Only process files that are likely to be images. This prevents us from
		// trying to process README files, .DS_Store files, temporary files, etc.
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
			return nil // Skip non-image files silently
		}

		// Add this image path to our list of candidates for processing
		imagePaths = append(imagePaths, path)
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", dir, err)
	}

	// STEP 2: Apply sampling strategy if configured
	var selectedPaths []string
	
	if ic.config.SamplesPerClass <= 0 {
		// NO SAMPLING: Use all discovered images (original behavior)
		selectedPaths = imagePaths
	} else if len(imagePaths) <= ic.config.SamplesPerClass {
		// INSUFFICIENT IMAGES: Use all available (can't sample more than we have)
		selectedPaths = imagePaths
		if len(imagePaths) < ic.config.SamplesPerClass {
			fmt.Printf("  Note: Class '%s' has only %d images (requested %d)\n", 
				label, len(imagePaths), ic.config.SamplesPerClass)
		}
	} else {
		// SAMPLING ENABLED: Randomly select the requested number of images
		selectedPaths = sampleImagePaths(imagePaths, ic.config.SamplesPerClass)
	}

	// STEP 3: Load and process the selected images
	var images []ImageData

	for _, path := range selectedPaths {
		// ATTEMPT TO LOAD AND PROCESS THE IMAGE:
		// This is where the image gets converted from a file on disk to numerical
		// data that the neural network can work with
		pixels, err := ic.loadAndProcessImage(path)
		if err != nil {
			// RESILIENT ERROR HANDLING:
			// Log the problem but don't stop processing. A few bad images shouldn't
			// prevent training on thousands of good images. This is especially
			// important when dealing with large datasets that might contain some
			// corrupted files.
			fmt.Printf("Warning: Failed to load image %s: %v\n", path, err)
			continue // Continue with other images
		}

		// CONVERT LABEL TO CLASS INDEX:
		// Neural networks work with numbers, not strings. We convert the string
		// label (like "A", "*", "?") to a numerical index using our predefined
		// mapping system.
		classIndex, exists := ClassMapping[label]
		if !exists {
			return nil, fmt.Errorf("unknown class label: %s", label)
		}

		// CREATE THE TRAINING EXAMPLE:
		// Combine the numerical pixel data with the label information to create
		// a complete training example
		images = append(images, ImageData{
			Pixels:     pixels,     // The actual image data as 784 floating-point numbers
			Label:      label,      // Human-readable label ("A", "b", "7", "*")
			ClassIndex: classIndex, // Numerical index for the neural network (0-93)
		})
	}

	return images, nil
}

// sampleImagePaths randomly selects a subset of image paths.
//
// RANDOM SAMPLING IMPLEMENTATION:
// This function implements a partial Fisher-Yates shuffle algorithm to randomly
// select N items from a larger collection. This approach ensures:
// - Each image has an equal probability of being selected
// - No duplicate selections (can't select the same image twice)
// - Efficient O(N) performance (only shuffles as much as needed)
//
// WHY RANDOM SAMPLING?
// Random selection helps maintain the statistical properties of the original
// dataset even with smaller sample sizes. This is much better than simply
// taking the first N images, which might introduce bias based on:
// - File naming conventions (alphabetical ordering)
// - Creation dates (chronological ordering)
// - File system organization
//
// SAMPLING DIVERSITY:
// By randomly selecting images, we ensure that our training subset contains
// diverse examples rather than potentially similar images that might be
// grouped together by filename, creation date, or other non-random factors.
func sampleImagePaths(paths []string, sampleSize int) []string {
	if sampleSize <= 0 || sampleSize >= len(paths) {
		return paths // Return all if sample size is invalid or larger than available
	}

	// Create a copy of the paths slice to avoid modifying the original
	pathsCopy := make([]string, len(paths))
	copy(pathsCopy, paths)

	// PARTIAL FISHER-YATES SHUFFLE ALGORITHM:
	// We only need to shuffle the first 'sampleSize' elements, not the entire array.
	// This is more efficient than shuffling everything and then taking the first N.
	for i := 0; i < sampleSize; i++ {
		// Pick a random index from the remaining unshuffled elements
		j := i + rand.Intn(len(pathsCopy)-i)
		
		// Swap the current element with the randomly selected element
		pathsCopy[i], pathsCopy[j] = pathsCopy[j], pathsCopy[i]
	}

	// Return the first 'sampleSize' elements (which are now randomly selected)
	return pathsCopy[:sampleSize]
}

// prepareDataForTraining converts ImageData to the format needed by the neural network.
//
// FROM HUMAN FORMAT TO NEURAL NETWORK FORMAT:
// This function handles the final conversion step - taking our collected ImageData
// structs and converting them into the matrix format that the neural network library
// expects. This is where individual examples become part of a batch that can be
// processed efficiently by the mathematical operations in the network.
//
// MATRIX FORMAT REQUIREMENTS:
// Neural networks work with matrices (2D arrays) where:
// - Each row represents one training example (one image and its label)
// - Each column in the input matrix represents one feature (one pixel value)
// - Each column in the output matrix represents one possible class
// - Labels become "one-hot encoded" vectors (explained below)
//
// ONE-HOT ENCODING EXPLAINED:
// Instead of representing the label "A" as the number 0, we represent it as a vector
// like [1, 0, 0, 0, 0, ...] where only the position corresponding to "A" contains
// a 1, and all other positions contain 0. This format works better with the softmax
// output layer that produces probability distributions.
//
// EXAMPLE FOR 94 CLASSES:
// If "A" has ClassIndex 0: [1, 0, 0, 0, ..., 0] (94 elements, only first is 1)
// If "*" has ClassIndex 62: [0, 0, ..., 1, 0, ..., 0] (94 elements, only 63rd is 1)
//
// UPDATED FOR 94 CLASSES:
// This function now handles 94 different character classes instead of the original
// 62, including all punctuation marks in addition to letters and digits.
//
// SAMPLING IMPACT:
// When sampling is enabled, this function processes fewer ImageData instances,
// resulting in smaller input/output matrices and dramatically faster training times.
func (ic *ImageClassifier) prepareDataForTraining(data []ImageData) ([][]float64, [][]float64, error) {
	numSamples := len(data)

	// ALLOCATE OUTPUT MATRICES:
	// Pre-allocate the slices to avoid repeated memory allocations during the loop,
	// which improves performance especially with large datasets
	inputs := make([][]float64, numSamples)   // Each row is pixel values for one image
	outputs := make([][]float64, numSamples)  // Each row is one-hot encoded label

	// CONVERT EACH TRAINING EXAMPLE:
	for i, sample := range data {
		// INPUT PREPARATION:
		// The pixel data is already in the right format (784 float64 values),
		// so we can directly assign it
		inputs[i] = sample.Pixels

		// OUTPUT PREPARATION (ONE-HOT ENCODING):
		// Create a vector of all zeros, then set the correct class position to 1.0
		oneHot := make([]float64, ic.config.OutputSize) // All zeros initially (94 elements)
		oneHot[sample.ClassIndex] = 1.0                 // Set correct class to 1.0
		outputs[i] = oneHot

		// EXAMPLE OF ONE-HOT ENCODING:
		// If sample is "A" (ClassIndex = 0):
		//   oneHot = [1.0, 0.0, 0.0, 0.0, ..., 0.0] (1.0 at position 0)
		// If sample is "a" (ClassIndex = 26):
		//   oneHot = [0.0, 0.0, ..., 1.0, 0.0, ..., 0.0] (1.0 at position 26)
		// If sample is "*" (ClassIndex = 62):
		//   oneHot = [0.0, 0.0, ..., 1.0, 0.0, ..., 0.0] (1.0 at position 62)
	}

	return inputs, outputs, nil
}

// DATA LOADING BEST PRACTICES DEMONSTRATED:

// 1. ROBUST FILE SYSTEM HANDLING:
// - Check if directories exist before trying to read them
// - Handle file system errors gracefully without crashing
// - Use cross-platform path manipulation (filepath.Join instead of string concatenation)
// - Support different image formats (PNG, JPEG) commonly used in datasets

// 2. RESILIENT ERROR HANDLING:
// - Individual file failures don't stop the entire loading process
// - Clear warning messages help identify data quality issues early
// - Early validation prevents training with empty or insufficient datasets
// - Meaningful error messages aid in troubleshooting

// 3. EFFICIENT MEMORY USAGE:
// - Pre-allocate slices when final size is known (avoids repeated allocations)
// - Process files one at a time rather than loading everything into memory first
// - Use appropriate data structures for the specific task
// - Sampling reduces memory usage by processing fewer images

// 4. CLEAR SEPARATION OF CONCERNS:
// - loadTrainingData: Orchestrates the overall data loading process
// - loadImagesFromDirectory: Handles a single directory with sampling
// - sampleImagePaths: Implements the random sampling algorithm
// - prepareDataForTraining: Converts to neural network format
// Each function has a single, clear responsibility making code easier to maintain

// 5. USER-FRIENDLY FEEDBACK:
// - Progress messages show what's happening during potentially long operations
// - Warning messages help identify data issues without stopping execution
// - Meaningful error messages aid in debugging and problem resolution
// - Comprehensive statistics provide transparency about dataset characteristics

// 6. EXTENSIBLE DESIGN:
// - Easy to add new character types by extending the directory structure
// - Punctuation handling demonstrates how to accommodate special cases
// - Sampling feature is additive and doesn't break existing functionality
// - Maintains backward compatibility with existing data organization

// DATA QUALITY CONSIDERATIONS FOR PUNCTUATION WITH SAMPLING:

// 1. BALANCED DATASETS:
// Sampling ensures equal representation across all character classes, which is
// especially important for punctuation marks that might be less common than
// letters in typical text datasets.

// 2. VISUALLY SIMILAR CHARACTERS:
// When sampling, ensure you still have enough examples of visually similar
// characters (like "." and "," or "O" and "0") to maintain the ability to
// distinguish between them effectively.

// 3. FONT VARIATIONS:
// Random sampling helps ensure font variety is preserved even in smaller datasets,
// which is crucial for punctuation marks that can vary significantly across
// different fonts and typographic styles.

// SAMPLING STRATEGY RECOMMENDATIONS:

// 1. DEVELOPMENT WORKFLOW:
// - Start with 10-50 samples per class for rapid prototyping and architecture testing
// - Scale to 100-500 samples for hyperparameter tuning and development
// - Use 1000+ samples for validation and pre-production testing  
// - Train final production model with full dataset (samples=0)

// 2. RESOURCE MANAGEMENT:
// - Use sampling when working on resource-constrained development systems
// - Smaller datasets reduce memory usage and training time dramatically
// - Maintain balanced representation across all 94 character classes
// - Enable rapid iteration during development phase

// 3. QUALITY ASSURANCE:
// - Monitor class balance in sampled datasets to ensure fairness
// - Ensure sufficient examples for visually similar character pairs
// - Validate that sampling doesn't introduce systematic bias
// - Test across different sample sizes to understand accuracy trade-offs

// PERFORMANCE IMPACT OF SAMPLING:

// TRAINING TIME SCALING (approximate):
// - 10 samples/class: ~99% time reduction (940 total images vs 1.3M+)
// - 100 samples/class: ~95% time reduction (9,400 total images)
// - 1000 samples/class: ~85% time reduction (94,000 total images)
// - Full dataset: Baseline performance (1,300,000+ total images)

// MEMORY USAGE SCALING:
// Memory usage scales proportionally with dataset size. Smaller samples enable
// training on lower-memory systems and make development possible on laptops
// or other resource-constrained environments.

// ACCURACY CONSIDERATIONS:
// - Smaller samples may have lower final accuracy but enable rapid iteration
// - Random sampling preserves dataset characteristics better than systematic sampling
// - Final production models should use full datasets when computational resources allow
// - Balance development speed with accuracy requirements based on project phase