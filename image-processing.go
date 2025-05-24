package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
)

// loadAndProcessImage loads an image file and converts it to normalized pixel values.
//
// FROM HUMAN-READABLE IMAGES TO MACHINE-READABLE NUMBERS:
// This function bridges the gap between how humans see images (as visual patterns)
// and how neural networks process them (as arrays of numbers). Every image,
// whether it's a photo, drawing, or scanned document, gets converted into
// a consistent numerical format that the network can understand.
//
// THE PREPROCESSING PIPELINE:
// 1. Load image file from disk (PNG or JPEG)
// 2. Decode image data into pixel values
// 3. Convert color pixels to grayscale (if needed)
// 4. Normalize pixel values to 0.0-1.0 range
// 5. Flatten 2D image into 1D array for neural network input
//
// WHY PREPROCESSING MATTERS:
// Neural networks are very sensitive to input format. If training images are
// processed one way but prediction images are processed differently, the
// network will fail even if it was trained perfectly. Consistent preprocessing
// is absolutely critical for reliable results.
func (ic *ImageClassifier) loadAndProcessImage(imagePath string) ([]float64, error) {
	// STEP 1: Open the image file
	// This creates a file handle but doesn't read the entire file into memory yet
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close() // Always close files to prevent resource leaks

	// STEP 2: Determine image format and decode accordingly
	// Different image formats (PNG, JPEG) require different decoders
	// We determine the format from the file extension for simplicity
	var img image.Image
	ext := strings.ToLower(filepath.Ext(imagePath))

	switch ext {
	case ".png":
		// PNG ADVANTAGES:
		// - Lossless compression (no quality degradation)
		// - Supports transparency
		// - Good for simple graphics and text
		// - Larger file sizes than JPEG
		img, err = png.Decode(file)
	case ".jpg", ".jpeg":
		// JPEG ADVANTAGES:
		// - Excellent compression for photos
		// - Smaller file sizes
		// - Lossy compression (some quality loss)
		// - No transparency support
		img, err = jpeg.Decode(file)
	default:
		return nil, fmt.Errorf("unsupported image format: %s", ext)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// STEP 3: Convert image to our standard format
	// This handles color-to-grayscale conversion and flattening to 1D array
	pixels := ic.imageToPixels(img)

	// STEP 4: Validate the image dimensions
	// Neural networks expect a fixed input size. If the image doesn't match
	// our expected dimensions, something went wrong in preprocessing
	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("image has %d pixels, expected %d (ensure images are %dx%d)",
			len(pixels), ic.config.InputSize, ic.config.ImageWidth, ic.config.ImageHeight)
	}

	return pixels, nil
}

// imageToPixels converts an image to normalized pixel values.
//
// THE CONVERSION PROCESS:
// This function is where the "magic" happens - converting visual information
// into numbers that preserve the essential features for character recognition.
//
// KEY DECISIONS MADE HERE:
// 1. Convert color to grayscale (color rarely matters for character recognition)
// 2. Normalize to 0.0-1.0 range (neural networks work best with small numbers)
// 3. Flatten 2D image to 1D array (neural networks expect vector input)
//
// GRAYSCALE CONVERSION FORMULA:
// We use the standard luminance formula that mimics human vision:
// Gray = 0.299*Red + 0.587*Green + 0.114*Blue
//
// This isn't just an average - it weights colors by how bright they appear
// to human eyes. Green contributes most to perceived brightness, red moderately,
// and blue least. This produces more natural-looking grayscale images.
func (ic *ImageClassifier) imageToPixels(img image.Image) []float64 {
	// Get image dimensions
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	
	// Create output array to hold all pixel values
	// For a 28×28 image, this will be an array of 784 numbers
	pixels := make([]float64, width*height)

	// PIXEL PROCESSING LOOP:
	// We iterate through every pixel in the image, converting each one
	// from color to grayscale and normalizing the value
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// Get the color values for this pixel
			// Go's image package returns 16-bit values (0-65535) regardless
			// of the original image format
			r, g, b, _ := img.At(x, y).RGBA()

			// GRAYSCALE CONVERSION:
			// Apply the luminance formula to convert RGB to a single grayscale value
			// We divide by RGBAMax to normalize from 0-65535 range to 0.0-1.0 range
			//
			// WHY NORMALIZE?
			// Neural networks work best when inputs are in a consistent, small range.
			// Raw pixel values (0-65535) are too large and can cause numerical
			// instability during training. The 0.0-1.0 range is ideal:
			// - 0.0 represents black (no light)
			// - 1.0 represents white (maximum light)
			// - Values in between represent shades of gray
			gray := (RedLuminance*float64(r) + GreenLuminance*float64(g) + BlueLuminance*float64(b)) / RGBAMax

			// FLATTEN 2D TO 1D:
			// Neural networks expect 1D input, but images are 2D. We convert
			// 2D coordinates (x, y) to a 1D index using row-major order:
			// Row 0: pixels 0, 1, 2, ..., width-1
			// Row 1: pixels width, width+1, width+2, ..., 2*width-1
			// And so on...
			//
			// COORDINATE ADJUSTMENT:
			// We subtract bounds.Min because the image bounds might not start at (0,0)
			pixelIndex := (y-bounds.Min.Y)*width + (x-bounds.Min.X)
			pixels[pixelIndex] = gray
		}
	}

	return pixels
}

// IMAGE PROCESSING BEST PRACTICES DEMONSTRATED:

// 1. CONSISTENT PREPROCESSING:
// Every image goes through exactly the same processing steps, ensuring
// that training and prediction data are in identical formats.

// 2. ROBUST ERROR HANDLING:
// We check for file I/O errors, unsupported formats, and dimension mismatches.
// Clear error messages help debug problems quickly.

// 3. FORMAT FLEXIBILITY:
// Supporting both PNG and JPEG covers most use cases. The code is structured
// to make adding new formats easy (just add another case to the switch).

// 4. EFFICIENT MEMORY USAGE:
// We allocate the pixel array once and fill it in place, rather than using
// append() in a loop which would cause multiple memory allocations.

// 5. HUMAN-CENTRIC GRAYSCALE:
// Using the standard luminance formula produces grayscale images that look
// natural to humans and preserve important visual features for recognition.

// COMMON PITFALLS AVOIDED:

// 1. INCONSISTENT NORMALIZATION:
// Some implementations normalize to 0-255 range instead of 0.0-1.0, causing
// training/inference mismatches.

// 2. WRONG GRAYSCALE CONVERSION:
// Simple averaging (R+G+B)/3 doesn't account for human vision and can make
// some colors appear too bright or too dark.

// 3. DIMENSION ASSUMPTIONS:
// We validate that processed images match expected dimensions rather than
// assuming all input images are correctly sized.

// 4. RESOURCE LEAKS:
// We use defer to ensure files are always closed, even if errors occur.

// This preprocessing pipeline is critical for the success of the entire
// machine learning system. Getting it right ensures that the neural network
// receives clean, consistent input data that it can learn from effectively.