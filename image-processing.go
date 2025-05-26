package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
)

// loadAndProcessImage loads an image file and converts it to normalized pixel values.
// This is the ORIGINAL function used during training - maintains strict consistency.
//
// TRAINING DATA REQUIREMENTS:
// This function is used for loading training data, which must meet strict requirements:
// - Images must be exactly 28×28 pixels (no resizing performed)
// - All preprocessing must be identical across all training images
// - Any image that doesn't meet requirements causes an error
//
// This strict approach ensures training data consistency, which is critical for
// neural network performance. The network learns to expect input in exactly this format.
func (ic *ImageClassifier) loadAndProcessImage(imagePath string) ([]float64, error) {
	// Open the image file
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	// Decode the image based on file extension
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

	// Convert image to pixel array
	pixels := ic.imageToPixels(img)

	// Strict validation: training images must be exactly the right size
	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("image has %d pixels, expected %d (ensure images are %dx%d)",
			len(pixels), ic.config.InputSize, ic.config.ImageWidth, ic.config.ImageHeight)
	}

	return pixels, nil
}

// loadAndProcessImageForPrediction loads ANY size/type image and converts it to 28x28 grayscale for prediction.
//
// PREDICTION-SPECIFIC PREPROCESSING:
// This function is specifically designed for prediction use cases where users might
// provide images of any size, format, or quality. It automatically handles:
// 1. Loading images in common formats (PNG, JPEG)
// 2. Intelligent resizing to exactly 28×28 while preserving aspect ratio
// 3. Converting color images to grayscale using perceptual weighting
// 4. Normalizing pixel values to the 0.0-1.0 range expected by the network
// 5. Flattening the 2D image into the 1D array format the network expects
//
// WHY SEPARATE FROM TRAINING PIPELINE?
// - Training images should remain strictly controlled (exactly 28×28) for consistency
// - Prediction should be user-friendly and accept any reasonable input format
// - This separation keeps training data requirements clear while making prediction practical
// - Different error handling: training fails on bad images, prediction tries to adapt
func (ic *ImageClassifier) loadAndProcessImageForPrediction(imagePath string) ([]float64, error) {
	// STEP 1: Open and decode the image (same as training function)
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

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

	// STEP 2: Resize to target dimensions (this is the key difference!)
	// This step handles images of any size and intelligently resizes them
	// to the 28×28 format the network expects
	resizedImg := ic.resizeImageToTarget(img, ic.config.ImageWidth, ic.config.ImageHeight)

	// STEP 3: Convert to pixels using existing logic
	// Once resized, we use the same pixel conversion as training to ensure consistency
	pixels := ic.imageToPixels(resizedImg)

	// STEP 4: Validate final dimensions (should always be correct now)
	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("internal error: processed image has %d pixels, expected %d",
			len(pixels), ic.config.InputSize)
	}

	return pixels, nil
}

// resizeImageToTarget resizes an image to target dimensions while preserving aspect ratio.
//
// SMART RESIZING STRATEGY:
// This function implements intelligent image resizing that:
// - Calculates scale factors needed to fit image within target dimensions
// - Uses the smaller scale factor to ensure the entire image fits
// - Centers the scaled image within the target canvas
// - Fills remaining space with white background (typical for character images)
//
// ASPECT RATIO PRESERVATION:
// Rather than stretching images (which would distort characters), we maintain
// the original proportions and add padding as needed. This ensures characters
// don't appear unnaturally wide or tall.
//
// EXAMPLE: 100×50 image → 28×28 target
// - Scale factors: 28/100=0.28 (width), 28/50=0.56 (height)  
// - Use smaller factor (0.28), giving final scaled size of 28×14
// - Center the 28×14 image within 28×28 canvas
// - Fill top and bottom with white background (7 pixels each)
func (ic *ImageClassifier) resizeImageToTarget(src image.Image, targetWidth, targetHeight int) image.Image {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	// If already correct size, return as-is (optimization)
	if srcWidth == targetWidth && srcHeight == targetHeight {
		return src
	}

	// Calculate scale factor to preserve aspect ratio
	// We need to fit the image within the target dimensions, so we use the
	// smaller of the two scale factors to ensure nothing gets cut off
	scaleX := float64(targetWidth) / float64(srcWidth)
	scaleY := float64(targetHeight) / float64(srcHeight)
	scale := min(scaleY, scaleX)

	// Calculate scaled dimensions (will be <= target dimensions)
	scaledWidth := int(float64(srcWidth) * scale)
	scaledHeight := int(float64(srcHeight) * scale)

	// Create scaled image using nearest neighbor interpolation
	// This is simple but effective for the relatively small target size
	scaledImg := image.NewRGBA(image.Rect(0, 0, scaledWidth, scaledHeight))
	
	// Manual scaling using nearest neighbor interpolation
	// More sophisticated interpolation methods exist, but nearest neighbor
	// is sufficient for our purposes and avoids additional dependencies
	for y := range scaledHeight {
		for x := range scaledWidth {
			// Map scaled coordinates back to source coordinates
			srcX := int(float64(x) / scale)
			srcY := int(float64(y) / scale)
			
			// Ensure we don't go out of bounds (edge case handling)
			if srcX >= srcWidth {
				srcX = srcWidth - 1
			}
			if srcY >= srcHeight {
				srcY = srcHeight - 1
			}
			
			// Copy pixel from source to scaled image
			scaledImg.Set(x, y, src.At(srcBounds.Min.X+srcX, srcBounds.Min.Y+srcY))
		}
	}

	// Create final target-sized canvas with white background
	// White background is appropriate for character recognition since most
	// text is dark characters on light backgrounds
	finalImg := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	white := color.RGBA{255, 255, 255, 255}
	draw.Draw(finalImg, finalImg.Bounds(), &image.Uniform{white}, image.Point{}, draw.Src)

	// Center the scaled image within the target canvas
	offsetX := (targetWidth - scaledWidth) / 2
	offsetY := (targetHeight - scaledHeight) / 2
	
	// Copy the scaled image to the center of the white canvas
	draw.Draw(finalImg, 
		image.Rect(offsetX, offsetY, offsetX+scaledWidth, offsetY+scaledHeight),
		scaledImg, 
		image.Point{0, 0}, 
		draw.Over)

	return finalImg
}

// imageToPixels converts an image to normalized pixel values.
//
// PIXEL CONVERSION PROCESS:
// This function performs the critical conversion from visual image data to
// the numerical format that neural networks require:
// 1. Extract RGB color values for each pixel
// 2. Convert RGB to grayscale using perceptual luminance weights
// 3. Normalize values from 0-65535 range to 0.0-1.0 range
// 4. Flatten 2D pixel grid into 1D array
//
// GRAYSCALE CONVERSION:
// We use perceptually-weighted conversion rather than simple averaging because
// human vision is more sensitive to green light than red or blue. The weights
// (0.299 R + 0.587 G + 0.114 B) produce grayscale that looks natural and
// preserves important visual information for character recognition.
//
// NORMALIZATION:
// Neural networks work best with inputs in a consistent, small range. Raw pixel
// values (0-65535) are too large and varied. By normalizing to 0.0-1.0, we put
// all inputs in a range the network can handle effectively.
func (ic *ImageClassifier) imageToPixels(img image.Image) []float64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	
	// Pre-allocate the pixel array for efficiency
	pixels := make([]float64, width*height)

	// Process each pixel in the image
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// Get RGBA values for this pixel (16-bit color channels)
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert RGB to grayscale using perceptual luminance formula
			// This mimics how human vision perceives brightness
			gray := (RedLuminance*float64(r) + GreenLuminance*float64(g) + BlueLuminance*float64(b)) / RGBAMax

			// Calculate 1D index from 2D coordinates (row-major order)
			pixelIndex := (y-bounds.Min.Y)*width + (x-bounds.Min.X)
			pixels[pixelIndex] = gray
		}
	}

	return pixels
}

// analyzeImagePolarity determines if an image is predominantly white on black or black on white.
//
// POLARITY DETECTION:
// Different image sources may have different polarities:
// - Scanned documents: usually black text on white background
// - Screenshots from dark themes: white text on black background
// - Artistic images: could be either
//
// We sample pixels from corners and center to determine the background color,
// then return true for black-on-white (light background), false for white-on-black.
//
// WHY THIS MATTERS:
// The neural network was likely trained on images with a consistent polarity.
// If we detect the wrong polarity, we can invert the image to match the training data.
//
// Returns true for black characters on white background, false for white characters on black background.
func (ic *ImageClassifier) analyzeImagePolarity(pixels []float64) bool {
	// Sample pixels from corners and center to determine background color
	// This heuristic assumes the background is relatively uniform and occupies
	// most of the corner and center areas
	width := ic.config.ImageWidth
	height := ic.config.ImageHeight
	
	// Define sample positions: corners and center
	sampleIndices := []int{
		0,                       // Top-left corner
		width - 1,               // Top-right corner
		width * (height - 1),    // Bottom-left corner
		width * height - 1,      // Bottom-right corner
		(width*height)/2,        // Center of image
	}

	// Calculate average brightness of sampled areas
	var sumBrightness float64
	validSamples := 0
	for _, idx := range sampleIndices {
		if idx >= 0 && idx < len(pixels) { // Ensure index is within bounds
			sumBrightness += pixels[idx]
			validSamples++
		}
	}
	
	if validSamples == 0 {
		// Fallback: assume black on white if we can't sample
		return true
	}
	
	avgBrightness := sumBrightness / float64(validSamples)

	// If average brightness of sampled areas is high (closer to 1.0), 
	// it's likely a white/light background with dark characters
	// We use 0.5 as threshold: >0.5 = light background, ≤0.5 = dark background
	return avgBrightness > 0.5 
}

// invertPixels inverts the grayscale values of the image pixels.
//
// PIXEL INVERSION:
// This function flips the brightness of every pixel:
// - 0.0 (black) becomes 1.0 (white)
// - 1.0 (white) becomes 0.0 (black) 
// - 0.5 (gray) stays 0.5 (gray)
//
// WHEN TO USE:
// If polarity analysis detects that an image has the opposite polarity from
// what the network expects, we can invert it to match the training data format.
func (ic *ImageClassifier) invertPixels(pixels []float64) []float64 {
	inverted := make([]float64, len(pixels))
	for i, p := range pixels {
		inverted[i] = 1.0 - p // Flip: 0.0 ↔ 1.0, 0.3 → 0.7, etc.
	}
	return inverted
}

// saveProcessedImage saves the processed pixel data as a 28x28 grayscale PNG image.
//
// DEBUGGING AND VISUALIZATION:
// This function is invaluable for debugging preprocessing issues. It converts
// the numerical pixel data back into a visual image that humans can examine.
// This lets you see exactly what data the neural network receives as input.
//
// COMMON DEBUGGING USE CASES:
// - Verify that image resizing worked correctly
// - Check that grayscale conversion looks reasonable
// - Confirm that polarity inversion was applied correctly
// - Ensure normalization didn't cause unexpected artifacts
func (ic *ImageClassifier) saveProcessedImage(pixels []float64, filename string) error {
	// Create a new grayscale image with the target dimensions
	img := image.NewGray(image.Rect(0, 0, ic.config.ImageWidth, ic.config.ImageHeight))

	// Convert each normalized pixel value back to 8-bit grayscale
	for y := range ic.config.ImageHeight {
		for x := range ic.config.ImageWidth {
			pixelIndex := y*ic.config.ImageWidth + x
			if pixelIndex < len(pixels) {
				// Convert normalized float (0.0-1.0) back to uint8 (0-255) for image format
				grayVal := uint8(pixels[pixelIndex] * 255)
				img.SetGray(x, y, color.Gray{Y: grayVal})
			}
		}
	}

	// Save as PNG file
	outFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output image file: %w", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, img)
	if err != nil {
		return fmt.Errorf("failed to encode PNG image: %w", err)
	}

	return nil
}

// IMAGE PROCESSING BEST PRACTICES DEMONSTRATED:

// 1. PREPROCESSING CONSISTENCY:
// The most critical aspect of image processing for machine learning is consistency.
// Training and prediction must use identical preprocessing steps, or the model
// will fail even if it was trained perfectly.

// 2. FLEXIBLE INPUT HANDLING:
// For prediction, we support various input formats and sizes while standardizing
// the output format. This makes the system user-friendly while maintaining
// the strict requirements needed for neural network input.

// 3. ASPECT RATIO PRESERVATION:
// When resizing images, we preserve aspect ratios to avoid character distortion.
// Stretching characters would make them unrecognizable to a network trained
// on properly proportioned characters.

// 4. PERCEPTUAL GRAYSCALE CONVERSION:
// We use weighted conversion that matches human vision rather than simple
// averaging. This preserves important visual information and produces images
// that look natural to human observers.

// 5. DEBUGGING SUPPORT:
// The ability to save processed images is crucial for debugging preprocessing
// issues. Visual inspection often reveals problems that are hard to detect
// from numerical data alone.

// 6. POLARITY HANDLING:
// Real-world images may have different polarities (dark-on-light vs light-on-dark).
// Detecting and correcting polarity mismatches improves robustness across
// different image sources.

// COMMON IMAGE PROCESSING PITFALLS:

// 1. INCONSISTENT PREPROCESSING:
// Using different preprocessing for training vs prediction is the most common
// cause of poor model performance in production.

// 2. POOR RESIZE ALGORITHMS:
// Simple scaling without aspect ratio consideration can distort characters
// beyond recognition.

// 3. INCORRECT NORMALIZATION:
// Getting the normalization range wrong (e.g., 0-255 instead of 0.0-1.0)
// will cause complete model failure.

// 4. IGNORING COLOR SPACE:
// Naive RGB-to-grayscale conversion can lose important visual information
// that affects character recognition accuracy.

// 5. MISSING EDGE CASES:
// Not handling unusual input sizes, formats, or corrupted images can cause
// production systems to crash unexpectedly.

// This image processing module provides robust, production-ready preprocessing
// that handles real-world variability while maintaining the strict consistency
// required for neural network input. The separation between training and
// prediction preprocessing allows for both rigorous training requirements
// and user-friendly prediction interfaces.