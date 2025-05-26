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
// This is the ORIGINAL function used during training - unchanged.
func (ic *ImageClassifier) loadAndProcessImage(imagePath string) ([]float64, error) {
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

	pixels := ic.imageToPixels(img)

	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("image has %d pixels, expected %d (ensure images are %dx%d)",
			len(pixels), ic.config.InputSize, ic.config.ImageWidth, ic.config.ImageHeight)
	}

	return pixels, nil
}

// loadAndProcessImageForPrediction loads ANY size/type image and converts it to 28x28 grayscale for prediction.
//
// PREDICTION-SPECIFIC PREPROCESSING:
// This function is specifically designed for prediction use cases where users
// might provide images of any size or format. It automatically:
// 1. Loads the image (any supported format)
// 2. Resizes to exactly 28x28 while preserving aspect ratio
// 3. Converts to grayscale and normalizes pixel values
// 4. Returns the same format expected by the neural network
//
// WHY SEPARATE FROM TRAINING PIPELINE?
// - Training images should remain strictly controlled (28x28) for consistency
// - Prediction should be user-friendly and accept any reasonable input
// - This separation keeps training data requirements clear while making prediction practical
func (ic *ImageClassifier) loadAndProcessImageForPrediction(imagePath string) ([]float64, error) {
	// STEP 1: Open and decode the image (same as training)
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

	// STEP 2: Resize to target dimensions (this is the new part!)
	resizedImg := ic.resizeImageToTarget(img, ic.config.ImageWidth, ic.config.ImageHeight)

	// STEP 3: Convert to pixels using existing logic
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
// - Calculate scale factor needed to fit image within target dimensions
// - Use the smaller scale factor to ensure entire image fits
// - Center the scaled image within target canvas
// - Fill remaining space with white background (typical for character images)
//
// EXAMPLE: 100x50 image → 28x28 target
// - Scale factors: 28/100=0.28 (width), 28/50=0.56 (height)  
// - Use 0.28 (smaller), giving 28x14 scaled image
// - Center within 28x28 canvas, fill top/bottom with white
func (ic *ImageClassifier) resizeImageToTarget(src image.Image, targetWidth, targetHeight int) image.Image {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	// If already correct size, return as-is
	if srcWidth == targetWidth && srcHeight == targetHeight {
		return src
	}

	// Calculate scale factor to preserve aspect ratio
	scaleX := float64(targetWidth) / float64(srcWidth)
	scaleY := float64(targetHeight) / float64(srcHeight)
	scale := scaleX
	if scaleY < scaleX {
		scale = scaleY
	}

	// Calculate scaled dimensions
	scaledWidth := int(float64(srcWidth) * scale)
	scaledHeight := int(float64(srcHeight) * scale)

	// Create scaled image
	scaledImg := image.NewRGBA(image.Rect(0, 0, scaledWidth, scaledHeight))
	
	// Manual scaling using nearest neighbor (simple but effective)
	for y := 0; y < scaledHeight; y++ {
		for x := 0; x < scaledWidth; x++ {
			// Map scaled coordinates back to source coordinates
			srcX := int(float64(x) / scale)
			srcY := int(float64(y) / scale)
			
			// Ensure we don't go out of bounds
			if srcX >= srcWidth {
				srcX = srcWidth - 1
			}
			if srcY >= srcHeight {
				srcY = srcHeight - 1
			}
			
			// Get pixel from source and set in scaled image
			scaledImg.Set(x, y, src.At(srcBounds.Min.X+srcX, srcBounds.Min.Y+srcY))
		}
	}

	// Create final target-sized canvas with white background
	finalImg := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	white := color.RGBA{255, 255, 255, 255}
	draw.Draw(finalImg, finalImg.Bounds(), &image.Uniform{white}, image.Point{}, draw.Src)

	// Center the scaled image
	offsetX := (targetWidth - scaledWidth) / 2
	offsetY := (targetHeight - scaledHeight) / 2
	
	draw.Draw(finalImg, 
		image.Rect(offsetX, offsetY, offsetX+scaledWidth, offsetY+scaledHeight),
		scaledImg, 
		image.Point{0, 0}, 
		draw.Over)

	return finalImg
}

// imageToPixels converts an image to normalized pixel values.
// (This function remains exactly the same - no changes needed)
func (ic *ImageClassifier) imageToPixels(img image.Image) []float64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	
	pixels := make([]float64, width*height)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			gray := (RedLuminance*float64(r) + GreenLuminance*float64(g) + BlueLuminance*float64(b)) / RGBAMax

			pixelIndex := (y-bounds.Min.Y)*width + (x-bounds.Min.X)
			pixels[pixelIndex] = gray
		}
	}

	return pixels
}

// analyzeImagePolarity determines if an image is predominantly white on black or black on white.
// Returns true for black on white (light background, dark foreground), false for white on black.
func (ic *ImageClassifier) analyzeImagePolarity(pixels []float64) bool {
	// Sample a few pixels from the corners and center to determine background color
	// This is a heuristic and assumes the background is relatively uniform.
	width := ic.config.ImageWidth
	height := ic.config.ImageHeight
	
	// Check corners and center
	sampleIndices := []int{
		0,                       // Top-left
		width - 1,               // Top-right
		width * (height - 1),    // Bottom-left
		width * height - 1,      // Bottom-right
		(width*height)/2,        // Center
	}

	var sumBrightness float64
	for _, idx := range sampleIndices {
		if idx >= 0 && idx < len(pixels) { // Ensure index is within bounds
			sumBrightness += pixels[idx]
		}
	}
	
	avgBrightness := sumBrightness / float64(len(sampleIndices))

	// If average brightness of sampled areas is high (closer to 1.0), it's likely a white background.
	// We use a threshold (e.g., 0.5) to decide.
	return avgBrightness > 0.5 
}

// invertPixels inverts the grayscale values of the image pixels.
// 0.0 (black) becomes 1.0 (white), and 1.0 (white) becomes 0.0 (black).
func (ic *ImageClassifier) invertPixels(pixels []float64) []float64 {
	inverted := make([]float64, len(pixels))
	for i, p := range pixels {
		inverted[i] = 1.0 - p
	}
	return inverted
}

// saveProcessedImage saves the processed pixel data as a 28x28 grayscale PNG image.
func (ic *ImageClassifier) saveProcessedImage(pixels []float64, filename string) error {
	img := image.NewGray(image.Rect(0, 0, ic.config.ImageWidth, ic.config.ImageHeight))

	for y := 0; y < ic.config.ImageHeight; y++ {
		for x := 0; x < ic.config.ImageWidth; x++ {
			pixelIndex := y*ic.config.ImageWidth + x
			if pixelIndex < len(pixels) {
				// Convert normalized float (0.0-1.0) back to uint8 (0-255) for grayscale
				grayVal := uint8(pixels[pixelIndex] * 255)
				img.SetGray(x, y, color.Gray{Y: grayVal})
			}
		}
	}

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