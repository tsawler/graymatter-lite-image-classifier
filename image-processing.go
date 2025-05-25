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
// 4. SAVES the processed image for inspection/debugging
// 5. Returns the same format expected by the neural network
//
// WHY SAVE THE PROCESSED IMAGE?
// Saving the processed image is invaluable for debugging:
// - Verify preprocessing is working correctly
// - See exactly what the neural network "sees"
// - Diagnose prediction issues
// - Compare processed images across different inputs
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

	// STEP 3: Save the processed image for inspection
	// This allows you to see exactly what the neural network is analyzing
	if err := ic.saveProcessedImage(resizedImg, "image_to_predict.png"); err != nil {
		// Log warning but don't fail the prediction if saving fails
		fmt.Printf("Warning: Failed to save processed image: %v\n", err)
	} else {
		fmt.Println("Saved processed image as: image_to_predict.png")
	}

	// STEP 4: Convert to pixels using existing logic
	pixels := ic.imageToPixels(resizedImg)

	// STEP 5: Validate final dimensions (should always be correct now)
	if len(pixels) != ic.config.InputSize {
		return nil, fmt.Errorf("internal error: processed image has %d pixels, expected %d",
			len(pixels), ic.config.InputSize)
	}

	return pixels, nil
}

// saveProcessedImage saves the processed image to the file system.
//
// WHY SAVE THE PROCESSED IMAGE?
// This is incredibly useful for debugging and understanding your model:
// 1. VERIFICATION: See exactly what the neural network analyzes
// 2. DEBUGGING: Compare processed vs original to check preprocessing
// 3. DIAGNOSIS: If predictions are wrong, inspect the processed image
// 4. QUALITY CONTROL: Ensure image quality is sufficient for recognition
//
// THE PROCESSED IMAGE SHOWS:
// - 28×28 pixels (network input size)
// - Grayscale conversion result
// - Aspect ratio preservation and centering
// - White background padding if needed
//
// TROUBLESHOOTING WITH SAVED IMAGES:
// - Blurry image → original might be too small or low quality
// - Wrong colors → grayscale conversion issues
// - Poor centering → resizing algorithm issues
// - Unexpected content → wrong input file
func (ic *ImageClassifier) saveProcessedImage(img image.Image, filename string) error {
	// Create the output file
	outFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file %s: %w", filename, err)
	}
	defer outFile.Close()

	// Save as PNG for best quality (lossless compression)
	// PNG is ideal for grayscale images with text/characters
	if err := png.Encode(outFile, img); err != nil {
		return fmt.Errorf("failed to encode image as PNG: %w", err)
	}

	return nil
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

// BENEFITS OF SAVING PROCESSED IMAGES:

// 1. VISUAL DEBUGGING:
// You can open "image_to_predict.png" and see exactly what your neural
// network is analyzing. This is invaluable when predictions seem wrong.

// 2. PREPROCESSING VERIFICATION:
// Compare the saved image to your original to ensure:
// - Proper grayscale conversion
// - Correct resizing and aspect ratio preservation
// - Appropriate centering and padding

// 3. QUALITY ASSESSMENT:
// Check if the processed image has sufficient quality for recognition:
// - Is text/character clear and readable?
// - Is the image too blurry or pixelated?
// - Does the character fill an appropriate amount of the 28x28 space?

// 4. TRAINING DATA COMPARISON:
// Compare processed prediction images to your training data to ensure
// they have similar characteristics (brightness, size, positioning).

// 5. TROUBLESHOOTING WORKFLOW:
// When a prediction is wrong:
// 1. Look at "image_to_predict.png"
// 2. Compare to training examples of the predicted vs actual character
// 3. Identify if the issue is preprocessing, model, or input quality

// EXAMPLE USAGE OUTPUT:
// When you run prediction, you'll see:
// "Saved processed image as: image_to_predict.png"
// "Predicted character: 'A'"
// "Confidence: 95.2%"
//
// Then you can examine image_to_predict.png to see the 28x28 grayscale
// image that produced this prediction.