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