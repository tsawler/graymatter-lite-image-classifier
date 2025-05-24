package main

import "github.com/tsawler/graymatter-lite"

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
			SavePath:       "image_classifier",
			SaveInterval:   100,
			EnablePlotting: true,
			PlottingURL:    "http://localhost:8080",
		},
	}
}