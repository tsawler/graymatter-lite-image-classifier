package main

import "github.com/tsawler/graymatter-lite"

// ClassGroup represents a group of characters with their directory info
type ClassGroup struct {
	DirName              string
	StartChar, EndChar   rune
}

// ImageData represents a single training example
type ImageData struct {
	Pixels     []float64
	Label      string
	ClassIndex int
}

// ImageClassifier encapsulates the training and prediction logic
type ImageClassifier struct {
	config  *Config
	network *graymatter.Network
}

// NewImageClassifier creates a new image classifier
func NewImageClassifier(config *Config) *ImageClassifier {
	return &ImageClassifier{config: config}
}

// LoadModel loads a pre-trained model from a file
func LoadModel(filename string) (*ImageClassifier, error) {
	network, _, err := graymatter.LoadNetwork(filename)
	if err != nil {
		return nil, err
	}
	
	// Create a default config - in production you'd want to save/load config too
	config := NewDefaultConfig()
	
	return &ImageClassifier{
		config:  config,
		network: network,
	}, nil
}