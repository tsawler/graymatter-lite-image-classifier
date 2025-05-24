package main

import (
	"fmt"
	"log"
)

func main() {
	fmt.Println("Starting Image Classification Training...")

	config := NewDefaultConfig()
	classifier := NewImageClassifier(config)

	// Use simplified training with validation
	if err := classifier.TrainWithValidation(); err != nil {
		log.Fatal("Training failed:", err)
	}

	// Make prediction on "a.png"
	fmt.Println("\nMaking prediction on 'a.png'...")
	prediction, confidence, err := classifier.Predict("a.png")
	if err != nil {
		log.Printf("Failed to predict image: %v", err)
	} else {
		fmt.Printf("Prediction: '%s' (confidence: %.2f%%)\n", prediction, confidence*100)
	}

	fmt.Println("Program completed successfully!")
}