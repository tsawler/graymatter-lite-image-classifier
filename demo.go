package main

import (
	"fmt"
	"log"
)

// DemoTraining shows how to train a new model
func DemoTraining() {
	fmt.Println("=== Training Demo ===")
	
	config := NewDefaultConfig()
	config.TrainingOptions.Iterations = 100 // Shorter for demo
	config.TrainingOptions.EnablePlotting = true
	config.TrainingOptions.EnableSaving = true
	
	classifier := NewImageClassifier(config)
	
	if err := classifier.TrainWithValidation(); err != nil {
		log.Printf("Training failed: %v", err)
		return
	}
	
	// Save the trained model
	if err := classifier.SaveModel("demo_model.json", "Demo image classifier"); err != nil {
		log.Printf("Failed to save model: %v", err)
	} else {
		fmt.Println("Model saved successfully!")
	}
}

// DemoInference shows how to load a model and make predictions
func DemoInference() {
	fmt.Println("=== Inference Demo ===")
	
	// Load a pre-trained model
	classifier, metadata, err := LoadModelForInference("demo_model.json")
	if err != nil {
		log.Printf("Failed to load model: %v", err)
		return
	}
	
	fmt.Printf("Loaded model: %s\n", metadata.Description)
	fmt.Printf("Training accuracy: %.2f%%\n", metadata.TrainingAccuracy*100)
	
	// Make predictions on test images
	testImages := []string{"a.png", "b.png", "1.png", "Z.png"}
	
	for _, imagePath := range testImages {
		prediction, confidence, err := classifier.Predict(imagePath)
		if err != nil {
			log.Printf("Prediction failed for %s: %v", imagePath, err)
			continue
		}
		
		fmt.Printf("Image: %s -> Prediction: '%s' (%.2f%% confidence)\n", 
			imagePath, prediction, confidence*100)
	}
}

// DemoEvaluation shows how to evaluate a model on test data
func DemoEvaluation() {
	fmt.Println("=== Evaluation Demo ===")
	
	classifier, _, err := LoadModelForInference("demo_model.json")
	if err != nil {
		log.Printf("Failed to load model: %v", err)
		return
	}
	
	// Evaluate on test data
	accuracy, err := classifier.EvaluateModel("test_data")
	if err != nil {
		log.Printf("Evaluation failed: %v", err)
		return
	}
	
	fmt.Printf("Test accuracy: %.2f%%\n", accuracy*100)
}
