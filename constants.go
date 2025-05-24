package main

// Image processing constants for converting color images to grayscale.
//   
// WHY CONVERT TO GRAYSCALE?
// Neural networks work with numbers, and color images have 3 numbers per pixel
// (red, green, blue). For character recognition, color usually doesn't matter -
// a black 'A' on white paper and a blue 'A' on yellow paper are the same character.
// Converting to grayscale reduces the input from 3 values per pixel to 1 value,
// making the network simpler and faster while preserving the essential information.
const (
	// RedLuminance: Weight for red channel in grayscale conversion (29.9%)
	// GreenLuminance: Weight for green channel in grayscale conversion (58.7%)  
	// BlueLuminance: Weight for blue channel in grayscale conversion (11.4%)
	//
	// THESE ARE NOT ARBITRARY:
	// These weights come from how the human eye perceives brightness. Our eyes
	// are most sensitive to green light, moderately sensitive to red, and least
	// sensitive to blue. The grayscale conversion formula mimics human perception:
	//
	// Gray = 0.299*Red + 0.587*Green + 0.114*Blue
	//
	// This produces grayscale images that look natural to human eyes and preserve
	// the most important visual information for character recognition.
	//
	// ALTERNATIVE APPROACHES:
	// Simple average (R+G+B)/3 would be mathematically simpler but would make
	// yellow text look too bright and blue text look too dark compared to how
	// humans perceive them. Using these weighted values produces better results.
	RedLuminance   = 0.299
	GreenLuminance = 0.587
	BlueLuminance  = 0.114
	
	// RGBAMax: Maximum value for RGBA color channels (65535 for 16-bit)
	//
	// WHY 65535?
	// Go's image package uses 16-bit color channels internally, where each
	// color component ranges from 0 to 65535. Even if your original image
	// uses 8-bit color (0-255), Go converts it to this higher precision format.
	// We need to know this maximum value to normalize pixel values to the
	// 0.0-1.0 range that neural networks prefer.
	//
	// NORMALIZATION IMPORTANCE:
	// Neural networks work best when input values are in a consistent, small range.
	// Raw pixel values (0-65535) are too large and varied. By dividing by RGRAMax,
	// we normalize pixels to 0.0-1.0, where 0.0 is black and 1.0 is white.
	RGBAMax = 65535.0
)

// ClassMapping maps class names (like "A", "b", "7") to numerical indices (0, 1, 2, ...).
// IndexToClass provides the reverse mapping from indices back to class names.
//
// WHY NUMERICAL INDICES?
// Neural networks output numbers, not strings. When classifying characters, the
// network doesn't output "A" - it outputs a vector of probabilities like:
// [0.95, 0.02, 0.01, 0.01, 0.01, ...] where each position corresponds to a class.
// Position 0 might represent "A", position 1 might represent "B", etc.
//
// MAPPING STRATEGY:
// We assign indices systematically:
// - Uppercase A-Z get indices 0-25
// - Lowercase a-z get indices 26-51  
// - Digits 0-9 get indices 52-61
// This gives us 62 total classes, which matches our network's output size.
//
// CONSISTENCY IS CRITICAL:
// The same mapping must be used during training (when we convert labels to indices)
// and during prediction (when we convert network outputs back to characters).
// If these mappings don't match, the network will appear to make wrong predictions
// even when it's actually working correctly!
var ClassMapping = generateClassMapping()
var IndexToClass = make(map[int]string)

// generateClassMapping creates the mapping programmatically to avoid errors.
//
// WHY GENERATE PROGRAMMATICALLY?
// Rather than manually typing out all 62 mappings (which would be error-prone),
// we use Go's character arithmetic to generate them systematically. This ensures:
// - No typos in the mapping
// - Consistent ordering
// - Easy to modify if we add/remove character classes
//
// CHARACTER ARITHMETIC IN GO:
// In Go, characters are just numbers (Unicode code points). 'A' is 65, 'B' is 66, etc.
// We can loop from 'A' to 'Z' by incrementing the character value, which is much
// cleaner than listing every character manually.
func generateClassMapping() map[string]int {
	mapping := make(map[string]int)
	index := 0

	// UPPERCASE LETTERS: A-Z → indices 0-25
	// 'A' maps to 0, 'B' maps to 1, ..., 'Z' maps to 25
	for i := 'A'; i <= 'Z'; i++ {
		char := string(i)           // Convert rune to string
		mapping[char] = index       // Forward mapping: "A" → 0
		IndexToClass[index] = char  // Reverse mapping: 0 → "A"
		index++
	}

	// LOWERCASE LETTERS: a-z → indices 26-51
	// 'a' maps to 26, 'b' maps to 27, ..., 'z' maps to 51
	for i := 'a'; i <= 'z'; i++ {
		char := string(i)
		mapping[char] = index
		IndexToClass[index] = char
		index++
	}

	// DIGITS: 0-9 → indices 52-61
	// '0' maps to 52, '1' maps to 53, ..., '9' maps to 61
	for i := '0'; i <= '9'; i++ {
		char := string(i)
		mapping[char] = index
		IndexToClass[index] = char
		index++
	}

	return mapping
}

// EXAMPLE USAGE IN TRAINING:
// When we load a training image from "data/upper/A/image1.png":
// 1. We extract "A" from the file path as the label
// 2. We look up ClassMapping["A"] to get index 0
// 3. We create a one-hot vector [1, 0, 0, 0, ...] where position 0 is 1

// EXAMPLE USAGE IN PREDICTION:
// When the network outputs probabilities [0.95, 0.02, 0.01, ...]:
// 1. We find the highest probability is at index 0
// 2. We look up IndexToClass[0] to get "A"
// 3. We return "A" as the predicted character

// EXTENSIBILITY:
// Adding new character classes is easy - just add them to this function.
// For example, to add punctuation marks:
//   for _, char := range []string{".", ",", "!", "?"} {
//       mapping[char] = index
//       IndexToClass[index] = char
//       index++
//   }
// Just remember to update the network's OutputSize to match the new total!