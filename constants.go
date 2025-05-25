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

// ClassMapping maps class names (like "A", "b", "7", "*") to numerical indices (0, 1, 2, ...).
// IndexToClass provides the reverse mapping from indices back to class names.
//
// WHY NUMERICAL INDICES?
// Neural networks output numbers, not strings. When classifying characters, the
// network doesn't output "A" - it outputs a vector of probabilities like:
// [0.95, 0.02, 0.01, 0.01, 0.01, ...] where each position corresponds to a class.
// Position 0 might represent "A", position 1 might represent "B", etc.
//
// UPDATED MAPPING STRATEGY:
// We assign indices systematically:
// - Uppercase A-Z get indices 0-25
// - Lowercase a-z get indices 26-51  
// - Digits 0-9 get indices 52-61
// - Punctuation marks get indices 62-93
// This gives us 94 total classes, which matches our network's output size.
//
// CONSISTENCY IS CRITICAL:
// The same mapping must be used during training (when we convert labels to indices)
// and during prediction (when we convert network outputs back to characters).
// If these mappings don't match, the network will appear to make wrong predictions
// even when it's actually working correctly!
var ClassMapping = generateClassMapping()
var IndexToClass = make(map[int]string)

// PunctuationDirToChar maps directory names to actual punctuation characters.
//
// WHY THIS MAPPING?
// Some punctuation characters can't be used as directory names on most file systems:
// - "/" is a path separator
// - "?" has special meaning in some shells
// - "*" is a wildcard character
// - ":" has special meaning on Windows
// - etc.
//
// So we use descriptive directory names and map them to the actual characters.
// This keeps the file system organization clean while preserving the actual
// character information needed for training.
var PunctuationDirToChar = map[string]string{
	"asterisk":   "*",  // * symbol
	"backslash":  "\\", // \ symbol
	"colon":      ":",  // : symbol
	"dot":        ".",  // . symbol
	"gt":         ">",  // > symbol
	"lt":         "<",  // < symbol
	"pipe":       "|",  // | symbol
	"question":   "?",  // ? symbol
	"quote":      "\"", // " symbol
	"slash":      "/",  // / symbol
	// Direct character mappings (these can be used as directory names)
	"_":  "_",
	"-":  "-",
	",":  ",",
	";":  ";",
	"!":  "!",
	"'":  "'",
	"(":  "(",
	")":  ")",
	"[":  "[",
	"]":  "]",
	"{":  "{",
	"}":  "}",
	"@":  "@",
	"&":  "&",
	"#":  "#",
	"%":  "%",
	"`":  "`",
	"^":  "^",
	"+":  "+",
	"=":  "=",
	"~":  "~",
	"$":  "$",
}

// generateClassMapping creates the mapping programmatically to avoid errors.
//
// WHY GENERATE PROGRAMMATICALLY?
// Rather than manually typing out all 94 mappings (which would be error-prone),
// we use Go's character arithmetic to generate them systematically. This ensures:
// - No typos in the mapping
// - Consistent ordering
// - Easy to modify if we add/remove character classes
//
// CHARACTER ARITHMETIC IN GO:
// In Go, characters are just numbers (Unicode code points). 'A' is 65, 'B' is 66, etc.
// We can loop from 'A' to 'Z' by incrementing the character value, which is much
// cleaner than listing every character manually.
//
// UPDATED FOR PUNCTUATION:
// Now we also handle punctuation marks by iterating through our predefined
// mapping of directory names to actual characters.
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

	// PUNCTUATION MARKS: → indices 62-93
	// We iterate through our predefined punctuation mapping to ensure
	// consistent ordering and proper character representation
	//
	// IMPORTANT: The order here determines the class indices, so we use
	// a consistent iteration order by sorting the directory names
	punctuationDirs := []string{
		// Symbols that use their actual character as directory name
		"!", "#", "$", "%", "&", "'", "(", ")", "+", ",", "-", ".", ":", ";", 
		"=", "@", "[", "]", "^", "_", "`", "{", "}", "~",
		// Symbols that use descriptive directory names
		"asterisk", "backslash", "colon", "dot", "gt", "lt", "pipe", "question", "quote", "slash",
	}

	// Only map the directories that actually exist in our PunctuationDirToChar mapping
	for _, dirName := range punctuationDirs {
		if char, exists := PunctuationDirToChar[dirName]; exists {
			mapping[char] = index
			IndexToClass[index] = char
			index++
		}
	}

	return mapping
}

// EXAMPLE USAGE IN TRAINING:
// When we load a training image from "data/punctuation/asterisk/image1.png":
// 1. We extract "asterisk" from the directory name
// 2. We look up PunctuationDirToChar["asterisk"] to get "*"
// 3. We look up ClassMapping["*"] to get the index (e.g., 62)
// 4. We create a one-hot vector where position 62 is 1

// EXAMPLE USAGE IN PREDICTION:
// When the network outputs probabilities with highest at index 62:
// 1. We look up IndexToClass[62] to get "*"
// 2. We return "*" as the predicted character

// EXTENSIBILITY:
// Adding new punctuation marks is easy:
// 1. Add the directory name and character to PunctuationDirToChar
// 2. The generateClassMapping function will automatically include it
// 3. Remember to update the network's OutputSize to match the new total!

// TOTAL CLASSES: 26 + 26 + 10 + 32 = 94 characters