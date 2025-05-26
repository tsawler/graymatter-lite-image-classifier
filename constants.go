package main

// Image processing constants for converting color images to grayscale.
//   
// WHY CONVERT TO GRAYSCALE?
// Neural networks work with numbers, and color images have 3 numbers per pixel
// (red, green, blue channels). For character recognition, color information is
// usually not important - a black 'A' on white paper and a blue 'A' on yellow
// paper represent the same character. Converting to grayscale:
// - Reduces input complexity from 3 values per pixel to 1 value
// - Makes the neural network simpler and faster to train
// - Reduces memory usage significantly (3x reduction)
// - Preserves the essential shape information needed for character recognition
const (
	// RedLuminance: Weight for red channel in grayscale conversion (29.9%)
	// GreenLuminance: Weight for green channel in grayscale conversion (58.7%)  
	// BlueLuminance: Weight for blue channel in grayscale conversion (11.4%)
	//
	// THESE WEIGHTS ARE NOT ARBITRARY:
	// These values come from decades of research on human visual perception. Our eyes
	// have different sensitivities to different wavelengths of light:
	// - Most sensitive to green light (hence the highest weight at 58.7%)
	// - Moderately sensitive to red light (29.9%)
	// - Least sensitive to blue light (11.4%)
	//
	// The standard grayscale conversion formula is:
	// Gray = 0.299*Red + 0.587*Green + 0.114*Blue
	//
	// This produces grayscale images that look natural to human eyes and preserve
	// the most important visual information for character recognition tasks.
	//
	// ALTERNATIVE APPROACHES AND WHY WE DON'T USE THEM:
	// - Simple average (R+G+B)/3: Mathematically simpler but would make yellow text
	//   look too bright and blue text look too dark compared to human perception
	// - Max(R,G,B): Would lose important texture and detail information
	// - Single channel: Would be biased toward that color's characteristics
	//
	// Using perceptually-weighted values produces significantly better results
	// for any application where humans need to interpret the output.
	RedLuminance   = 0.299
	GreenLuminance = 0.587
	BlueLuminance  = 0.114
	
	// RGBAMax: Maximum value for RGBA color channels (65535 for 16-bit)
	//
	// WHY 65535 AND NOT 255?
	// Go's image package uses 16-bit color channels internally, where each
	// color component ranges from 0 to 65535 (2^16 - 1). Even if your original
	// image uses 8-bit color (0-255 per channel), Go automatically converts
	// it to this higher precision format for consistency.
	//
	// NORMALIZATION IMPORTANCE:
	// Neural networks work best when input values are in a consistent, small range.
	// Raw pixel values (0-65535) are too large and varied for optimal learning.
	// By dividing by RGBAMax, we normalize pixels to the 0.0-1.0 range:
	// - 0.0 represents completely black
	// - 1.0 represents completely white  
	// - 0.5 represents middle gray
	//
	// This normalization is critical for neural network performance because:
	// - Large input values can cause numerical instability
	// - Consistent ranges help with weight initialization
	// - Activation functions work best with inputs in expected ranges
	RGBAMax = 65535.0
)

// ClassMapping maps class names (like "A", "b", "7", "*") to numerical indices (0, 1, 2, ...).
// IndexToClass provides the reverse mapping from indices back to class names.
//
// WHY NUMERICAL INDICES?
// Neural networks are fundamentally mathematical systems that work with numbers,
// not strings or symbols. When classifying characters, the network doesn't output
// "A" directly - it outputs a vector of probabilities like:
// [0.95, 0.02, 0.01, 0.01, 0.01, ...] 
// where each position corresponds to a different character class.
//
// Position 0 might represent "A", position 1 might represent "B", etc.
// We need consistent mappings to convert between human-readable characters
// and the numerical indices the network uses internally.
//
// UPDATED MAPPING STRATEGY FOR 94 CLASSES:
// We assign indices systematically to make the system predictable and debuggable:
// - Uppercase A-Z get indices 0-25    (26 classes)
// - Lowercase a-z get indices 26-51   (26 classes)  
// - Digits 0-9 get indices 52-61      (10 classes)
// - Punctuation marks get indices 62-93 (32 classes)
// Total: 26 + 26 + 10 + 32 = 94 classes
//
// This systematic approach makes it easy to:
// - Debug issues by understanding which index range has problems
// - Add new character types by extending the ranges
// - Analyze performance by character type (letters vs digits vs punctuation)
//
// CONSISTENCY IS ABSOLUTELY CRITICAL:
// The exact same mapping must be used during:
// - Training (when converting labels to indices for the network)
// - Prediction (when converting network outputs back to characters)
// - Model saving and loading (to ensure compatibility)
//
// If these mappings don't match exactly, the network will appear to make wrong
// predictions even when it's actually working correctly! This is one of the most
// common sources of bugs in machine learning systems.
var ClassMapping = generateClassMapping()
var IndexToClass = make(map[int]string)

// PunctuationDirToChar maps directory names to actual punctuation characters.
//
// WHY THIS MAPPING IS NECESSARY:
// Many punctuation characters cannot be used as directory names on common
// file systems due to operating system restrictions:
// - "/" is the path separator on Unix/Linux/macOS
// - "\" is the path separator on Windows  
// - "?" has special meaning in shells and command prompts
// - "*" is a wildcard character in most operating systems
// - ":" has special meaning on Windows (drive letters)
// - "<", ">", "|" are shell redirection operators
// - Quotes can cause parsing issues in many contexts
//
// SOLUTION: DESCRIPTIVE DIRECTORY NAMES
// We use human-readable directory names that clearly describe the punctuation
// mark, then map them to the actual characters programmatically. This approach:
// - Keeps the file system organization clean and browsable
// - Avoids operating system compatibility issues
// - Makes it obvious what each directory contains
// - Prevents accidental shell interpretation of special characters
//
// MAPPING EXAMPLES:
// Directory Name → Actual Character → Use Case
// "asterisk"     → "*"             → Multiplication, emphasis
// "question"     → "?"             → Questions, uncertainty
// "slash"        → "/"             → Dates, fractions, paths
// "backslash"    → "\"             → Escape sequences, Windows paths
var PunctuationDirToChar = map[string]string{
	// CHARACTERS THAT REQUIRE DESCRIPTIVE NAMES (file system incompatible)
	"asterisk":   "*",  // Multiplication, emphasis, wildcards
	"backslash":  "\\", // Escape sequences, Windows file paths
	"colon":      ":",  // Time notation, ratios (conflicts with drive letters on Windows)
	"dot":        ".",  // Sentences, decimals, abbreviations
	"gt":         ">",  // Greater than, shell redirection
	"lt":         "<",  // Less than, shell redirection, HTML tags
	"pipe":       "|",  // Shell pipes, logical OR, tables
	"question":   "?",  // Questions, uncertainty, wildcards
	"quote":      "\"", // Quotations, string delimiters
	"slash":      "/",  // Division, dates, file paths, fractions
	
	// CHARACTERS THAT CAN BE USED DIRECTLY AS DIRECTORY NAMES
	// These don't conflict with file system or shell conventions
	"_":  "_",   // Underscores: variable names, emphasis
	"-":  "-",   // Hyphens: compound words, ranges, negative numbers
	",":  ",",   // Commas: lists, decimal separators (some locales)
	";":  ";",   // Semicolons: statement separators, complex lists
	"!":  "!",   // Exclamation: emphasis, commands, factorial
	"'":  "'",   // Apostrophes: contractions, possessives, quotes
	"(":  "(",   // Parentheses: grouping, function calls, asides
	")":  ")",   // Closing parentheses
	"[":  "[",   // Square brackets: arrays, lists, citations
	"]":  "]",   // Closing square brackets
	"{":  "{",   // Curly braces: sets, code blocks, formatting
	"}":  "}",   // Closing curly braces
	"@":  "@",   // At signs: email addresses, mentions, decorators
	"&":  "&",   // Ampersands: logical AND, company names ("Smith & Co")
	"#":  "#",   // Hash/pound: numbers, hashtags, preprocessor directives
	"%":  "%",   // Percent: percentages, modulo operation, formatting
	"`":  "`",   // Backticks: code formatting, command substitution
	"^":  "^",   // Carets: exponentiation, XOR, beginning of line
	"+":  "+",   // Plus: addition, positive numbers, concatenation
	"=":  "=",   // Equals: assignment, comparison, equations
	"~":  "~",   // Tildes: approximation, home directory, bitwise NOT
	"$":  "$",   // Dollar signs: currency, variables, end of line regex
}

// generateClassMapping creates the mapping programmatically to avoid errors.
//
// WHY GENERATE PROGRAMMATICALLY INSTEAD OF HARDCODING?
// Rather than manually typing out all 94 character mappings (which would be
// extremely error-prone and tedious to maintain), we use Go's character
// arithmetic to generate them systematically. This approach provides:
// - No possibility of typos in the mapping tables
// - Consistent, predictable ordering that's easy to understand
// - Easy modification if we need to add or remove character classes
// - Self-documenting code (the algorithm makes the structure clear)
// - Automatic consistency between ClassMapping and IndexToClass
//
// CHARACTER ARITHMETIC IN GO:
// In Go, characters (runes) are just numbers representing Unicode code points.
// 'A' is 65, 'B' is 66, 'a' is 97, 'b' is 98, etc.
// We can loop from 'A' to 'Z' by incrementing the character value, which is
// much cleaner and less error-prone than manually listing every character.
func generateClassMapping() map[string]int {
	mapping := make(map[string]int)
	index := 0

	// UPPERCASE LETTERS: A-Z → indices 0-25
	// 'A' maps to 0, 'B' maps to 1, ..., 'Z' maps to 25
	for i := 'A'; i <= 'Z'; i++ {
		char := string(i)           // Convert rune to string for use as map key
		mapping[char] = index       // Forward mapping: "A" → 0, "B" → 1, etc.
		IndexToClass[index] = char  // Reverse mapping: 0 → "A", 1 → "B", etc.
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
	// We iterate through our predefined punctuation mapping to ensure consistent
	// ordering and proper character representation
	//
	// IMPORTANT: The order here determines the class indices, so we use a
	// consistent iteration order by defining the sequence explicitly rather
	// than relying on Go's map iteration (which is randomized for security)
	punctuationOrder := []string{
		// Direct character mappings first (alphabetical by character)
		"!", "#", "$", "%", "&", "'", "(", ")", "+", ",", "-", ".", ":", ";", 
		"=", "@", "[", "]", "^", "_", "`", "{", "}", "~",
		// Descriptive mappings second (alphabetical by directory name)
		"asterisk", "backslash", "colon", "dot", "gt", "lt", "pipe", "question", "quote", "slash",
	}

	// Process punctuation in our defined order
	for _, dirName := range punctuationOrder {
		if char, exists := PunctuationDirToChar[dirName]; exists {
			mapping[char] = index
			IndexToClass[index] = char
			index++
		}
	}

	return mapping
}

// EXAMPLE USAGE SCENARIOS:

// DURING TRAINING:
// When we load a training image from "data/punctuation/asterisk/image1.png":
// 1. Extract "asterisk" from the directory path
// 2. Look up PunctuationDirToChar["asterisk"] to get "*"
// 3. Look up ClassMapping["*"] to get the numerical index (e.g., 62)
// 4. Create a one-hot vector where position 62 is 1.0, all others are 0.0
// 5. Use this as the target output for training the network

// DURING PREDICTION:
// When the network outputs probabilities with the highest at index 62:
// 1. Identify that index 62 has the highest probability
// 2. Look up IndexToClass[62] to get "*"
// 3. Return "*" as the predicted character to the user

// EXTENSIBILITY FOR FUTURE EXPANSION:
// Adding new punctuation marks is straightforward:
// 1. Add the directory name and character to PunctuationDirToChar
// 2. Add the directory name to punctuationOrder in the right position
// 3. Update the network's OutputSize to match the new total class count
// 4. The generateClassMapping function will automatically include the new character

// DEBUGGING AND MAINTENANCE:
// The systematic approach makes debugging easier:
// - Index 0-25: Check uppercase letter recognition
// - Index 26-51: Check lowercase letter recognition  
// - Index 52-61: Check digit recognition
// - Index 62-93: Check punctuation recognition
//
// If problems occur in a specific range, you know which character type to investigate.

// TOTAL CHARACTER CLASSES: 26 + 26 + 10 + 32 = 94 characters
// This comprehensive character set covers:
// - All English letters (uppercase and lowercase)
// - All Arabic numerals (0-9)
// - Common punctuation marks used in typical text
// - Special characters needed for programming and technical writing