package main

const (
	// Image processing constants
	RedLuminance   = 0.299
	GreenLuminance = 0.587
	BlueLuminance  = 0.114
	RGBAMax        = 65535.0
)

// ClassMapping maps class names to output indices
var ClassMapping = generateClassMapping()
var IndexToClass = make(map[int]string)

// generateClassMapping creates the mapping programmatically
func generateClassMapping() map[string]int {
	mapping := make(map[string]int)
	index := 0

	// Uppercase A-Z (indices 0-25)
	for i := 'A'; i <= 'Z'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	// Lowercase a-z (indices 26-51)
	for i := 'a'; i <= 'z'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	// Digits 0-9 (indices 52-61)
	for i := '0'; i <= '9'; i++ {
		mapping[string(i)] = index
		IndexToClass[index] = string(i)
		index++
	}

	return mapping
}