package main

import (
	"regexp"
	"strings"
)

var (
	isVersion = regexp.MustCompile(`[0-9]+\.[0-9]+(\.[0-9]+)+`)
	isDate    = regexp.MustCompile(`[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]`)
	isDashes  = regexp.MustCompile(`^-+$`)
)

func tokenize(software string) map[string]int {
	result := make(map[string]int)

	software = strings.Replace(software, `"`, "", -1)
	software = strings.Replace(software, `(`, "", -1)
	software = strings.Replace(software, `)`, "", -1)
	software = strings.Replace(software, `'`, "", -1)

	for _, token := range strings.Fields(strings.ToLower(software)) {
		token = cleanupToken(token)

		if !isRelevantToken(token) {
			continue
		}

		if _, exists := result[token]; !exists {
			result[token] = 0
		}

		result[token]++
	}

	return result
}

func cleanupToken(token string) string {
	token = strings.TrimPrefix(token, "-")
	token = strings.TrimPrefix(token, ",")
	token = strings.TrimSuffix(token, "-")
	token = strings.TrimSuffix(token, ",")

	return token
}

func isRelevantToken(token string) bool {
	if len(token) == 0 {
		return false
	}

	if isVersion.MatchString(token) || isDate.MatchString(token) || isDashes.MatchString(token) {
		return false
	}

	return true
}
