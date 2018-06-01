package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

type valuelessMap map[string]struct{}

type sample struct {
	input  string
	tokens map[string]int
	labels valuelessMap
}

type trainingDataSet struct {
	allTokens valuelessMap
	allLabels valuelessMap
	separator rune

	samples []sample
}

type trainingMetadata struct {
	Tokens    int    `json:"tokens"`
	Labels    int    `json:"labels"`
	Samples   int    `json:"samples"`
	Separator string `json:"separator"`
}

func newTrainingDataSet() *trainingDataSet {
	return &trainingDataSet{
		allTokens: make(valuelessMap),
		allLabels: make(valuelessMap),
		separator: ';',
	}
}

func (t *trainingDataSet) Add(input string, labels map[string]string) {
	sample := sample{}
	sample.input = input
	sample.tokens = tokenize(input)
	sample.labels = make(valuelessMap)

	for token := range sample.tokens {
		if _, exists := t.allTokens[token]; !exists {
			t.allTokens[token] = struct{}{}
		}
	}

	for property, value := range labels {
		label := fmt.Sprintf("%s:%s", property, value)
		if _, exists := t.allLabels[label]; !exists {
			t.allLabels[label] = struct{}{}
		}

		sample.labels[label] = struct{}{}
	}

	t.samples = append(t.samples, sample)
}

func (t *trainingDataSet) Metadata() trainingMetadata {
	return trainingMetadata{
		Tokens:    t.NumTokens(),
		Labels:    t.NumLabels(),
		Samples:   t.NumSamples(),
		Separator: string(t.separator),
	}
}

func (t *trainingDataSet) NumTokens() int {
	return len(t.allTokens)
}

func (t *trainingDataSet) NumLabels() int {
	return len(t.allLabels)
}

func (t *trainingDataSet) NumSamples() int {
	return len(t.samples)
}

func (t *trainingDataSet) CSV() string {
	tokens := getKeys(t.allTokens)
	labels := getKeys(t.allLabels)

	header := []string{"input"}
	header = append(header, tokens...)
	header = append(header, labels...)

	data := make([]string, 0)

	for _, sample := range t.samples {
		values := []string{sample.input}

		for _, token := range tokens {
			values = append(values, strconv.Itoa(sample.tokens[token]))
		}

		for _, label := range labels {
			value := "0"

			if _, exists := sample.labels[label]; exists {
				value = "1"
			}

			values = append(values, value)
		}

		data = append(data, strings.Join(values, string(t.separator)))
	}

	sort.Strings(data)

	csv := []string{strings.Join(header, string(t.separator))}
	csv = append(csv, data...)

	return strings.Join(csv, "\n")
}

func createTrainingDataSet(source string) (*trainingDataSet, error) {
	file, err := os.Open(source)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	type sample struct {
		Labels  map[string]string
		Aliases []string
	}

	testcases := make(map[string]sample)
	decoder := yaml.NewDecoder(file)

	err = decoder.Decode(&testcases)
	if err != nil {
		return nil, err
	}

	set := newTrainingDataSet()
	for swname, sample := range testcases {
		set.Add(swname, sample.Labels)

		for _, alias := range sample.Aliases {
			set.Add(alias, sample.Labels)
		}
	}

	return set, nil
}

func getKeys(m valuelessMap) []string {
	result := make([]string, len(m))

	i := 0
	for key := range m {
		result[i] = key
		i++
	}

	sort.Strings(result)

	return result
}
