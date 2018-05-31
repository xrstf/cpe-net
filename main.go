package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

const (
	inputNeurons  = 27
	outputNeurons = 25
)

func main() {
	flag.Parse()

	// err := createTokenList()
	// if err != nil {
	// 	log.Fatal(err)
	// }

	if flag.Arg(0) == "training" {
		err := createTrainingDataSet()
		if err != nil {
			log.Fatal(err)
		}

		os.Exit(0)
	}

	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("data/training.csv")

	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  inputNeurons,
		outputNeurons: outputNeurons,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Train the neural network.
	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// fmt.Printf("predictions = %#v\n", predictions)

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()

	// fmt.Printf("predictions = %#v (numPreds = %d)\n", predictions, numPreds)

	for i := 0; i < numPreds; i++ {
		fmt.Printf("PREDICTION %d\n", i+1)

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		// fmt.Printf("label row = %#v (this is the expectation)\n", labelRow)
		// fmt.Printf("at 0 = %#v\n", predictions.At(i, 0))
		// fmt.Printf("at 1 = %.6f\n", predictions.At(i, 1))

		expects := make([]string, 0)
		for _, val := range labelRow {
			expects = append(expects, fmt.Sprintf("%.3F", val))
		}

		predict := make([]string, 0)
		for j := 0; j < len(labelRow); j++ {
			predict = append(predict, fmt.Sprintf("%.3F", predictions.At(i, j)))
		}

		failure := 0.0
		diffs := make([]string, 0)

		for j := 0; j < len(labelRow); j++ {
			e := labelRow[j]
			p := predictions.At(i, j)
			d := p - e

			if d < 0 {
				d = -d
			}

			diffs = append(diffs, fmt.Sprintf("%.3F", d))
			failure += d
		}

		// fmt.Printf("expectation = %s\n", strings.Join(expects, ", "))
		// fmt.Printf("prediction  = %s\n", strings.Join(predict, ", "))
		fmt.Printf("failure     = %.2f%%\n", failure/float64(len(labelRow))*100)

		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nnumPreds = %d\n", numPreds)
	fmt.Printf("\ntruePosNeg = %d\n", truePosNeg)
	fmt.Printf("\nAccuracy = %0.2f\n", accuracy)
}

type testcase map[string]string

type training struct {
	inputs  map[string]int
	outputs map[string]struct{}
}

type trainingDataSet struct {
	inputNeurons  map[string]struct{}
	outputNeurons map[string]struct{}

	testcases []training
}

func newTrainingDataSet() *trainingDataSet {
	return &trainingDataSet{
		inputNeurons:  make(map[string]struct{}),
		outputNeurons: make(map[string]struct{}),
	}
}

func (t *trainingDataSet) Add(inputTokens map[string]int, outputNeurons map[string]string) {
	train := training{}
	train.inputs = inputTokens
	train.outputs = make(map[string]struct{})

	for token := range inputTokens {
		if _, exists := t.inputNeurons[token]; !exists {
			t.inputNeurons[token] = struct{}{}
		}
	}

	for property, value := range outputNeurons {
		neuronName := fmt.Sprintf("%s__%s", property, value)
		if _, exists := t.outputNeurons[neuronName]; !exists {
			t.outputNeurons[neuronName] = struct{}{}
		}

		train.outputs[neuronName] = struct{}{}
	}

	t.testcases = append(t.testcases, train)
}

func (t *trainingDataSet) CSV() string {
	inputs := make([]string, len(t.inputNeurons))
	outputs := make([]string, len(t.outputNeurons))

	i := 0
	for neuron := range t.inputNeurons {
		inputs[i] = neuron
		i++
	}

	i = 0
	for neuron := range t.outputNeurons {
		outputs[i] = neuron
		i++
	}

	header := make([]string, 0)
	header = append(header, inputs...)
	header = append(header, outputs...)

	csv := make([]string, 0)
	csv = append(csv, strings.Join(header, ","))

	for _, testcase := range t.testcases {
		values := make([]string, 0)

		for _, name := range inputs {
			values = append(values, strconv.Itoa(testcase.inputs[name]))
		}

		for _, name := range outputs {
			value := "0"

			if _, exists := testcase.outputs[name]; exists {
				value = "1"
			}

			values = append(values, value)
		}

		csv = append(csv, strings.Join(values, ","))
	}

	return strings.Join(csv, "\n")
}

func createTrainingDataSet() error {
	file, err := os.Open("data/training.json")
	if err != nil {
		return err
	}
	defer file.Close()

	testcases := make(map[string]testcase)
	decoder := json.NewDecoder(file)

	err = decoder.Decode(&testcases)
	if err != nil {
		return err
	}

	training := newTrainingDataSet()

	for swname, test := range testcases {
		tokens := tokenize(swname)

		training.Add(tokens, test)
	}

	fmt.Println(training.CSV())

	log.Printf("input neurons  = %d\n", len(training.inputNeurons))
	log.Printf("output neurons = %d\n", len(training.outputNeurons))

	return nil
}

func createTokenList() error {
	file, err := os.Open("data/softwares.txt")
	if err != nil {
		return err
	}
	defer file.Close()

	tokens := make(map[string]struct{})

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		for token := range tokenize(scanner.Text()) {
			if _, exists := tokens[token]; !exists {
				tokens[token] = struct{}{}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	output := make([]string, len(tokens))
	i := 0

	for token := range tokens {
		output[i] = token
		i++
	}

	sort.Strings(output)

	out, err := os.Create("data/tokens.txt")
	if err != nil {
		return err
	}
	defer out.Close()

	for _, token := range output {
		out.WriteString(token + "\n")
	}

	return nil
}

// NewNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// train trains a neural network using backpropagation.
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// backpropagate completes the backpropagation method.
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < nn.config.numEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

// predict makes a prediction based on a trained
// neural network.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

// sumAlongAxis sums a matrix along a
// particular dimension, preserving the
// other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = inputNeurons + outputNeurons

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, inputNeurons*(len(rawCSVData)-1))
	labelsData := make([]float64, outputNeurons*(len(rawCSVData)-1))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {
		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i > (inputNeurons - 1) {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData)-1, inputNeurons, inputsData)
	labels := mat.NewDense(len(rawCSVData)-1, outputNeurons, labelsData)
	return inputs, labels
}
