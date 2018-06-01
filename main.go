package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"strings"

	"gonum.org/v1/gonum/mat"
)

var (
	hiddenNeuronsFlag = flag.Int("hidden", 3, "number of hidden neurons")
	epochsFlag        = flag.Int("epochs", 5000, "number of epochs during training phase")
	learnRateFlag     = flag.Float64("learnrate", 0.3, "learning rate")
	thresholdFlag     = flag.Float64("threshold", 0.75, "required confidence rate to consider a label a true match")
)

func main() {
	log.SetFlags(0)
	flag.Parse()

	var err error

	if flag.Arg(0) == "training" {
		err = convertTrainingDataSet()
	} else {
		err = queryNetwork(strings.Join(flag.Args(), " "))
	}

	if err != nil {
		log.Fatal(err)
	}
}

func queryNetwork(query string) error {
	/////////////////////////////////////////////////////////////////////////////
	// step 1: prepare the network

	log.Println("Reading training data set...")

	content, err := ioutil.ReadFile("data/training.meta.json")
	if err != nil {
		return err
	}

	meta := trainingMetadata{}
	err = json.Unmarshal(content, &meta)
	if err != nil {
		return err
	}

	// Form the training matrices.
	inputs, inputNames, labels, labelNames := makeInputsAndLabels("data/training.csv", meta)
	rows, _ := inputs.Dims()

	log.Printf("Found %d training samples.", rows)

	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  meta.Tokens,
		outputNeurons: meta.Labels,
		hiddenNeurons: *hiddenNeuronsFlag,
		numEpochs:     *epochsFlag,
		learningRate:  *learnRateFlag,
	}

	// train the neural network
	log.Println("Training neural network...")

	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		return err
	}

	log.Println("Training completed.")

	/////////////////////////////////////////////////////////////////////////////
	// step 2: prepare our "query"

	queryTokens := tokenize(query)
	log.Printf("Tokenized query: %v\n", queryTokens)

	// map the tokens to those available in our network
	queryInputs := make([]float64, len(inputNames))

	for i, token := range inputNames {
		queryInputs[i] = float64(queryTokens[token])
		delete(queryTokens, token)
	}

	if len(queryTokens) > 0 {
		log.Printf("Warning, these tokens do not appear in our training data set and are ignored: %v\n", queryTokens)
	}

	queryMatrix := mat.NewDense(1, meta.Tokens, queryInputs)

	/////////////////////////////////////////////////////////////////////////////
	// step 3: run the query through the network

	log.Println("Predicting result...")

	// Make the predictions using the trained model.
	predictions, err := network.predict(queryMatrix)
	if err != nil {
		return err
	}

	prediction := predictions.RawRowView(0)

	/////////////////////////////////////////////////////////////////////////////
	// step 4: map the result to the neuron names

	log.Println("Prediction:")

	for i, token := range labelNames {
		predictedValue := prediction[i]

		// must be at least 75% sure that the network is correct
		if predictedValue > *thresholdFlag {
			parts := strings.SplitN(token, ":", 2)
			log.Printf("  %s: %s (%.2f%% certain)\n", parts[0], parts[1], predictedValue*100)
		}
	}

	return nil
}

func convertTrainingDataSet() error {
	set, err := createTrainingDataSet("data/training.yaml")
	if err != nil {
		return err
	}

	log.Printf("Samples : %d\n", set.NumSamples())
	log.Printf("Tokens  : %d\n", set.NumTokens())
	log.Printf("Labels  : %d\n", set.NumLabels())

	err = ioutil.WriteFile("data/training.csv", []byte(set.CSV()), 0644)
	if err != nil {
		return err
	}

	encoded, err := json.Marshal(set.Metadata())
	if err != nil {
		return err
	}

	return ioutil.WriteFile("data/training.meta.json", encoded, 0644)
}
