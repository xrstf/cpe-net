# CPE network

This is just an experiment with neural networks. The network code itself is based
on https://github.com/dwhitena/gophernet.

## Usage

There are two modes: you can either use the program to convert the manually
edited `data/training.json` to a CSV file or you can use the program as a
network, using the CSV file to train it and then query it.

### Generate Training Data

Extend the samples in `data/training.json`. Then run `make training` to compile
the program and make it convert the JSON to CSV. This will also update the
`data/training.meta.json` file.

### Query The Network

Run `make` to compile it. Then start it and give your query as CLI arguments
(the program will automatically join multiple arguments together into one query):

    $ make

    $ echo "The following two invocations are equivalent."

    $ ./cpe-net "Windows 10 x64"
    $ ./cpe-net Windows 10 x64
    Reading training data set...
    Found 32 training samples.
    Training neural network...
    Training completed.
    Tokenized query: map[windows:1 10:1 x64:1]
    Predicting result...
    Prediction:
      product: Windows (99.99% certain)
      vendor: Microsoft (99.99% certain)
      version: 10 (79.22% certain)
