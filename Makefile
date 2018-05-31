default: build

build: fix
	go build -v -tags netgo -ldflags '-s -w' .

fix: *.go
	goimports -l -w .
	gofmt -l -w .

training: default
	cpe-net.exe training > data/training.csv
	copy data\training.csv data\test.csv

magic: default
	cpe-net.exe
