default: build

build: fix
	go build -v -ldflags '-s -w' .

fix: *.go
	goimports -l -w .
	gofmt -l -w .

training: default
	cpe-net.exe training

magic: default
	cpe-net.exe
