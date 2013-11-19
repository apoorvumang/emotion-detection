all: test detect
test: test.cpp
	g++ test.cpp -o test `pkg-config --libs opencv`
detect: detect.cpp
	g++ detect.cpp -o detect `pkg-config --libs opencv`