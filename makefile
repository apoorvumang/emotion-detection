all: eye-train eye-detect mouth-train mouth-detect severity-detect main
eye-train: eye-train.cpp
	g++ eye-train.cpp -o eye-train `pkg-config --libs opencv`
eye-detect: eye-detect.cpp
	g++ eye-detect.cpp -o eye-detect `pkg-config --libs opencv`
mouth-train: mouth-train.cpp
	g++ mouth-train.cpp -o mouth-train `pkg-config --libs opencv`
mouth-detect: mouth-detect.cpp
	g++ mouth-detect.cpp -o mouth-detect `pkg-config --libs opencv`
severity-detect: severity-detect.cpp
	g++ severity-detect.cpp -o severity-detect `pkg-config --libs opencv`
main: main.cpp
	g++ main.cpp -o main `pkg-config --libs opencv`