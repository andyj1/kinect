K4A_FLAGS = -lk4a
K4ABT_FLAGS = -lk4abt
OPENCV_FLAGS = `pkg-config --cflags --libs opencv`
OUTPUT_FILENAME = program

all:
	g++ stats.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)
	./$(OUTPUT_FILENAME)