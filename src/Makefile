K4A_FLAGS = -lk4a
K4ABT_FLAGS = -lk4abt
OPENCV_FLAGS = `pkg-config --cflags --libs opencv`
OUTPUT_FILENAME = program

one:
	g++ single.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

onerun: 
	./$(OUTPUT_FILENAME)

two:
	g++ multi.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

tworun: 
	./$(OUTPUT_FILENAME) 2 9 6 15 1

sync:
	g++ sync.cpp $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

syncrun:
	./$(OUTPUT_FILENAME) 2 9 6 15 1

gen:
	g++ generalize_sync.cpp $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

genrun:
	./$(OUTPUT_FILENAME) 3 9 6 15 1