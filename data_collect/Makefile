K4A_FLAGS = -lk4a
K4ABT_FLAGS = -lk4abt
OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
OUTPUT_FILENAME = -o program

one:
	g++ single.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) $(OUTPUT_FILENAME)

run: 
	./program


two:
	g++ multi.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) $(OUTPUT_FILENAME)

tworun: 
	./program 2 9 6 15 1

