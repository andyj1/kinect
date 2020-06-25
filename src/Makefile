K4A_FLAGS = -lk4a
K4ABT_FLAGS = -lk4abt
OPENCV_FLAGS = `pkg-config --cflags --libs opencv`
OUTPUT_FILENAME = program

# 1 device
one:
	g++ one.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

onerun: 
	./$(OUTPUT_FILENAME)

stats:
	g++ stats.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)
	./$(OUTPUT_FILENAME)

gen:
	g++ generalize_sync.cpp $(K4A_FLAGS) $(K4ABT_FLAGS) $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)
	./$(OUTPUT_FILENAME) 3 9 6 15 

# transformed, projected, synced, 2 devices
two:
	g++ two.cpp $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

tworun:
	./$(OUTPUT_FILENAME) 2 9 6 15

# transformed, projected, synced, 3 devices
sync:
	g++ sync.cpp $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) -o $(OUTPUT_FILENAME)

syncrun:
	./$(OUTPUT_FILENAME) 3 9 6 15