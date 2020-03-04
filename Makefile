K4A_FLAGS = -lk4a
K4ABT_FLAGS = -lk4abt
OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
OUTPUT_FILENAME = -o program

all:
	g++ one_unit_body_tracker.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) $(OUTPUT_FILENAME)

run: 
	./program


ir:
	g++ IR_image_track.cpp  $(K4A_FLAGS) $(K4ABT_FLAGS)  $(OPENCV_FLAGS) $(OUTPUT_FILENAME)

irrun: 
	./program 



