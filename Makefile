EXEC := "build/Project3-Pathtracer"

.PHONY: all debug release run run-debug format clean

all: debug

debug:
	(mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=debug && make)

release:
	(mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=release && make)

run:
	${EXEC} scene=data/scenes/sampleScene.txt

run-debug:
	CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 cuda-gdb --args ${EXEC} scene=data/scenes/sampleScene.txt

format:
	astyle --mode=c --style=1tbs -pcHs4 -r 'src/*.cpp' 'src/*.h' 'src/*.cu' 'src/*.h'

clean:
	rm -rf build
