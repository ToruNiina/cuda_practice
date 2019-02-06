LIB := -lglfw -lGL -lX11
LIB += -lcurand

.PHONY: all
all: main.o ising.o render.o glad.o
	nvcc main.o ising.o render.o glad.o $(LIB)

.PHONY: clean
clean:
	rm ./*.o a.out

ising.o:
	nvcc -c ising.cu -std=c++11 -O2

render.o:
	nvcc -c render.cu -std=c++11 -O2

main.o:
	nvcc -c main.cu -std=c++11 -O2 -I./glad/include -I/usr/local/cuda/include

glad.o:
	gcc -c glad/src/glad.c -O2 -I./glad/include
