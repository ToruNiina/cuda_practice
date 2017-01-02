all:
	nvcc -std=c++11 -O3 -DOUTPUT_PNG ising_model.cu -o ising -lcurand -lpng12

gif:
	convert -loop 0 -delay 4 ising*.png ising.gif

clean:
	rm ./ising ./ising*.png
