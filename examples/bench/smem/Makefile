all:
	python -m turingas.main -i lds32.sass -o lds32.cubin
	nvcc -arch=sm_75 main.cu -lcuda -o test

clean:
	rm lds32.cubin test
