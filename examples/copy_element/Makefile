ARCH = 75

all:
	python -m turingas.main -i copy_element.sass -o copy.cubin -arch $(ARCH)
	nvcc main.cu -lcuda -arch=sm_$(ARCH) -o copy

clean:
	rm copy.cubin copy 
