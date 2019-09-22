python ../../main.py -i ldc.sass -o a.cubin
nvcc main.cu -lcuda -arch=sm_75
