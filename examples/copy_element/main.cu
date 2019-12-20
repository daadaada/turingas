#include <cuda.h>
#include <iostream>

using namespace std;

int main() {
  int * input;
  int * output;
  int * input_d;
  int * output_d;

  input  = (int*)malloc(sizeof(int));
  output = (int*)malloc(sizeof(int));

  input[0] = 10;
  output[0] = 0;

	cout << "Before the copy kernel." << endl;
	cout << "Input value:\t" << input[0] << endl;
  cout << "Output value:\t" << (uint)output[0] << endl;

  cudaMalloc((void**)&input_d,  sizeof(int));
  cudaMalloc((void**)&output_d, sizeof(int));

  cudaMemcpy(input_d, input, sizeof(int), cudaMemcpyHostToDevice);
  
  CUmodule module;
  CUfunction kernel;

  cuModuleLoad(&module, "copy.cubin");
  cuModuleGetFunction(&kernel, module, "kern");

  void * args[2] = {&input_d, &output_d};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 1, 1, 1, 
                 0, 0, args, 0);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(input, input_d, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "After the copy kernel." << endl;
	cout << "Input value:\t" << input[0] << endl;
  cout << "Output value:\t" << output[0] << endl;


  return 0;
  
}
