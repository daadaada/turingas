#include <cuda.h>
#include <iostream>

using namespace std;

int main() {
  int * input;
  int * output;
  int * input_d;
  int * output_d;

  input  = (int*)malloc(2*sizeof(int));
  output = (int*)malloc(2*sizeof(int));

  input[0] = 10;
  input[1] = 20;
  output[0] = 0;
  output[1] = 0;

	cout << "Before the copy kernel." << endl;
	cout << "Input value:\t" << input[0] << "\t" << input[1] << endl;
  cout << "Output value:\t" << output[0] << "\t" << output[1] << endl;

  cudaMalloc((void**)&input_d,  2*sizeof(int));
  cudaMalloc((void**)&output_d, 2*sizeof(int));

  cudaMemcpy(input_d, input, 2*sizeof(int), cudaMemcpyHostToDevice);
  
  CUmodule module;
  CUfunction kernel;

  cuModuleLoad(&module, "copy.cubin");
  cuModuleGetFunction(&kernel, module, "kern");

  void * args[2] = {&input_d, &output_d};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 1, 1, 1, 
                 0, 0, args, 0);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, 2*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(input, input_d, 2*sizeof(int), cudaMemcpyDeviceToHost);

	cout << "After the copy kernel." << endl;
	cout << "Input value:\t" << input[0] << "\t" << input[1] << endl;
  cout << "Output value:\t" << output[0] << "\t" << output[1] << endl;


  return 0;
  
}
