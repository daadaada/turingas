#include <cuda.h>
#include <iostream>

using namespace std;

int main() {
  int * input;
  int * output;
  int * input_d;
  int * output_d;

  input  = (int*)malloc(2 * sizeof(int));
  output = (int*)malloc(2 * sizeof(int));

  input[0] = 0;
  output[0] = 0;

  cudaMalloc((void**)&input_d,  2 * sizeof(int));
  cudaMalloc((void**)&output_d, 2 * sizeof(int));

  cudaMemcpy(input_d, input, 2 * sizeof(int), cudaMemcpyHostToDevice);
  
  CUmodule module;
  CUfunction kernel;

  cuModuleLoad(&module, "a.cubin");
  cuModuleGetFunction(&kernel, module, "kern");

  void * args[2] = {&input_d, &output_d};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 32, 1, 1, 
                 0, 0, args, 0);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(input, input_d, 2 * sizeof(int), cudaMemcpyDeviceToHost);

  cout << "Result:\t" << (uint)output[0] << endl;


  return 0;
  
}
