#include <cuda.h>
#include <stdio.h>
#include <string.h>

char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void run(char * name){
	char * file_name = concat(name, ".cubin");

	int *output;
	cudaMalloc((void**)&output, sizeof(int)*128);

	CUmodule module;
	CUfunction kernel;

	cuModuleLoad(&module, file_name);
	cuModuleGetFunction(&kernel, module, "kern");

	void * args[1] = {&output};
	cuLaunchKernel(kernel, 1, 1, 1,
			32, 1, 1,
			32*1024, 0, args, 0);

	int *output_h = (int*)malloc(sizeof(int)*32);

	cudaMemcpy(output_h, output, sizeof(int)*32, cudaMemcpyDeviceToHost);

	printf("%s took %d clocks.\n", name, output_h[0]);
	printf("Each instruction takes %.2f clocks.\n\n", (float)output_h[0]/(128.0*128.0));

	cudaFree(output);
	free(output_h);
}

int main(){
	run("lds32");
	return 0;
}
