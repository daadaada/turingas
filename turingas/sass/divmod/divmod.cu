__global__
void divmod(int *a, int *q, int *r, int *d){
	int tmp = a[0];
	/* q[0] = tmp/d[0]; */
	r[0] = tmp%d[0];
}

