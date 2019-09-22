#include <mma.h>

using namespace nvcuda;

__global__ 
void wmma(half *a, half *b, half* c){
	__shared__ half a_buffer[16*16];
	__shared__ half b_buffer[16*16];

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

	wmma::load_matrix_sync(a_frag, a_buffer, 18);
	wmma::load_matrix_sync(b_frag, b_buffer, 16);

	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

	wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_col_major);
}

