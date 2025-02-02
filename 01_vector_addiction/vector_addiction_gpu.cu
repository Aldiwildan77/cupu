#include <stdio.h>
#include <stdlib.h>

#define N 512

// formula: Cx = Ax + Bx
void host_add(int *a, int *b, int *c)
{
  for (int idx = 0; idx < N; idx++)
  {
    c[idx] = a[idx] + b[idx];
  }
}

// formula: Cx = Ax + Bx, multi-block
__global__ void device_add(int *a, int *b, int *c)
{
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_array(int *data)
{
  for (int idx = 0; idx < N; idx++)
  {
    data[idx] = idx;
  }
}

void print_output(int *a, int *b, int *c)
{
  for (int idx = 0; idx < N; idx++)
  {
    printf("a[%d] + b[%d] = c[%d] --> %d + %d = %d\n", idx, idx, idx, a[idx], b[idx], c[idx]);
  }
}

int main(int argc, char const *argv[])
{
  // data source (host)
  int *a;
  int *b;
  int *c;

  // copy data (to device)
  int *d_a;
  int *d_b;
  int *d_c;

  int size = N * sizeof(int);

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  fill_array(a);
  fill_array(b);

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // 1 block 1 thread
  // d_c = d_a + d_b
  // N stand for number of blocks so the d_c will be N size and filled with the result of d_a + d_b
  device_add<<<N, 1>>>(d_a, d_b, d_c);

  // send d_c to host (c)
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  print_output(a, b, c);

  // free host mem
  free(a);
  free(b);
  free(c);

  // free device mem
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

// $ nvcc -o vector_addiction_gpu vector_addiction_gpu.cu && ./vector_addiction_gpu | tee vector_addiction_gpu.txt
