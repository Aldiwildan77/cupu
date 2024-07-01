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
  int *a;
  int *b;
  int *c;

  int size = N * sizeof(int);

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  fill_array(a);
  fill_array(b);
  fill_array(c);

  host_add(a, b, c);
  print_output(a, b, c);

  free(a);
  free(b);
  free(c);

  return 0;
}

// $ nvcc -o vector_addiction vector_addiction.cu && ./vector_addiction | tee vector_addiction.txt
