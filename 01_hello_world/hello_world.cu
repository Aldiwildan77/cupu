#include <stdio.h>
#include <stdlib.h>

// This function will be called from GPU as a kernel asynchronously
/*
  ex: print_from_gpu<<<1, 1>>>(); // print with 1 thread 1 block
  output:
    Hello World from GPU!
    from thread [0, 0]

  ex: print_from_gpu<<<3, 1>>>(); // print with 3 thread 1 block
  output:
    Hello World from GPU!
    Hello World from GPU!
    Hello World from GPU!
    from thread [0, 1]
    from thread [0, 0]
    from thread [0, 2]

  note: tanpa loop, thread akan dijalankan secara paralel
*/
__global__ void print_from_gpu(void)
{
  printf("Hello World from GPU!\n");
  printf("from thread [%d, %d]\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[])
{
  printf("Hello World from CPU!\n");
  print_from_gpu<<<1, 1>>>(); // print with 1 thread 1 block --> 1 thread
  print_from_gpu<<<1, 2>>>(); // print with 1 thread 2 block --> 2 threads
  print_from_gpu<<<2, 2>>>(); // print with 2 thread 2 block --> 4 threads
  print_from_gpu<<<3, 1>>>(); // print with 3 thread 1 block --> 3 threads
  cudaDeviceSynchronize();    // wait for GPU to finish, not necessary (try to remove this)
  return 0;
}

// int main(int argc, char const *argv[])
// {
//   printf("Hello World from CPU!\n");
//   print_from_gpu<<<1, 1>>>(); // print with 1 thread 1 block
//   cudaDeviceSynchronize(); // wait for GPU to finish
//   return 0;
// }

// nvcc -o hello_world hello_world.cu && ./hello_world > hello_world.txt