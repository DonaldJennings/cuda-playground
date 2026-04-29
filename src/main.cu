#include <iostream>

__global__ void hello_kernel()
{
    printf("Hello from GPU!\n");
}

int main()
{
    hello_kernel<<<1, 1>>>();

    cudaDeviceSynchronize();

    std::cout << "Done\n";
}