
#include <vector>
#include <iostream>

#include <cuda/cmath>

__global__ void addVectors(const float* vectorA, const float* vectorB, float* resultingVector, int size)
{
    // Calculate the global thread ID 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // If the thread ID is within the bounds of the vectors, perform the addition
    // Since this index is unique for each thread, we can safely write to the resulting vector without
    // worrying about race conditions.
    if (i < size)
    {
        resultingVector[i] = vectorA[i] + vectorB[i];
    }
}

int main()
{
    // Example vectors to add
    std::vector<float> vectorA = { 1.0f, 2.0f, 3.0f };
    std::vector<float> vectorB = { 4.0f, 5.0f, 6.0f };
    
    std::vector<float> resultingVector(vectorA.size());

    // Device pointers for the vectors
    float *d_vectorA, *d_vectorB, *d_resultingVector;

    // Allocate memory on the device using the device pointers
    // We allocate from the device pointers, to the size of the host vectors, multiplied by the size of the data type (float in this case).
    cudaMalloc(&d_vectorA, vectorA.size() * sizeof(float));
    cudaMalloc(&d_vectorB, vectorB.size() * sizeof(float));
    cudaMalloc(&d_resultingVector, resultingVector.size() * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_vectorA, vectorA.data(), vectorA.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB, vectorB.data(), vectorB.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel to add the vectors
    int threadsPerBlock = 256;

    // The number of blocks is calculated to ensure that we have enough threads to cover all elements in the vectors.
    // It is possible for the blocks and grids to be of higher dimensions (e.g., 2D or 3D), but for simplicity, we are using a 1D configuration here.
    int blocks = cuda::ceil_div(vectorA.size(), threadsPerBlock);

    // All threads within a thread block are executed on the same streaming multiprocessor, and can share data through shared memory.
    // The number of blocks is calculated to ensure that we have enough threads to cover all elements
    // This is an asynchronous call with respect to the host, meaning that the CPU can continue executing
    addVectors<<<blocks, threadsPerBlock>>>(d_vectorA, d_vectorB, d_resultingVector, vectorA.size());

    // Since kernel launches are asynchronous, we need to synchronize the device to ensure that the kernel has finished executing before we attempt to copy the results back to the host.
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(resultingVector.data(), d_resultingVector, resultingVector.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Resulting vector: ";
    for (const auto& element : resultingVector)    {
        std::cout << element << " ";
    }
    std::cout << "\n";

    cudaFree(d_vectorA);
    cudaFree(d_vectorB);
    cudaFree(d_resultingVector);
    return 0;
}