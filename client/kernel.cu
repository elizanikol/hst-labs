#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "kernel.h"
//#include "helper_functions.h"
//#include "helper_cuda.h"


const auto NumThreads = 1024;
const auto SecondsMultiplier = 0.001;


void gpuAssert(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout << "Cuda error: " << cudaGetErrorString(code) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void MaxElement(double* data, int* offsets, double* results, int resultsSize)
{
    const auto threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= resultsSize)
    {
        return;
    }

    double maxElement = data[offsets[threadId]];
    const auto rowSize = offsets[threadId + 1] - offsets[threadId];
    for (auto i = 1; i < rowSize; ++i)
    {
        auto tmp = data[offsets[threadId] + i];
        if (tmp > maxElement)
        {
            maxElement = tmp;
        }
    }
    
    results[threadId] = pow(maxElement, 2.0);
}


void flattenMatricesAndCalculateOffsets(double* data, const std::vector<std::vector<double>>& matrices, int* offsets)
{
    double* currentData = data;

    offsets[0] = 0;
    int* currentOffsets = offsets + 1;

    for (size_t i = 0; i < matrices.size(); ++i)
    {
        std::copy(matrices[i].begin(), matrices[i].end(), currentData);
        currentData += matrices[i].size();

        // calculate offsets
        const auto rowSize = static_cast<int>(std::sqrt(matrices[i].size()));
        std::vector<int> rowOffsets(rowSize, rowSize);
        rowOffsets[0] += *(currentOffsets - 1);
        std::partial_sum(rowOffsets.begin(), rowOffsets.end(), currentOffsets);
        currentOffsets += rowSize;
    }
}

void ArrayTo2DVector(double* results, std::vector<std::vector<double>>& maxSquaresVectors)
{
    double* current = results;
    for (auto& v : maxSquaresVectors)
    {
        std::copy(current, current + v.size(), v.begin());
        current += v.size();
    }
}

void prepareData(size_t& dataSize, size_t& offsetsSize, size_t& resultsSize,
    const std::vector<std::vector<double>>& matrices, std::vector<std::vector<double>>& maxSquaresVectors)
{
    for (size_t matrixId = 0; matrixId < matrices.size(); ++matrixId)
    {
        const auto matrixSize = matrices[matrixId].size();
        const auto rowSize = static_cast<int>(std::sqrt(matrixSize));

        dataSize += matrixSize;
        resultsSize += rowSize;

        maxSquaresVectors[matrixId].resize(rowSize);
    }
    offsetsSize = resultsSize + 1;
}

double CalculateMaxElementsSquare(const std::vector<std::vector<double>>& matrices, std::vector<std::vector<double>>& maxSquaresVectors)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t dataSize = 0;
    size_t offsetsSize = 0;
    size_t resultsSize = 0;
    prepareData(dataSize, offsetsSize, resultsSize, matrices, maxSquaresVectors);

    double* data = (double*)malloc(sizeof(double) * dataSize);
    int* offsets = (int*)malloc(sizeof(int) * offsetsSize);
    double* results = (double*)malloc(sizeof(double) * resultsSize);

    double* deviceData;
    int* deviceOffsets;
    double* deviceResults;
    gpuAssert(cudaMalloc((void**)&deviceData, sizeof(double) * dataSize));
    gpuAssert(cudaMalloc((void**)&deviceOffsets, sizeof(int) * offsetsSize));
    gpuAssert(cudaMalloc((void**)&deviceResults, sizeof(double) * resultsSize));

    flattenMatricesAndCalculateOffsets(data, matrices, offsets);

    cudaMemcpy(deviceData, data, sizeof(double) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOffsets, offsets, sizeof(int) * offsetsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceResults, results, sizeof(double) * resultsSize, cudaMemcpyHostToDevice);

    const auto gridSize = static_cast<int>((resultsSize + NumThreads - 1) / NumThreads);
    const auto blockSize = static_cast<int>(NumThreads);

   /* for (int i = 0; i < dataSize; ++i)
    {
        std::cout << data[i] << ' ';
    }
    std::cout << std::endl;

    for (int i = 0; i < offsetsSize; ++i)
    {
        std::cout << offsets[i] << ' ';
    }
    std::cout << std::endl;*/

    cudaEventRecord(start);
    MaxElement<<<gridSize, blockSize>>>(deviceData, deviceOffsets, deviceResults, static_cast<int>(resultsSize));
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    gpuAssert(cudaEventSynchronize(stop));

    gpuAssert(cudaMemcpy(results, deviceResults, sizeof(double) * resultsSize, cudaMemcpyDeviceToHost));

    /*for (int i = 0; i < resultsSize; ++i)
    {
        std::cout << results[i] << ' ';
    }
    std::cout << std::endl;*/
    ArrayTo2DVector(results, maxSquaresVectors);
    gpuAssert(cudaFree(deviceData));
    gpuAssert(cudaFree(deviceResults));
    gpuAssert(cudaFree(deviceOffsets));

    free(data);
    free(offsets);
    free(results);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds * SecondsMultiplier;
}
