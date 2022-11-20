#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>

#include "kernel.h"


const std::string InputDataFile{ "../Test.b" };
const std::string ResultsFile{ "../Results.b" };
const size_t MBMultiplier = 1024 * 1024;


int getFileSize(const std::string& path)
{
    std::ifstream file(path, std::ifstream::binary | std::ifstream::ate);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return file.tellg();
}

void ReadMatricesFromFile(const std::string& path, std::vector<std::vector<double>>& matrices)
{
    std::ifstream file(path, std::ifstream::binary);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t rowsNumber = 0;
    while (file.read(reinterpret_cast<char*>(&rowsNumber), sizeof(rowsNumber)))
    {
        const size_t matrixSize = rowsNumber * rowsNumber;
        std::vector<double> flatMatrix(matrixSize);
        if (!file.read(reinterpret_cast<char*>(&flatMatrix[0]), static_cast<std::streamsize>(matrixSize) * sizeof(double)))
        {
            std::cerr << "Reading from file: " << path << " failed\n";
            std::exit(EXIT_FAILURE);
        }
        matrices.push_back(flatMatrix);
    }

    file.close();
}

void SaveResults(const std::string& path, const std::vector<std::vector<double>>& vectors, const double elapsedSeconds)
{
    std::ofstream file(path, std::ofstream::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (auto& v : vectors)
    {
        file.write(reinterpret_cast<const char*>(&v[0]), static_cast<std::streamsize>(v.size()) * sizeof(double));
    }

    const auto fileSizeInBytes = getFileSize(InputDataFile);
    file << "\nElapsed time: " << elapsedSeconds << "s for data of size: " << fileSizeInBytes
        << " bytes (" << fileSizeInBytes / MBMultiplier << ") MB.";

    file.close();
    if (!file)
    {
        std::cerr << "Writing to file: " << path << " failed\n";
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    std::vector<std::vector<double>> matrices;
    ReadMatricesFromFile(InputDataFile, matrices);

    std::vector<std::vector<double>> maxSquaresVectors(matrices.size());
    const auto elapsedMilliseconds = CalculateMaxElementsSquare(matrices, maxSquaresVectors);

    SaveResults(ResultsFile, maxSquaresVectors, elapsedMilliseconds);

    const auto fileSizeInBytes = getFileSize(InputDataFile);
    std::cout << "elapsed time: " << elapsedMilliseconds << "s for data of size: "
        << fileSizeInBytes << " bytes (" << fileSizeInBytes / MBMultiplier << ") MB.\n";
}
