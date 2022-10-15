#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <omp.h>


using Matrix = std::vector<std::vector<double>>;
const std::string InputDataFile{ "../../Test.b" };
const std::string ResultsFile{ "../../Results.b" };
const size_t MBMultiplier = 1024 * 1024;
const size_t ThreadsNumber = 4;


void ReadMatricesFromFile(const std::string& path, std::vector<Matrix>& matrices)
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
        Matrix matrix(rowsNumber, std::vector<double>(rowsNumber));
        for (auto& row : matrix)
        {
            if (!file.read(reinterpret_cast<char*>(&row[0]), static_cast<std::streamsize>(rowsNumber) * sizeof(double)))
            {
                std::cerr << "Reading from file: " << path << " failed\n";
                std::exit(EXIT_FAILURE);
            }
        }
        matrices.push_back(matrix);
    }

    file.close();
}

template<class ForwardIt>
ForwardIt MaxElement(ForwardIt itFirst, ForwardIt itLast)
{
    if (itFirst == itLast)
    {
        return itLast;
    }

    auto itMax = itFirst;
    ++itFirst;
    for (; itFirst != itLast; ++itFirst)
    {
        if (*itMax < *itFirst)
        {
            itMax = itFirst;
        }
    }
    return itMax;
}

void CalculateMaxElementsSquare(const std::vector<Matrix>& matrices, std::vector<std::vector<double>>& maxSquaresVectors)
{
#pragma omp parallel for num_threads(ThreadsNumber)
    for (size_t i = 0; i < matrices.size(); ++i)
    {
        maxSquaresVectors[i].resize(matrices[i].size());
        for (size_t row = 0; row < matrices[i].size(); ++row)
        {
            const auto maxElementIt = MaxElement(std::begin(matrices[i][row]), std::end(matrices[i][row]));
            if (maxElementIt == std::end(matrices[i][row]))
            {
                std::cerr << "Empty row in square matrix\n";
                std::exit(EXIT_FAILURE);
            }
            maxSquaresVectors[i][row] = pow(*maxElementIt, 2.0);
        }
    }
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

    const auto fileSizeInBytes = std::filesystem::file_size(InputDataFile);
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
    std::vector<Matrix> matrices;
    ReadMatricesFromFile(InputDataFile, matrices);

    std::vector<std::vector<double>> maxSquaresVectors(matrices.size());
    auto start = omp_get_wtime();
    CalculateMaxElementsSquare(matrices, maxSquaresVectors);
    auto end = omp_get_wtime();
    double elapsedSeconds = end - start;

    SaveResults(ResultsFile, maxSquaresVectors, elapsedSeconds);

    const auto fileSizeInBytes = std::filesystem::file_size(InputDataFile);
    std::cout << "elapsed time: " << elapsedSeconds << "s for data of size: "
        << fileSizeInBytes << " bytes (" << fileSizeInBytes / MBMultiplier << ") MB.\n";
}
