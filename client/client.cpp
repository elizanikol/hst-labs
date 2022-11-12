#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mpi.h>


const std::string InputDataFile{ "../../Test.b" };
const std::string ResultsFile{ "../../Results.b" };
const int MBMultiplier = 1024 * 1024;
const int MasterId = 0;


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

void CalculateMaxElementsSquare(const std::vector<std::vector<double>>& matrices, std::vector<std::vector<double>>& maxSquaresVectors)
{
    for (auto i = 0; i < static_cast<int>(matrices.size()); ++i)
    {
        const int rowsNumber = static_cast<int>(std::sqrt(matrices[i].size()));
        maxSquaresVectors[i].resize(rowsNumber);

        auto itRowStart = std::begin(matrices[i]);
        auto itRowEnd = itRowStart + rowsNumber;
        for (int row = 0; row < rowsNumber; ++row)
        {
            const auto maxElementIt = MaxElement(itRowStart, itRowEnd);
            if (maxElementIt == itRowEnd)
            {
                std::cerr << "Empty row in square matrix\n";
                std::exit(EXIT_FAILURE);
            }
            maxSquaresVectors[i][row] = pow(*maxElementIt, 2.0);
            itRowStart += rowsNumber;
            itRowEnd += rowsNumber;
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

size_t MatricesNumberForProcess(size_t processId, size_t matricesSize, size_t processesTotalNumber)
{
    const auto avgMatricesNumberPerProcess = matricesSize / processesTotalNumber;
    const auto remainder = matricesSize % processesTotalNumber;
    return avgMatricesNumberPerProcess + (processId < remainder);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    auto currentProcessRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcessRank);
    auto processesTotalNumber = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processesTotalNumber);

    std::vector<std::vector<double>> matrices;
    // stored on master for gathering result data:
    std::vector<size_t> matricesNumberForEachProcess(processesTotalNumber);
    std::vector<std::vector<size_t>> matricesSizesForEachProcess(processesTotalNumber);

    if (currentProcessRank == MasterId)
    {
        // 1. read initial data from file
        ReadMatricesFromFile(InputDataFile, matrices);

        // 2. distribute data between proccesses
        auto matricesNumberForMaster = MatricesNumberForProcess(MasterId, matrices.size(), processesTotalNumber);
        auto offset = matricesNumberForMaster;
        for (int processId = 1; processId < processesTotalNumber; ++processId)
        {
            // 2a. send number of matrices to processId
            const auto matricesNumber = MatricesNumberForProcess(processId, matrices.size(), processesTotalNumber);
            MPI_Send(&matricesNumber, 1, MPI_UNSIGNED, processId, 0, MPI_COMM_WORLD);

            matricesNumberForEachProcess[processId] = matricesNumber;
            matricesSizesForEachProcess[processId].resize(matricesNumber);

            // 2b. for each matrix in a group send first its size, then its data to processId
            for (size_t i = 0; i < matricesNumber; ++i)
            {
                const auto matrixSize = matrices[offset].size();
                matricesSizesForEachProcess[processId][i] = matrixSize;
                MPI_Send(&matrixSize, 1, MPI_UNSIGNED, processId, 0, MPI_COMM_WORLD);
                MPI_Send(&matrices[offset][0], matrixSize, MPI_DOUBLE, processId, 0, MPI_COMM_WORLD);
                ++offset;
            }
        }
        // 3. delete all of the matrices sent to other processes
        matrices.resize(matricesNumberForMaster);
    }
    else
    {
        // 4. On each slave process receive partial data from master
        size_t matricesNumber = 0;
        MPI_Recv(&matricesNumber, 1, MPI_UNSIGNED, MasterId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrices.resize(matricesNumber);
        for (size_t i = 0; i < matricesNumber; ++i)
        {
            size_t matrixSize = 0;
            MPI_Recv(&matrixSize, 1, MPI_UNSIGNED, MasterId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matrices[i].resize(matrixSize);
            MPI_Recv(&matrices[i][0], matrixSize, MPI_DOUBLE, MasterId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // 5. Calculate vectors of max squares in each process and profile execution time
    double start = 0;
    double end = 0;
    double elapsedSeconds = 0;

    std::vector<std::vector<double>> maxSquaresVectors(matrices.size());
    MPI_Barrier(MPI_COMM_WORLD);

    if (currentProcessRank == MasterId)
    {
        start = MPI_Wtime();
    }

    CalculateMaxElementsSquare(matrices, maxSquaresVectors);
    MPI_Barrier(MPI_COMM_WORLD);

    if (currentProcessRank == MasterId)
    {
        end = MPI_Wtime();
        elapsedSeconds = end - start;
    }
    else
    {
        for (const auto& v : maxSquaresVectors)
        {
            MPI_Send(&v[0], v.size(), MPI_DOUBLE, MasterId, 0, MPI_COMM_WORLD);
        }
    }

    // 6. Gather data from slaves on master
    if (currentProcessRank == MasterId)
    {
        for (int processId = 1; processId < processesTotalNumber; ++processId)
        {
            for (size_t i = 0; i < matricesNumberForEachProcess[processId]; ++i)
            {
                int squaresVectorSize = static_cast<int>(std::sqrt(matricesSizesForEachProcess[processId][i]));
                std::vector<double> v(squaresVectorSize);
                MPI_Recv(&v[0], squaresVectorSize, MPI_DOUBLE, processId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                maxSquaresVectors.push_back(v);
            }
        }

        SaveResults(ResultsFile, maxSquaresVectors, elapsedSeconds);

        const auto fileSizeInBytes = std::filesystem::file_size(InputDataFile);
        std::cout << "elapsed time: " << elapsedSeconds << "s for data of size: "
            << fileSizeInBytes << " bytes (" << fileSizeInBytes / MBMultiplier << ") MB.\n";
    }

    MPI_Finalize();
}
