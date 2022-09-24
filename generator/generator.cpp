#include <iostream>
#include <fstream>
#include <random>


const std::string InputDataFile{"../../Test.b"};
const size_t MBMultiplier = 1024 * 1024;
const size_t MaxRowsNumber = 50;
const size_t MinRowsNumber = 2;
const size_t MaxElementValue = 10;
const size_t MinElementValue = 0;


void GenerateMatricesAndSaveThemToFile(const std::string& path, const size_t fileSizeInMB)
{
    std::default_random_engine gen(std::random_device{}());
    std::uniform_int_distribution<size_t> distSize(MinRowsNumber, MaxRowsNumber);
    std::uniform_real_distribution<double> distElement(MinElementValue, MaxElementValue);

    std::ofstream file(path, std::ofstream::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t totalBytesWritten = 0;
    const auto fileSizeInBytes = fileSizeInMB * MBMultiplier;
    while (totalBytesWritten < fileSizeInBytes)
    {
        const size_t rowsNumber = distSize(gen);
        file.write(reinterpret_cast<const char*>(&rowsNumber), sizeof(rowsNumber));

        std::vector<double> matrix(static_cast<size_t>(std::pow(rowsNumber, 2)));
        std::generate(matrix.begin(), matrix.end(), [&gen, &distElement]() mutable { return distElement(gen); });
        file.write(reinterpret_cast<const char*>(&matrix[0]), static_cast<std::streamsize>(matrix.size()) * sizeof(double));
        totalBytesWritten += sizeof(rowsNumber) + matrix.size() * sizeof(double);
    }

    file.close();
    if (!file) 
    {
        std::cerr << "Writing to file: " << path << " failed\n";
        std::exit(EXIT_FAILURE);
    }
}


int main()
{
    std::cout << "Enter file size in MB" << std::endl;
    size_t fileSizeInMB = 0;
    std::cin >> fileSizeInMB;

    GenerateMatricesAndSaveThemToFile(InputDataFile, fileSizeInMB);
    std::cout << "Data successfully generated into: " << InputDataFile;
}
