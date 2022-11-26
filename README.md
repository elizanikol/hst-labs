# HST-MPI+Cuda

## Бизнес логика:
Сформировать M результирующих векторов как квадрат максимального значения по каждой строке M исходных квадратных матриц.

## Вариант исполнения:
Реализация через MPI и Cuda.

## Особенности исполнения:
Считывание данных происходит из бинарного файла;
Данные генерируются утилитой, принимающей в качестве параметров размер данных для обработки в мегабайтах и имя файла;
Программа выполняет бизнес-логику и записывает результат в выходной файл;
В конце файла с результатами сохраняется информация о времени выполнения вычислений и размере обработанных данных.

## Описание алгоритма выполнения бизнес-логики:
1. Для генерации исходных данных используется функция ```GenerateMatricesAndSaveThemToFile()```. 
С помощью ```std::default_random_engine``` определяется количество строк очередной квадратной матрицы, которая затем заполняется произвольными значениями типа double в диапазоне [0, 10] и записывается в бинарный файл ```InputDataFile```. 
Генерация матриц заканчивается после того, как размер файла с данными превысит размер, введенный пользователем.

2. После этого запускается программа-клиент, считывающая сгенерированные матрицы из ```InputDataFile```. В функции ```CalculateMaxElementsSquare()``` над исходными данными выполняются вычисления: для каждой строки i-ой матрицы вычисляется максимальное значение, которое возводится в квадрат и записывается в результирующий i-ый вектор.
3. Определяется время выполнения вычислений, которое записывается в бинарный файл ```ResultsFile``` вместе с результатами вычислений.

## Описание логики распараллеливания:
В главном потоке, имеющем rank = MasterID (0), выполняется чтение исходных данных из файла и распределение их между остальными процессами. Количество матриц, передаваемое процессу, вычисляется с помощью функции:
```
size_t MatricesNumberForProcess(size_t processId, size_t matricesSize, size_t processesTotalNumber)
{
    const auto avgMatricesNumberPerProcess = matricesSize / processesTotalNumber;
    const auto remainder = matricesSize % processesTotalNumber;
    return avgMatricesNumberPerProcess + (processId < remainder);
}
```
При этом каждому процессу сначала предполагается назначить одинаковое количество матриц для обработки, а оставшиеся после этого матрицы по очереди добавляются по одной к каждому из первых процессов в списке.

Пересылка данных между мастером и остальными процессами осуществляется с помощью функций MPI_Send и MPI_Recv. При этом в главном процессе в переменных ```matricesNumberForEachProcess``` и ```matricesSizesForEachProcess``` сохраняется количество и размер пересылаемых матриц каждому процессу для повторного использования этих значений при сборе данных после выполнения вычислений.

Далее каждый процесс запускает ядро Cuda, в котором каждая строка группы матриц обрабатывается отдельным потоком.
Квадраты максимальных значений вычисляются в функции ```MaxElement```, для этого в память устройства загружаются сразу все исходные данные в ```deviceData``` и смещения строк каждой матрицы в массиве ```deviceOffsets```. Результаты вычислений записываются в массив ```deviceResults```. 
Распараллеливание осуществляется путем задания нужного числа нитей ```blockSize``` для одновременного выполнения вычислений (максимальное 1024 для видеокарты "Tesla K20Xm"), с последующим вычислением ```gridSize``` как ```(resultsSize + NumThreads - 1) / NumThreads```. Каждая нить вычисляет максимальный элемент одной строки матрицы, таким образом, общее число нитей равно суммарному количеству строк всех матриц в исходных данных.

Передача данных между хостом и устройством осуществляется с помощью функций ```cudaMemcpy(..., cudaMemcpyHostToDevice)``` и ```cudaMemcpy(..., cudaMemcpyDeviceToHost)```.
Для определения времени вычислений используется функция MPI_Wtime(), вызываемая из основного потока перед и после выполнения вычислений всеми процессами.

## График зависимости времени вычислений от размера исходных данных:
Результаты для cuda приведены при ```blockSize=1024```, количество процессов MPI - 4.
![Снимок экрана 2022-11-20 191851](https://user-images.githubusercontent.com/55412039/204084574-2b562fee-1560-413c-a4b5-6c6fc191a029.png)

Время вычислений при использовании MPI совместно с Cuda сильно увеличивается за счет необходимости копирования большого количества данных для выполнения вычислений сразу над всей группой матриц на GPU.
