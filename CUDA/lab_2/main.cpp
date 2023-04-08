#include <iostream>
#include <stdio.h>
#include <time.h>
#define THREADS_PER_BLOCK 1024

using namespace std;

__global__ void BubbleMove(double* array, int N, int step);

__global__ void DotProduct(double const *deviceLHS, double const *deiceRHS, double *deviceRES, int N);



void BubbleSortCUDA(double *array_host, int N, int blockSize);

void PrintArray(const double* array, const int n);

void BubbleSortNoCUDA(double *array, int N);

bool IsSorted(double *array, int N);

void measureNoCUDA(int N);

void measureCUDA(int N);

double CalculateDotProductNoCuda(double *lhs, double *rhs, int N);

double CalculateDotProductCuda(double *hostLHS, double *hostRHS, int N);


int main (int argc, char *  argv []){
    srand(static_cast <unsigned> (time(0)));
    // int n = atoi(argv[1]);
    // double *a = (double *)malloc(n * sizeof(double));
    // for (int i = 0; i < n; i++) {
    //     a[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    // }
    // double *b = (double *)malloc(n * sizeof(double));
    // for (int i = 0; i < n; i++) {
    //     b[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    // }
    // PrintArray(a, n);
    // PrintArray(b, n);
    // cout << "CPU:  " << CalculateDotProductNoCuda(a, b, n) << endl;
    // cout << "CUDA:  " << CalculateDotProductCuda(a, b, n) << endl;

    //cout << "CPU:  " << CalculateDotProductNoCuda(a, b, n) << endl;//"   CUDA:  " << CalculateDotProductCuda(a, b, n) << endl;
    // for(int i = 4; i < 2000; i *= 2){
    //     cout << "N = " << i << "  Single: " << measureNoCUDA(i) << "   CUDA: " << measureCUDA(i, 32) << endl;
    // }

    cout << "CPU:" << endl;
    cout << "N      Time1     Time2      Time3" << endl;
    for (int n = 2; n < 1000000; n*=2 ){
        cout<< n << "   ";
        for (int i = 0; i < 3; i++){
            measureNoCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }

    cout << "GPU:" << endl;
    cout << "N      Time1     Time2      Time3" << endl;
    for (int n = 2; n < 1000000; n*=2 ){
        cout<< n << "   ";
        for (int i = 0; i < 3; i++){
            measureCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return 0;
}

__global__ void BubbleMove(double *array, int N, int step){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (N-1)) {
        if (step-2>=idx){
            if (array[idx] < array[idx + 1]){
                double helper = array[idx];
                array[idx] = array[idx + 1];
                array[idx + 1] = helper;
            }
        }
    }
}

__global__ void DotProduct(double const *deviceLHS, double const *deviceRHS, double *deviceRES, int N){
    // Shared array for threads in a block
    __shared__ double helper[THREADS_PER_BLOCK];

    // Get thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute dot products for all elemnts belonging to this thread
    double sum = 0;
    while (idx < N) {
        sum += deviceLHS[idx] * deviceRHS[idx];
        idx += blockDim.x * gridDim.x;
    }
    // ... and write into shared array
    helper[threadIdx.x] = sum; 

    __syncthreads(); 

    // Calculate results for current Block (reduction sum)
    // Works if THREADS_PER_BLOCK is power of 2
    int index = blockDim.x / 2;
    while (index != 0){
        if (threadIdx.x < index) {
            helper[threadIdx.x] += helper[threadIdx.x + index];
        }
            
        __syncthreads();

        index = index / 2;
    }

    // Save the results for block
    if (threadIdx.x == 0) {
        deviceRES[blockIdx.x] = helper[0];
    }
}

// Bubble sort on GPU CUDA
void BubbleSortCUDA(double *array_host, int N, int blockSize){
    double *array_device; 
    cudaMalloc((void **)&array_device, N * sizeof(double));
    cudaMemcpy(array_device, array_host, N*sizeof(double), cudaMemcpyHostToDevice);
    int nblocks = N / blockSize + 1;
    for (int step = 0; step <= N + N; step++) {
        // Step of bubble sort
        BubbleMove<<<nblocks, blockSize>>>(array_device, N, step);
        // Wait for all threads to finish changes
        //cudaThreadSynchronize();
        cudaDeviceSynchronize();
    }
    cudaMemcpy(array_host, array_device, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(array_device);
}

// Bubble sort on CPU in one thread
void BubbleSortNoCUDA(double *array, int N){
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N-i-1; j++) {
      if (array[j]<array[j + 1]){
        double helper = array[j];
        array[j] = array[j + 1];
        array[j + 1] = helper;
      }
    }
  }
}

// check if and array is sorted
bool IsSorted(double *array, int N){
    for (int i = 0; i < N-1; i++){
        if (array[i] < array[i+1]) {
            return false;
        }
    } 
    return true;
}

// measure time taken for all steps on CPU CUDA
void measureNoCUDA(int N) {
    //cout << N << "   ";
    double *array1 = (double *)malloc(N * sizeof(double));
    double *array2 = (double *)malloc(N * sizeof(double));
    clock_t start_time, end_time;
    float timeArray1Sorted, timeArray2Sorted, timeDotProduct;
    double resCPU = 0.0;
    for (int i = 0; i < N; i++) {
        array1[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    start_time = clock();
    BubbleSortNoCUDA(array1, N);
    end_time = clock();

    timeArray1Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // check that array is sorted
    if (IsSorted(array1, N)){
        //cout << timeArray1Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 1 NOT SORTED" << endl;
        return;
    }

    start_time = clock();
    BubbleSortNoCUDA(array2, N);
    end_time = clock();

    timeArray2Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // check that array is sorted
    if (IsSorted(array2, N)){
        //cout << timeArray2Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 2 NOT SORTED" << endl;
        return;
    }



    start_time = clock();
    resCPU = CalculateDotProductNoCuda(array1, array2, N);
    end_time = clock();

    timeDotProduct = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    //cout << timeDotProduct << "   ";


    free(array1);
    free(array2);
    cout << timeArray1Sorted + timeArray2Sorted + timeDotProduct;
}

// measure time taken for all steps on GPU CUDA
void measureCUDA(int N){
    //cout << N << "   ";
    double *array1 = (double *)malloc(N * sizeof(double));
    double *array2 = (double *)malloc(N * sizeof(double));
    clock_t start_time, end_time;
    float timeArray1Sorted, timeArray2Sorted, timeDotProduct;
    double resGPU = 0.0;
    double resCPU = 0.0;
    for (int i = 0; i < N; i++) {
        array1[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    start_time = clock();
    BubbleSortCUDA(array1, N, THREADS_PER_BLOCK);
    end_time = clock();

    timeArray1Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // check that array is sorted
    if (IsSorted(array1, N)){
        //cout << timeArray1Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 1 NOT SORTED" << endl;
        return;
    }

    start_time = clock();
    BubbleSortCUDA(array2, N, THREADS_PER_BLOCK);
    end_time = clock();

    timeArray2Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // check that array is sorted
    if (IsSorted(array2, N)){
        //cout << timeArray2Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 2 NOT SORTED" << endl;
        return;
    }



    start_time = clock();
    resGPU = CalculateDotProductCuda(array1, array2, N);
    end_time = clock();

    timeDotProduct = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // compare to the CPU result
    resCPU = CalculateDotProductNoCuda(array1, array2, N);
    if (abs(resCPU - resGPU) < 1e-2){
        //cout << timeDotProduct << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, DOT PRODUCT DOESN'T MATCH CPU" << endl;
        return;
    }


    free(array1);
    free(array2);

    cout << timeArray1Sorted + timeArray2Sorted + timeDotProduct;
}

// Print double array into console
void PrintArray(const double* array, const int n){
    for(int i = 0; i < n; i++){
        cout << array[i] <<  ' ';
    }
    cout << endl;
}

// Dot product of two arrays on CPU
double CalculateDotProductNoCuda(double *lhs, double *rhs, int N){
    double result = 0.0;
    for (int i = 0; i < N; i++){
        result += lhs[i] * rhs[i];
    }
    return result;
}

// Dot product of two arrays on GPU CUDA
double CalculateDotProductCuda(double *hostLHS, double *hostRHS, int N){
    double *hostInterRes; // helper for intermediate results
    double *deviceLHS, *deviceRHS, *deviceRES;

    // Thread blocks per Grid
    int gridDim = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Create vectors
    size_t interResSize = gridDim * sizeof(double);
    hostInterRes = (double*)malloc(interResSize);
    cudaMalloc(&deviceLHS, N * sizeof(double));
    cudaMalloc(&deviceRHS, N * sizeof(double));
    cudaMalloc(&deviceRES, interResSize);	

    // Arrays to gpu
    cudaMemcpy(deviceLHS, hostLHS, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRHS, hostRHS, N * sizeof(double), cudaMemcpyHostToDevice);

 
    DotProduct<<<gridDim, THREADS_PER_BLOCK>>>(deviceLHS, deviceRHS, deviceRES, N);
    
    // Array to cpu
    cudaMemcpy(hostInterRes, deviceRES, interResSize, cudaMemcpyDeviceToHost);

   
    cudaFree(deviceLHS);
    cudaFree(deviceRHS);
    cudaFree(deviceRES);

    // Final reduction
    double res = 0;
    for (int i = 0; i < gridDim; i++){
        res += hostInterRes[i];
    }

    return res;
}