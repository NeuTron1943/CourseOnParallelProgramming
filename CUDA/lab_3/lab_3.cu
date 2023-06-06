#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iomanip>
#define THREADS_PER_BLOCK 1024

using namespace std;


__global__ void SumEven(double* array, int N, double* deviceRes);


double SumEvenCUDA(double* hostArray, int N);

double SumEvenCPU(double* hostArray, int N);

void PrintArray(const double* array, const int n);

void measureNoCUDA(int N);

void measureCUDA(int N);


int main (int argc, char *  argv []){
    srand(static_cast <unsigned> (time(0)));

    cout << "CPU:" << endl;
    cout << "N        Time1     Time2      Time3" << endl;
    for (int n = 256; n < 1000000000; n*=2 ){
        cout<< setw(9) <<  n << "   ";
        for (int i = 0; i < 3; i++){
            measureNoCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }

    cout << "GPU:" << endl;
    cout << "N        Time1     Time2      Time3" << endl;
    //cout << "N      Time" << endl;
    for (int n = 256; n < 1000000000; n*=2 ){
        cout<< setw(9) <<  n << "   ";
        for (int i = 0; i < 3; i++){
            measureCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return 0;
}

// Thread code for sum of the elements in the even threads
__global__ void SumEven(double* array, int N, double* deviceRes){
    // shared array for threads in the block
    __shared__ double helper[THREADS_PER_BLOCK];

    // Get thread ID
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Create thread variable for storing sum
    double sum = 0.0;
    // If thread ID is even
    if (idx % 2 == 0 && idx < N){
        helper[threadIdx.x] = array[idx];
    }else{
        // ... or if odd sum equals zero
        helper[threadIdx.x] = 0.0;
    }
        
    __syncthreads(); 

    // Reduction
    int amountOfThreads = blockDim.x / 2;
    while (amountOfThreads > 0){
		if (threadIdx.x < amountOfThreads){
            helper[threadIdx.x] += helper[threadIdx.x + amountOfThreads];
		}

        __syncthreads();

		amountOfThreads /= 2;
	}

    // Save the results for block
    if (threadIdx.x == 0) {
        deviceRes[blockIdx.x] = helper[0];
    }
}

// Sum elements on even positions using CUDA
double SumEvenCUDA(double* hostArray, int N){
    // Make two arrays on device variables
    double *deviceArray, *deviceRes;
    // helper for intermediate results
    double *hostInterRes;
    // Variable to accumulate intermediate results 
    double res = 0.0;

    // Thread blocks per Grid
    int gridDim = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t interResSize = gridDim * sizeof(double);

    // Create vectors
    hostInterRes = (double*)malloc(interResSize);
    cudaMalloc(&deviceArray, N*sizeof(double));
    cudaMalloc(&deviceRes, interResSize);
    // Copy from host to device
    cudaMemcpy(deviceArray, hostArray, N * sizeof(double), cudaMemcpyHostToDevice);

    // Sum
    SumEven<<<gridDim, THREADS_PER_BLOCK>>>(deviceArray, N, deviceRes);

    // Copy from device to host
    cudaMemcpy(hostInterRes, deviceRes, interResSize, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(deviceArray);
    cudaFree(deviceRes);

    // Final reduction
    for (int i = 0; i < gridDim; i++){
        res += hostInterRes[i];
    }
    free(hostInterRes);
    return res;
}

// Sum elements on even positions on CPU
double SumEvenCPU(double* hostArray, int N){
    double result = 0.0;
    // iterate over half of the array size
    for (int i = 0; i < N / 2; i++){
        // ... end sum every second element
        result+= hostArray[2*i];
    }
    return result;
}

// measure time taken for all steps on CPU CUDA
void measureNoCUDA(int N){
    // Create array
    double *array = (double *)malloc(N * sizeof(double));
    // Create variables for saving time results
    clock_t start_time, end_time;
    float timeResult;
    double resCPU = 0.0;
    // Fill array with random numbers
    for (int i = 0; i < N; i++) {
        array[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    // Calculation for the array
    start_time = clock();
    resCPU = SumEvenCPU(array, N);
    end_time = clock();
    // Time taken
    timeResult = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Free memory
    free(array);
    // Print time info 
    cout << setw(10) << timeResult;
}

// measure time taken for algorithm on GPU CUDA
void measureCUDA(int N){
    // Create array
    double *array = (double *)malloc(N * sizeof(double));
    // Create variables for saving time results
    clock_t start_time, end_time;
    float timeResult;
    double resGPU = 0.0, resCPU = 0.0;
    // Fill array with random numbers
    for (int i = 0; i < N; i++) {
        array[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    // Calculation for the array
    start_time = clock();
    resGPU = SumEvenCUDA(array, N);
    end_time = clock();
    // Time taken
    timeResult = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // compare to the CPU result
    resCPU = SumEvenCPU(array, N);
    //cout << "CPU : " << resCPU << endl;
    //cout << "GPU : " << resGPU << endl;
    if (abs(resCPU - resGPU) < 1e-2){
        //cout << timeDotProduct << "   ";
    }
    else {
        free(array);
        cout << "ERROR, RESULT DOESN'T MATCH CPU" << endl;
        return;
    }

    // Free memory
    free(array);
    // Print time info
    cout << setw(10) << timeResult;
}

// Print double array into console
void PrintArray(const double* array, const int n){
    for(int i = 0; i < n; i++){
        cout << array[i] <<  ' ';
    }
    cout << endl;
}
