#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iomanip>
#define THREADS_PER_BLOCK 1024
#define MAX_DEPTH 16            // max depth for quick sort layers
#define INSERTION_SORT 32       // smaller than these - selection sort instead of quick

using namespace std;


__global__ void DotProduct(double const *deviceLHS, double const *deiceRHS, double *deviceRES, int N);

__global__ void cdp_simple_quicksort(double* vec, int start, int end, int depth);


void PrintArray(const double* array, const int n);

bool IsSorted(double *array, int N);

void measureNoCUDA(int N);

void measureCUDA(int N);

double CalculateDotProductNoCuda(double *lhs, double *rhs, int N);

double CalculateDotProductCuda(double *hostLHS, double *hostRHS, int N);

int MakePartition(double* vec, int start, int end);

void QuickSortCPU(double* vec, int start, int end);

void QuickSortCUDA(double* vec, unsigned int nitems);


int main (int argc, char *  argv []){
    srand(static_cast <unsigned> (time(0)));

    cout << "CPU:" << endl;
    cout << "N      Time1     Time2      Time3" << endl;
    for (int n = 2; n < 2000000; n*=2 ){
        cout<<setw(6) << n << "   ";
        for (int i = 0; i < 3; i++){
            measureNoCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }

    cout << "GPU:" << endl;
    cout << "N      Time1     Time2      Time3" << endl;
    //cout << "N      Time1" << endl;
    for (int n = 2; n < 2000000; n*=2 ){
        cout<< setw(6) << n << "   ";
        for (int i = 0; i < 3; i++){
            measureCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return 0;
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
        double value = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        if (value > 1e-7){
            array1[i] = value + 0.1;
        }
        value = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        if (value > 1e-7){
            array2[i] = value + 0.1;
        }
        //array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    start_time = clock();
    QuickSortCPU(array1, 0, N-1);
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
    QuickSortCPU(array2, 0, N-1);
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
    cout << setw(10) << timeArray1Sorted + timeArray2Sorted + timeDotProduct;
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
        double value = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        if (value > 1e-7){
            array1[i] = value + 0.1;
        }
        value = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        if (value > 1e-7){
            array2[i] = value + 0.1;
        }
        //array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    double *arrayCP = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++){
        arrayCP[i] = array1[i];
    }
    //cout << array1 << endl;
    start_time = clock();
    //BubbleSortCUDA(array1, N, THREADS_PER_BLOCK);
    QuickSortCUDA(array1, N);
    //run_qsort(array1, N);
    end_time = clock();
    //cout << array1 << endl;

    timeArray1Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // check that array is sorted
    if (IsSorted(array1, N)){
        //cout << timeArray1Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 1 NOT SORTED" << endl;
        //PrintArray(array1, N);
        //PrintArray(arrayCP, N);
        throw exception();
        return;
    }

    start_time = clock();
    //BubbleSortCUDA(array2, N, THREADS_PER_BLOCK);
    QuickSortCUDA(array2, N);
    //run_qsort(array2, N);
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

    cout << setw(10) << timeArray1Sorted + timeArray2Sorted + timeDotProduct;
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

// Making partition in vector for quick sort algorithm
int MakePartition(double* vec, int start, int end){
	// Taking first element as pivot point
	double pivot = vec[start];

	// Finding corret position of pivot element
	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (vec[i] >= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
	swap(vec[pivotIndex], vec[start]);

	// Now pivot element is on its true position
	// and we need to place elements greater than pivot on the left and less on the right
	int i = start;
	int j = end;

	// Number of missplaced elements is even, so we will use pair swaps
	while (i < pivotIndex && j > pivotIndex) {

		while (vec[i] >= pivot) {
			i++;
		}

		while (vec[j] < pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
			swap(vec[i++], vec[j--]);
		}
	}

	return pivotIndex;
}

// QuickSort on CPU, main sorting algorithm
void QuickSortCPU(double* vec, int start, int end){
	// base of the recursion
	if (start >= end)
		return;

	// partitioning the array
	int p = MakePartition(vec, start, end);

	// Sorting the left part
	QuickSortCPU(vec, start, p - 1);

	// Sorting the right part
	QuickSortCPU(vec, p + 1, end);
}


__device__ void selection_sort(double* data, int start, int end) {
    for (int i = start; i <= end; ++i) {
        double max_val = data[i];
        int max_idx = i;

        // Find the smallest value in the range [start, end].
        for (int j = i + 1; j <= end; ++j) {
            double val_j = data[j];

            if (val_j > max_val) {
                max_idx = j;
                max_val = val_j;
            }
        }

        // Swap the values.
        if (i != max_idx) {
            data[max_idx] = data[i];
            data[i] = max_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(double* vec, int start, int end, int depth) {
    if (depth >= MAX_DEPTH || end - start <= INSERTION_SORT) {
        selection_sort(vec, start, end);
        return;
    }

    double helper;

    // Make partitioning (like in serial mode)
    double pivot = vec[start];
	// Finding corret position of pivot element
	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (vec[i] >= pivot)
			count++;
	}
	// Giving pivot element its correct position
	int pivotIndex = start + count;
	//swap(vec[pivotIndex], vec[start]);
    helper = vec[pivotIndex];
    vec[pivotIndex] = vec[start];
    vec[start] = helper;

	// Now pivot element is on its true position
	// and we need to place elements greater than pivot on the left and less on the right
	int i = start;
	int j = end;

	// Number of missplaced elements is even, so we will use pair swaps
	while (i < pivotIndex && j > pivotIndex) {

		while (vec[i] >= pivot) {
			i++;
		}

		while (vec[j] < pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
			//swap(vec[i++], vec[j--]);
            helper = vec[i];
            vec[i] = vec[j];
            vec[j] = helper;
            i++;
            j--;
		}
	}

    // Launch a new block to sort the left part.
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(vec, start, pivotIndex - 1, depth + 1);
    cudaStreamDestroy(s);


    // Launch a new block to sort the right part.
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(vec, pivotIndex + 1, end, depth + 1);
    cudaStreamDestroy(s1);

}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void QuickSortCUDA(double* vec, unsigned int nitems) {
    double *array_device; 
    cudaMalloc((void **)&array_device, nitems * sizeof(double));
    cudaMemcpy(array_device, vec, nitems*sizeof(double), cudaMemcpyHostToDevice);
    // Launch on device
    int left = 0;
    int right = nitems - 1;
    // std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<<1, 1>>>(array_device, left, right, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(vec, array_device, nitems*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(array_device);
}

