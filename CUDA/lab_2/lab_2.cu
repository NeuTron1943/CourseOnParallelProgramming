#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iomanip>
#define THREADS_PER_BLOCK 1024
#define MAX_DEPTH 16            // max depth for quick sort layers
#define INSERTION_SORT 32       // smaller than these - selection sort instead of quick

using namespace std;

__global__ void BubbleMove(double* array, int N, int step);

__global__ void DotProduct(double const *deviceLHS, double const *deiceRHS, double *deviceRES, int N);

__global__ void cdp_simple_quicksort(double* vec, int start, int end, int depth);

__device__ void selection_sort(double* data, int start, int end); 

void BubbleSortCUDA(double *array_host, int N, int blockSize);

void PrintArray(const double* array, const int n);

void BubbleSortNoCUDA(double *array, int N);

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
    for (int n = 2; n < 1000000; n*=2 ){
        cout<< setw(6) <<  n << "   ";
        for (int i = 0; i < 3; i++){
            measureNoCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }

    cout << "GPU:" << endl;
    cout << "N      Time1     Time2      Time3" << endl;
    //cout << "N      Time" << endl;
    for (int n = 2; n < 1000000; n*=2 ){
        cout<< setw(6) <<  n << "   ";
        for (int i = 0; i < 3; i++){
            measureCUDA(n);
            cout << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return 0;
}

// One step for bubble sort with CUDA
__global__ void BubbleMove(double *array, int N, int step){
    // Get thread ID
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check that thread belongs to array
    if (idx < (N-1)) {
        // Check that "bubble" moved enough for this thread to run and that there are no intersections
        if (step-2 >= idx  && (idx - step) % 2 == 0){
            // Swap elements if need
            if (array[idx] < array[idx + 1]){
                double helper = array[idx];
                array[idx] = array[idx + 1];
                array[idx + 1] = helper;
            }
        }
    }
}

// Thread code for dot product on CUDA
__global__ void DotProduct(double const *deviceLHS, double const *deviceRHS, double *deviceRES, int N){
    // Shared array for threads in a block
    __shared__ double helper[THREADS_PER_BLOCK];

    // Get thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute dot products for elements and write into shared array
    helper[threadIdx.x] = deviceLHS[idx] * deviceRHS[idx]; 

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
    // Create, allocate memory and copy array to device (GPU)
    double *array_device; 
    cudaMalloc((void **)&array_device, N * sizeof(double));
    cudaMemcpy(array_device, array_host, N*sizeof(double), cudaMemcpyHostToDevice);
    // Calculate needed number of blocks according to block size
    int nblocks = N / blockSize + 1;
    // N+N steps for all needed changes sure to be made
    for (int step = 0; step <= N + N; step++) {
        // Step of bubble sort
        BubbleMove<<<nblocks, blockSize>>>(array_device, N, step);
        // Wait for all threads to finish changes
        //cudaThreadSynchronize();
        cudaDeviceSynchronize();
    }
    // Copy array from device to host
    cudaMemcpy(array_host, array_device, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(array_device);
}

// Bubble sort on CPU in one thread
void BubbleSortNoCUDA(double *array, int N){
    // Repeat N times
    for (int i = 0; i < N; i++){
        // Move "bubble" and swap elements if needed
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
    // Iterate over all elemtnts
    for (int i = 0; i < N-1; i++){
        // ... and make sure that they are sorted properly
        if (array[i] < array[i+1]) {
            return false;
        }
    } 
    return true;
}

// measure time taken for all steps on CPU CUDA
void measureNoCUDA(int N) {
    // Create arrays
    double *array1 = (double *)malloc(N * sizeof(double));
    double *array2 = (double *)malloc(N * sizeof(double));
    // Create variables for saving time results
    clock_t start_time, end_time;
    float timeArray1Sorted, timeArray2Sorted, timeDotProduct;
    double resCPU = 0.0;
    // Fill arrays with random numbers
    for (int i = 0; i < N; i++) {
        array1[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    // Sort first array
    start_time = clock();
    BubbleSortNoCUDA(array1, N);
    end_time = clock();

    // Time taken
    timeArray1Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Check that array is sorted
    if (IsSorted(array1, N)){
        //cout << timeArray1Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 1 NOT SORTED" << endl;
        return;
    }

    // Sort second array
    start_time = clock();
    BubbleSortNoCUDA(array2, N);
    end_time = clock();

    // Time taken
    timeArray2Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Check that array is sorted
    if (IsSorted(array2, N)){
        //cout << timeArray2Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 2 NOT SORTED" << endl;
        return;
    }

    // Calculate dot product of sorted arrays
    start_time = clock();
    resCPU = CalculateDotProductNoCuda(array1, array2, N);
    end_time = clock();

    // Time taken
    timeDotProduct = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Free memory
    free(array1);
    free(array2);
    // Print time info
    cout << setw(10) << timeArray1Sorted + timeArray2Sorted + timeDotProduct;
}

// measure time taken for all steps on GPU CUDA
void measureCUDA(int N){
    // Create arrays
    double *array1 = (double *)malloc(N * sizeof(double));
    double *array2 = (double *)malloc(N * sizeof(double));
    // Create variables for saving time results
    clock_t start_time, end_time;
    float timeArray1Sorted, timeArray2Sorted, timeDotProduct;
    // Create variables for saving results of all calculations
    double resGPU = 0.0;
    double resCPU = 0.0;
    // Fill arrays with random numbers
    for (int i = 0; i < N; i++) {
        array1[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        array2[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    // Sort first array
    start_time = clock();
    BubbleSortCUDA(array1, N, THREADS_PER_BLOCK);
    // QuickSortCUDA(array1, N);                    
    end_time = clock();

    // Time taken
    timeArray1Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Check that array is sorted
    if (IsSorted(array1, N)){
        //cout << timeArray1Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 1 NOT SORTED" << endl;
        return;
    }

    // Sort second array
    start_time = clock();
    BubbleSortCUDA(array2, N, THREADS_PER_BLOCK);
    // QuickSortCUDA(array2, N);
    end_time = clock();

    // Time taken
    timeArray2Sorted = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Check that array is sorted
    if (IsSorted(array2, N)){
        //cout << timeArray2Sorted << "   ";
    }
    else {
        free(array1);
        free(array2);
        cout << "ERROR, ARRAY 2 NOT SORTED" << endl;
        return;
    }


    // Calculate dot product of sorted arrays
    start_time = clock();
    resGPU = CalculateDotProductCuda(array1, array2, N);
    end_time = clock();

    // Time taken
    timeDotProduct = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // Compare to the CPU result
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

    // Free memory
    free(array1);
    free(array2);
    // Print time info
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

// Selection sort for final recursion steps of quick sort
// only for threads
__device__ void selection_sort(double* data, int start, int end) {
    for (int i = start; i <= end; ++i) {
        double max_val = data[i];
        int max_idx = i;

        // Find the biggest value in the range [start, end].
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
    // if max depth is reached or array is too small launch selection sort
    if (depth >= MAX_DEPTH || end - start <= INSERTION_SORT) {
        selection_sort(vec, start, end);
        return;
    }

    double helper;

    // Make partitioning (like in serial mode)
    // taking first elemt as pivot point
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
        // looking for missplaced elements on the left of the pivot 
		while (vec[i] >= pivot) {
			i++;
		}
        // looking for missplaced elements on the right of the pivot 
		while (vec[j] < pivot) {
			j--;
		}
        // Swap pair of missplaced elements
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
    // Create, allocate memory and copy array to device (GPU)
    double *array_device; 
    cudaMalloc((void **)&array_device, nitems * sizeof(double));
    cudaMemcpy(array_device, vec, nitems*sizeof(double), cudaMemcpyHostToDevice);
    // Launch on device
    int left = 0;
    int right = nitems - 1;
    // std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<<1, 1>>>(array_device, left, right, 0);
    cudaDeviceSynchronize();
    // Copy array from device to host
    cudaMemcpy(vec, array_device, nitems*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(array_device);
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
        // looking for missplaced elements on the left of the pivot 
		while (vec[i] >= pivot) {
			i++;
		}
        // looking for missplaced elements on the right of the pivot 
		while (vec[j] < pivot) {
			j--;
		}
        // Swap pair of missplaced elements
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