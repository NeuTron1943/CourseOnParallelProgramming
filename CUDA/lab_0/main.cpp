#include <iostream>

using namespace std;

int main (int argc, char *  argv []){
    int deviceCount;
    cudaDeviceProp devProp;
    cudaGetDeviceCount(&deviceCount);

    cout<< "Found devices: "<< deviceCount<<endl;

    for (int device = 0; device < deviceCount; device++){
        cudaGetDeviceProperties(&devProp, device);
        cout<<"Device: "<< device <<endl;
        cout<<"Compute capability: "<< devProp.major << '.' << devProp.minor<<endl;
        cout<<"Name: "<< devProp.name<<endl;
        cout<<"Total Global Memory: "<< devProp.totalGlobalMem<<endl;
        cout<<"Shared memory per block: "<< devProp.sharedMemPerBlock<<endl;
        cout<<"Registers per block: "<< devProp.regsPerBlock<<endl;
        cout<<"Warpsize: "<< devProp.warpSize<<endl;
        cout<<"Max threads per block: "<< devProp.maxThreadsPerBlock<<endl;
        cout<<"Total constant memory: "<< devProp.totalConstMem<<endl;

        cout << "Multiprocessors: " << devProp.multiProcessorCount << endl;
    }


    return 0;
}