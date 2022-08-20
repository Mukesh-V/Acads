// Mukesh V
// ME18B156

#include <stdio.h>
#include <math.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#define BLOCK_SIZE 256

using namespace std;

__global__ void sjf_assign(int m, int n, int *indx, int *result){
    int threadId = (blockIdx.x * blockDim.x + threadIdx.x)%n;
    int jobId = indx[threadId];
    
    result[jobId] = threadId%m;
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking jobs as input	
    thrust::host_vector<int> H_jobs(n);
    thrust::device_vector<int> D_jobs(n), D_indx(n);
    for ( int i=0; i< n; i++ )  {
        fscanf( inputfilepointer, "%d", &H_jobs[i] );
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
    
	thrust::sequence(D_indx.begin(), D_indx.end());
    D_jobs = H_jobs;
    thrust::sort_by_key(D_jobs.begin(), D_jobs.end(), D_indx.begin());

    thrust::host_vector<int> ind(n), j(n);
    ind = D_indx; j = D_jobs;

    int *d_indx = thrust::raw_pointer_cast(D_indx.data());
    int *d_jobs = thrust::raw_pointer_cast(D_jobs.data());
    int *execTimes, *result, *i;
    cudaHostAlloc(&result, n * sizeof(int), 0);
    cudaMalloc(&execTimes, m * sizeof(int));
    cudaMalloc(&i, sizeof(int));
    
    int n_blocks = ceil((float) n/BLOCK_SIZE);
    int block = (n > BLOCK_SIZE)? BLOCK_SIZE : n; 
    sjf_assign<<<n_blocks, block>>>(m, n, d_indx, result);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
}

