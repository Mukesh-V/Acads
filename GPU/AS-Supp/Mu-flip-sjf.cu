// Mukesh V
// ME18B156

#include <stdio.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define BLOCK_SIZE 256

__global__ void sequenceAssign(int length, int *data){
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadId < length)
    data[threadId] = threadId;
}

__global__ void flip_sjf(int m, int n, int *jobs, int *indx, int *result, int *execTimes){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; 
    int step = threadId%m;
    if(threadId < m)
      execTimes[threadId] = 0;

    if(threadId < n){
      int jobId = indx[threadId];
      result[jobId] = threadId%m;
      // For even batch of jobs, first laptop will receive shortest job available
      // For odd  batch of jobs, first laptop will receive longest  job available
      int batch = (int) threadId/m;
      if(batch%2==1){
      	if( (batch+1)*m < n )
        	step = (m-1) - threadId%m;
        else
        	step = (n-1) - batch*m - threadId%m;
      }
      atomicAdd(&execTimes[threadId%m], jobs[batch*m + step]);
    }
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
    int jobTotal = 0;
    for ( int i=0; i< n; i++ ){
        fscanf( inputfilepointer, "%d", &H_jobs[i] );
        jobTotal += H_jobs[i];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    int *d_indx = thrust::raw_pointer_cast(D_indx.data());
    int *d_jobs = thrust::raw_pointer_cast(D_jobs.data());
    int *execTimes, *result;

    cudaHostAlloc(&result, n * sizeof(int), 0);
    cudaHostAlloc(&execTimes, m * sizeof(int), 0);

    int n_blocks = ceil((float) n/BLOCK_SIZE);
    int block = (n > BLOCK_SIZE)? BLOCK_SIZE : n; 

    //==========================================================================================================
    
	sequenceAssign<<<n_blocks, block>>>(n, d_indx);
    D_jobs = H_jobs;
    thrust::sort_by_key(D_jobs.begin(), D_jobs.end(), D_indx.begin());
    flip_sjf<<<n_blocks, block>>>(m, n, d_jobs, d_indx, result, execTimes);

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
    for ( int i=0; i<n; i++ )
        fprintf( outputfilepointer, "%d ", result[i]);
    fprintf( outputfilepointer, "\n");
    int execTotal = 0;
    for ( int i=0; i<m; i++ ){
      execTotal += execTimes[i];
        fprintf( outputfilepointer, "%d ", execTimes[i]);
    }
    // fprintf( outputfilepointer, "\n");
    // fprintf( outputfilepointer, "%d %d", jobTotal, execTotal);
    
    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
}
