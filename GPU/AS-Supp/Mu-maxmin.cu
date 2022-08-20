// Mukesh V
// ME18B156

#include <stdio.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#define MAX_BLOCK_SIZE 1024

__device__ int findMin(int m, int* execTimes)
{   
    int min = execTimes[0];
    int index = 0;
    for(int i = 1; i < m; i++){
      if(execTimes[i] < min){  
        min = execTimes[i];
        index = i;
      }
    }
    return index;
}
__global__ void max_min(int m, int n, int *indx, int *jobs, int *execTimes, int *i, int *result){
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  execTimes[threadId] = 0;
  int index, allocation;

  while(*i>=0 && threadId<m){
    index = indx[*i]; allocation = findMin(m, execTimes);
    __syncthreads();
    if(threadId == allocation){
      result[index] = threadId;
      execTimes[threadId] += jobs[*i];
      atomicAdd(i, -1);
    } 
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
    for ( int i=0; i< n; i++ )  {
        fscanf( inputfilepointer, "%d", &H_jobs[i] );
    }
    int *d_indx = thrust::raw_pointer_cast(D_indx.data());
    int *d_jobs = thrust::raw_pointer_cast(D_jobs.data());
    int *d_execTimes, *d_result, *i;

    cudaMalloc(&d_execTimes, m * sizeof(int));
    cudaMalloc(&d_result, n * sizeof(int));
    cudaHostAlloc(&i, sizeof(int), 0);
    
    float milliseconds = 0;
    int n_blocks, block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    //==========================================================================================================

    thrust::device_ptr<int> jobs_dptr(d_jobs), indx_dptr(d_indx);
    thrust::sequence(indx_dptr, indx_dptr+n);
    D_jobs = H_jobs;
    thrust::sort_by_key(jobs_dptr, jobs_dptr+n, indx_dptr);

    *i = n-1;
    n_blocks = ceil((float) m/MAX_BLOCK_SIZE);
    block_size = (m > MAX_BLOCK_SIZE)? MAX_BLOCK_SIZE : m;
    max_min<<<n_blocks, block_size>>>(m, n, d_indx, d_jobs, d_execTimes, i, d_result);
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
    int *h_result = (int *) malloc(n*sizeof(int));
    cudaMemcpy(h_result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", h_result[i]);
    }
    fprintf( outputfilepointer, "\n");

    int *h_execTimes = (int *) malloc(m *sizeof(int));
    cudaMemcpy(h_execTimes, d_execTimes, m*sizeof(int), cudaMemcpyDeviceToHost);
    for ( int i=0; i<m; i++ )  {
        fprintf( outputfilepointer, "%d ", h_execTimes[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
}

