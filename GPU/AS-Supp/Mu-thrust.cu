// Mukesh V
// ME18B156

#include <stdio.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#define MAX_BLOCK_SIZE 64

__global__ void jobAssign(int idx, int allocationIdx, int *jobs, int *execTimes){
  execTimes[allocationIdx] += jobs[idx];
}
__global__ void sequenceAssign(int l, int *data){
  unsigned int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < l)
    data[globalId] = globalId; 
}
__global__ void fillAssign(int l, int *data, int value){
  unsigned int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  if(globalId < l)
    data[globalId] = value; 
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
    thrust::host_vector<int> H_jobs(n), H_indx(n), H_execTimes(m);
    thrust::device_vector<int> D_jobs(n), D_indx(n), D_execTimes(m);
    for ( int i=0; i< n; i++ )  {
        fscanf( inputfilepointer, "%d", &H_jobs[i] );
    }
    int *d_indx = thrust::raw_pointer_cast(D_indx.data());
    int *d_jobs = thrust::raw_pointer_cast(D_jobs.data());
    int *d_execTimes = thrust::raw_pointer_cast(D_execTimes.data());;
    
    float milliseconds = 0;
    int *result = (int *) malloc(n*sizeof(int));
    int allocationIdx, job = -1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    //==========================================================================================================
    

    thrust::device_ptr<int> jobs_dptr(d_jobs), indx_dptr(d_indx), exec_dptr(d_execTimes);

    int n_blocks, block_size;
    n_blocks = ceil((float) n/MAX_BLOCK_SIZE);
    block_size = (n > MAX_BLOCK_SIZE)? MAX_BLOCK_SIZE : n;
    sequenceAssign<<<n_blocks, block_size>>>(n, d_indx);

    n_blocks = ceil((float) m/MAX_BLOCK_SIZE);
    block_size = (m > MAX_BLOCK_SIZE)? MAX_BLOCK_SIZE : m;
    fillAssign<<<n_blocks, block_size>>>(m, d_execTimes, 0);

    D_jobs = H_jobs;
    thrust::sort_by_key(jobs_dptr, jobs_dptr+n, indx_dptr);
    H_indx = D_indx;

    for(int i=n-1; i>=0; i--){
      thrust::device_vector<int>::iterator iter = thrust::min_element(exec_dptr, exec_dptr + m);
      allocationIdx = iter - D_execTimes.begin();

      jobAssign<<<1,1>>>(i, allocationIdx, d_jobs, d_execTimes);
      result[H_indx[i]] = allocationIdx;
    }

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
    fprintf( outputfilepointer, "\n");

    H_execTimes = D_execTimes;
    H_execTimes[H_indx[0]] += job;  
    for ( int i=0; i<m; i++ )  {
        fprintf( outputfilepointer, "%d ", H_execTimes[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
}