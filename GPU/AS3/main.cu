// Mukesh V
// ME18B156

#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void dscheduler(int m, int n, int *execs, int *prior, int *result){
    int id = threadIdx.x;
    extern __shared__ int threadPriors[];                                       // stores priorities of threads

    __shared__ int freeId, queueLock, time;                                     // stores the free thread minimal id, index of popped task, time and if queue is locked.
    __shared__ bool locked;
    if (id == 0) {queueLock = 0; time = 0; locked = false;}
    threadPriors[id] = -1;
    
    int tempId=0, endTime=0;
    do{
        if(endTime == time && time != 0){ endTime = 0; locked = false; }        // If a thread completes its task
        if(endTime == 0) atomicMin(&freeId, id);                                // If free, it's eligible to compete for task allocation
        if(!locked){
          tempId = 0;
          while(prior[queueLock]!=threadPriors[tempId] && tempId < m) tempId++; // If not locked, find a thread with matching priorities
          if(tempId == m) tempId = freeId;                                      // if such a matching priority thread isnt found, allocate to minimal id thread
          if(tempId == id){
            freeId = m+1;                                                       // resetting freeId such that it doesnt collide with usual threadIds.
            if(endTime == 0){
              if(threadPriors[id] == -1)
                threadPriors[id] = prior[queueLock];                            // setting up priority of the first task ever run by a thread
              endTime = time + execs[queueLock];
              result[queueLock] = endTime; 
              queueLock++;                                                      // Once allocated, pop the next task
            }
            else locked = true;                                                 // If a matching priority thread is found and it isn't free, lock
          }
        }
        else ++time;                                                            // If locked, time flows
    }while(queueLock<n);
}


//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	int *d_exec,*d_prior,*d_result;
	
	//Allocating space for the host_arrays 
	d_exec = (int *) malloc(n * sizeof(int));
	d_prior = (int *) malloc(n * sizeof(int));	
	d_result = (int *) malloc(n * sizeof(int));	

	cudaMalloc(&d_exec, n * sizeof(int));
	cudaMalloc(&d_prior, n * sizeof(int));
	cudaMalloc(&d_result, n * sizeof(int));
	
	cudaMemcpy(d_exec, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prior, priority, n * sizeof(int), cudaMemcpyHostToDevice);
	
	dscheduler<<<1,m, m * sizeof(int)>>>(m, n, d_exec, d_prior, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);
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
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
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
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}

