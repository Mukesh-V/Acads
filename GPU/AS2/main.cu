#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

// write kernels here...
__global__ void multiply(int p, int q, int r, int s, int *P, int *Q, int *R, int* S, int flag){
	// if flag = 0, we calculate (P+Qt)R
	// else we calculate PQt
	
	// multiplies two matrices of dimensions rows * inter and inter * cols
	
	__shared__ int rows=p, inter, cols;
	if(flag == 0)
		inter = q, cols = r;
	else
		inter = r, cols = s; 
		
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  	int threadId = (blockId * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  	int row = threadId/n, col = threadId%n;
  	
  	__shared__ int sum = 0;
	int i;
	if(flag == 0)
		for(i=0; i < inter; i++)
			second_row[i] += (P[row*inter + i] + Q[i*inter + col]) * R[i*cols + col];
	else
		for(i=0; i < inter; i++)
			second_row[i] += P[row*inter + i] * Q[row*cols + i];
	
	S[row][col] = sum;
}

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_A, *d_B, *d_C, d*D;
	int *h_inter, *d_inter;
	int *d_X;	
	
	// allocate memory...
	h_inter = (int*) malloc(p * r * sizeof(int));
	
	cudaMalloc(&d_A, p * q * sizeof(int));
	cudaMalloc(&d_B, q * p * sizeof(int));
	cudaMalloc(&d_C, q * r * sizeof(int));
	cudaMalloc(&d_D, s * r * sizeof(int));
	cudaMalloc(&d_inter, p * r * sizeof(int));
	cudaMalloc(&d_X, p * s * sizeof(int));
	
	// copy the values...
	cudaMemcpy(d_A, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
	
	// call the kernels for doing required computations...
	dim3 block3(64,16,1);
	multiply<<<1, block3>>>(p, q, r, s, d_A, d_B, d_C, d_inter, 0);
	cudaMemcpy(d_inter, h_inter, p * r * sizeof(int), cudaMemcpyDeviceToHost);
	multiply<<<1, block3>>>(p, q, r, s, d_inter, d_D, d_C, d_X, 1);
	
	// copy the result back...
	cudaMemcpy(d_X, h_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
	
	// deallocate the memory...
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaFree(d_inter);
	cudaFree(d_X);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p
	);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}
