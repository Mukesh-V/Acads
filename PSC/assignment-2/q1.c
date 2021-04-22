#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

double func(double x){
    return 5 * sin(x);
}
void differentiate(double *sendbuf, double la, double lb, double ln, double h, int rank, int nprocs){
    int i;
    if(rank == 0){
        for(i = 0; i < ln; i++){
            if(i < 2)
                sendbuf[i] = ( func(la + (i+1)*h) - func(la + i*h) ) / h;
            else
                sendbuf[i] = ( -func(la + (i+2)*h) + 8*func(la + (i+1)*h) - 8*func(la + (i-1)*h) + func(la + (i-2)*h) ) / ( 12 * h );
        }
    }
    else if(rank == nprocs-1){
        for(i = 0; i < ln; i++){
            if(i > ln-3)
                sendbuf[i] = ( func(la + i*h) - func(la + (i-1)*h) ) / h;
            else
                sendbuf[i] = ( -func(la + (i+2)*h) + 8*func(la + (i+1)*h) - 8*func(la + (i-1)*h) + func(la + (i-2)*h) ) / ( 12 * h );
        }
    }
    else
        for(i = 0; i < ln; i++)
            sendbuf[i] = ( -func(la + (i+2)*h) + 8*func(la + (i+1)*h) - 8*func(la + (i-1)*h) + func(la + (i-2)*h) ) / ( 12 * h );
}

int main(int argc, char* argv[]){
    double a, b, la, lb, lsum, h;
    int rank, nprocs, proc, n, ln;
    double  *recvbuf, *sendbuf;

    a = 0.0; b = 3.0; n = 100000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(rank == 0){
        recvbuf = (double *) malloc( n * sizeof(double) );
    }

    h = (b-a)/n;
    ln = n/nprocs;

    la = a + rank * ln * h;
    lb = la + ln * h;
    sendbuf = (double *) malloc(ln * sizeof(double));
    differentiate(sendbuf, la, lb, ln, h, rank, nprocs);

    MPI_Gather(sendbuf, ln, MPI_DOUBLE, recvbuf, ln, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0){
        for(int i = 0; i < n; i++)
            printf("%lf %lf\n", recvbuf[i], 5*cos(a + i*h));
        free(recvbuf);
    }

    free(sendbuf);
    
    MPI_Finalize();
}