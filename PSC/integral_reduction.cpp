#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <bits/stdc++.h>
#include <chrono>
using namespace std;
#include <omp.h>

#define PI 3.14159265358

double func(double x)
{
    return (1+sqrt(sin(x)*sin(x)));
}

double integrate(int n, double a, double b)
{
    double h, x, total;
    int i;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int local_n;
    double local_a, local_b;

    h = (b - a) / n;

    local_n = n / thread_count;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    total = (func(local_a) + func(local_b)) / 2.0;
    for (i = 1; i <= local_n - 1; i++)
    {
        x = local_a + i * h;
        total += func(x);
    }
    total = total * h;

    return total;
}

int main(int argc, char *argv[])
{
    double a, b, final_result;
    int n;
    int thread_count = 1;

    if (argc == 2){
        thread_count = strtol(argv[1], NULL, 10);
        printf("Thread num : %d", thread_count);
    }
    else
        printf("Default num : 1");

    n = 100000;
    a = 0.0;
    b = 1000*PI;
    final_result = 0.0;

    auto start = chrono::high_resolution_clock::now();
    ios_base::sync_with_stdio(false);

#pragma omp parallel num_threads(thread_count) reduction(+: final_result)
    final_result += integrate(n, a , b);

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;

    cout << "\nThe area under the curve between 0 to 1000PI is equal to " << fixed << final_result << setprecision(9) << endl;
    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(9);
    cout << " sec" << endl;
    return 0;
}