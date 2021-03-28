#include <stdio.h>
#include <math.h>
#include <omp.h>

const int N = 401;
const double delta = 0.01;
const double threshold = 1;

double f(double x, double y)
{
    return 2 * (2 - pow(x, 2) - pow(y, 2));
}
double g(double x, double y)
{
    return (pow(x, 2) - 1) * (pow(y, 2) - 1);
}

void matcopy(double a[N][N], double b[N][N])
{
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            b[i][j] = a[i][j];
        }
    }
}

double compare(double mat[N][N], double exact[N][N])
{
    double max_diff = 0;
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            double diff = 100 * fabs(exact[i][j] - mat[i][j]) / mat[i][j];
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

void serial_gauss_seidel(double mat_old[N][N], double delta, double exact[N][N])
{
    FILE *fp;
    fp = fopen("history_q2a.txt", "w");
    int k;
    for (k = 1; k <= 10000; k++)
    {
        double mat_new[N][N];
        matcopy(mat_old, mat_new);
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                double x = -1 + i * delta;
                double y = -1 + j * delta;
                mat_new[i][j] = 0.25 * (mat_old[i + 1][j] + mat_new[i - 1][j] + mat_old[i][j + 1] + mat_new[i][j - 1]) + 0.25 * pow(delta, 2) * f(x, y);
            }
        }
        double max_diff = compare(mat_new, exact);
        printf("Iteration %d , max_diff %f \n", k, max_diff);
        fprintf(fp, "%d %f \n", k, max_diff);

        matcopy(mat_new, mat_old);
        if (max_diff <= threshold)
        {
            printf("Converged after %d iterations \n\n", k);
            fclose(fp);
            return;
        }
    }
    fclose(fp);
    printf("\n Max iter limit reached \n");
    return;
}

void init_zeroes(double mat[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            mat[i][j] = 0;
        }
    }
}

void init_exact(double mat[N][N], double delta)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double x = -1 + i * delta;
            double y = -1 + j * delta;
            mat[i][j] = g(x, y);
        }
    }
}

void plotwrite(double mat[N][N], double exact[N][N], int y)
{
    int j = (1 + y) / delta;
    FILE *fp;
    fp = fopen("plt_q2a.txt", "w");
    for (int i = 0; i < N; i++)
    {
        double x = -1 + i * delta;
        double y = -1 + j * delta;
        fprintf(fp, "%f %f %f %f \n", x, y, mat[i][j], exact[i][j]);
    }
    fclose(fp);

    FILE *fp;
    fp = fopen("data_q2a.txt", "w");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(fp, "%f ", mat[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
    double mat[N][N], exact[N][N];
    init_zeroes(mat);
    init_exact(exact, delta);

    double start_time = omp_get_wtime();
    serial_gauss_seidel(mat, delta, exact);
    double time = omp_get_wtime() - start_time;

    printf("Total time used is %f\n", time);

    int y = 0.5;
    plotwrite(mat, exact, y);

    return 0;
}