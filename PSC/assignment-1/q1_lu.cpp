#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <bits/stdc++.h>
using namespace std;
#include <omp.h>

double final_result;
const int n = 25;
const float al = 0.0;
const float bl = 3.0;
const float h = (bl - al) / n;

float f(float x)
{
    return (5 * sin(x));
}

void instantiate(int a[], char f)
{
    int i;
    for (int i = 0; i < n - 1; i++)
    {
        switch (f)
        {
        case 'a':
        case 'c':
            a[i] = 1;
            break;
        default:
            if (i == 0)
                a[i] = 1;
            else
                a[i] = 4;
            break;
        }
    }
    if (f == 'c')
        a[0] = 2;
    if (f == 'b')
        a[n] = 1;
    if (f == 'a')
        a[n - 2] = 2;

    int k;
    for (int k = 0; k < n - 1; k++)
        cout << a[k] << " ";
    if (f == 'b')
        cout << a[n] << "\n";
    else
        cout << "\n";
}

void yvector(float y[])
{
    int k;
    y[0] = (-2.5 * f(0) + 2 * f(h) + 0.5 * f(2 * h));
    y[n] = (-2.5 * f(n * h) - 2 * f((n - 1) * h) - 0.5 * f((n - 2) * h));

    for (k = 1; k < n - 1; k++)
        y[k] = 3 * (f((k + 1) * h) - f((k - 1) * h)) / h;
}
void ytransform(float y[], int l[])
{
    int k;

    for (k = 0; k < n - 1; k++)
        y[k + 1] -= l[k + 1] * y[k];
}

void xtransform(float x[], float y[], int c[], int u[])
{
    int k;
    for (k = n - 1; k > 0; k--)
        x[k] = (y[k] / u[k]) - (c[k] * x[k + 1] / u[k]);
}

int main(int argc, char *argv[])
{
    int a[n - 1], b[n], c[n - 1];
    int l[n - 1], u[n];
    float x[n], y[n];

    instantiate(c, 'c');
    instantiate(b, 'b');
    instantiate(a, 'a');
    yvector(y);

    int k = 0;
    u[0] = 1;
    for (k = 0; k < n - 1; k++)
    {
        l[k] = a[k + 1] / u[k];
        u[k + 1] = b[k + 1] - l[k] * c[k];
    }
    ytransform(y, l);
    x[n] = y[n] / u[n];
    xtransform(x, y, c, u);

    FILE *fp;
    fp = fopen("plt_q1_lu.txt", "w");

    for (k = 0; k < n; k++)
        fprintf(fp, "%f %f \n", x[k], 5 * cos(k * h));
    fclose(fp);
}