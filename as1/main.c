#include <omp.h>
#include <stdio.h>
#include <math.h>

#ifndef NUMT
#define NUMT 1
#endif
#define SIZE (1 << 16) // 65536

#define NUMTRIES 1024

float A[SIZE];
float B[SIZE];
float C[SIZE];

int main(int argc, char **argv)
{
#ifndef _OPENMP
    fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
    return 1;
#endif
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    omp_set_num_threads(NUMT);
    fprintf(stderr, "Using %d threads\n", NUMT);

    double maxMegaMults = 0.;

    for (int t = 0; t < NUMTRIES; t++)
    {
        double start = omp_get_wtime();

#pragma omp parallel for
        for (int i = 0; i < SIZE; i++)
        {
            C[i] = A[i] * B[i];
        }

        double end = omp_get_wtime();
        double megaMults = (double)SIZE / (end - start) / 1000000.0;
        if (megaMults > maxMegaMults)
        {
            maxMegaMults = megaMults;
        }
    }

    printf("Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults);
    return 0;
}
