#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define XMAX 1.
#define XMIN -1.
#define YMAX 1.
#define YMIN -1.

#ifndef NUMNODES
#define NUMNODES 2000
#endif

#define N 4

#ifndef NUMT
#define NUMT 1
#endif

float height(int, int);

int main(int argc, char *argv[])
{

    omp_set_num_threads(NUMT);
    // the area of a single full-sized tile:

    float fullTileArea = (((XMAX - XMIN) / (float)(NUMNODES - 1)) *
                          ((YMAX - YMIN) / (float)(NUMNODES - 1)));
    float halfTileArea = fullTileArea / 2.0;
    float quarterTileArea = fullTileArea / 4.0;

    // sum up the weighted heights into the variable "volume"
    // using an OpenMP for loop and a reduction:

    double time0 = omp_get_wtime();

    float totalArea = 0;
#pragma omp parallel for default(none) shared(fullTileArea, halfTileArea, quarterTileArea) reduction(+ \
                                                                                                     : totalArea)
    for (int i = 0; i < NUMNODES * NUMNODES; i++)
    {
        int iu = i % NUMNODES;
        int iv = i / NUMNODES;
        float h = height(iv, iu);
        // printf("Hpre: %f\n", h);
        // We're at a corner
        if ((iv == 0 && iu == 0) ||
            (iv == NUMNODES - 1 && iu == NUMNODES - 1) ||
            (iv == 0 && iu == NUMNODES - 1) ||
            (iv == NUMNODES - 1 && iu == 0))
        {
            h *= quarterTileArea;
        }
        // We're at an edge
        else if (iu == 0 ||
                 iv == 0 ||
                 iu == NUMNODES - 1 ||
                 iv == NUMNODES - 1)
        {
            h *= halfTileArea;
        }
        // We're somewhere in the middle
        else
        {
            h *= fullTileArea;
        }
        // Have to do this because idk
        h *= 2;
        // printf("Hafter: %f\n", h);
        totalArea += h;
        // printf("Area: %f\n", totalArea);
    }
    double time1 = omp_get_wtime();
    double megaNodes = (NUMNODES * NUMNODES) / (time1 - time0) / 1000000.0;
    printf("Area: { \"area\": %f, \"nodes\": %d, \"threads\": %d, \"meganodes\": %f }, \n", totalArea, NUMNODES, NUMT, megaNodes);
}

float height(int iv, int iu) // iu,iv = 0 .. NUMNODES-1
{
    float x = -1. + 2. * (float)iu / (float)(NUMNODES - 1); // -1. to +1.
    float y = -1. + 2. * (float)iv / (float)(NUMNODES - 1); // -1. to +1.

    float xn = pow(fabs(x), (double)N);
    float yn = pow(fabs(y), (double)N);
    float r = 1. - xn - yn;
    if (r < 0.0)
        return 0.0;
    float height = pow(1. - xn - yn, 1. / (float)N);
    return height;
}
