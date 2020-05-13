#include <xmmintrin.h>
#include <omp.h>
#include <stdio.h>
#define SSE_WIDTH 4

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 1024 // 1kb
#endif

float NormalMulSum(float *a, float *b, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

float SimdMulSum(float *a, float *b, int len)
{
    float sum[4] = {0., 0., 0., 0.};
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;
    register float *pa = a;
    register float *pb = b;

    __m128 ss = _mm_loadu_ps(&sum[0]);
    for (int i = 0; i < limit; i += SSE_WIDTH)
    {
        ss = _mm_add_ps(ss, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
        pa += SSE_WIDTH;
        pb += SSE_WIDTH;
    }
    _mm_storeu_ps(&sum[0], ss);

    for (int i = limit; i < len; i++)
    {
        sum[0] += a[i] * b[i];
    }

    return sum[0] + sum[1] + sum[2] + sum[3];
}

int main()
{
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = 3.0;
        b[i] = 2.0;
    }

    double time0 = omp_get_wtime();
    float sum = NormalMulSum(a, b, ARRAY_SIZE);
    double time1 = omp_get_wtime();
    float simd_sum = SimdMulSum(a, b, ARRAY_SIZE);
    double time2 = omp_get_wtime();

    printf("Sum/SimdSum: %d - %d\n", sum, simd_sum);
    printf("{\"array_size\": %d, \"control_time\": %d, \"simd_time\": %d}\n", ARRAY_SIZE, time1 - time0, time2 - time1);
}
