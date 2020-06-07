#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <xmmintrin.h>
#include <omp.h>
#include <assert.h>

#include "cl.h"
#include "cl_platform.h"

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 128
#endif

#ifndef CL_FILE_NAME
#define CL_FILE_NAME "./src/sum.cl"
#endif

#define SSE_WIDTH 4

using namespace std;

typedef struct
{
    vector<float> data;
    float megamults;
} bench_results;

bench_results simd_parallel(vector<float>, long unsigned int);
float simd_mul_sum(float *, float *, long unsigned int);
bench_results non_parallel(vector<float>, long unsigned int);
bench_results omp_parallel(vector<float>, long unsigned int);
void wait_for_cl_command_queue(cl_command_queue);
bench_results opencl_parallel(vector<float>, long unsigned int);
float megamults_from_time(float time0, float time1, long unsigned int original_size);

int main()
{
    long unsigned int original_size = 0;
    ifstream file("./signals.txt");
    vector<float> lines;
    string line;
    while (getline(file, line))
    {
        lines.push_back(stof(line));
    }
    file.close();
    original_size = lines.size();
    lines.resize(lines.size() * 2);
    for (int i = 0; i < original_size; i++)
    {
        lines[i + original_size] = lines[i];
    }

    bench_results non_res = non_parallel(lines, original_size);
    bench_results simd_res = simd_parallel(lines, original_size);

    omp_set_num_threads(1);
    bench_results omp_single_res = omp_parallel(lines, original_size);

    omp_set_num_threads(12);
    bench_results omp_parallel_res = omp_parallel(lines, original_size);

    bench_results cl_res = opencl_parallel(lines, original_size);

#ifdef _PERF
    cout << "No Parallelism:" << non_res.megamults << endl
         << "SIMD Parallelism:" << simd_res.megamults << endl
         << "OMP 1-Thread:" << omp_single_res.megamults << endl
         << "OMP 12-Threads:"
         << omp_parallel_res.megamults << endl
         << "OpenCL:" << cl_res.megamults << endl;
#endif

#ifdef _DATA
    for (int i = 1; i <= 512; i++)
    {
        cout << i << ":" << cl_res.data[i] << endl;
    }
#endif
}

bench_results non_parallel(vector<float> in, long unsigned int original_size)
{
    vector<float> results;
    results.resize(original_size);

    double t0 = omp_get_wtime();
    for (int shift = 0; shift < original_size; shift++)
    {
        float sum = 0.0;

        for (int i = 0; i < original_size; i++)
        {
            sum += in[i] * in[i + shift];
        }

        results[shift] = sum;
    }
    double t1 = omp_get_wtime();

    bench_results res =
        {
            .data = results,
            .megamults = megamults_from_time(t0, t1, original_size),
        };

    return res;
}

bench_results omp_parallel(vector<float> in, long unsigned int original_size)
{
    vector<float> results;
    results.resize(original_size);

    double t0 = omp_get_wtime();
#pragma omp parallel for default(none) shared(results, original_size, in)
    for (int shift = 0; shift < original_size; shift++)
    {
        float sum = 0.0;

        for (int i = 0; i < original_size; i++)
        {
            sum += in[i] * in[i + shift];
        }

        results[shift] = sum;
    }
    double t1 = omp_get_wtime();

    bench_results res =
        {
            .data = results,
            .megamults = megamults_from_time(t0, t1, original_size),
        };

    return res;
}

bench_results simd_parallel(vector<float> in, long unsigned int original_size)
{
    vector<float> results;
    results.resize(original_size);

    double t0 = omp_get_wtime();
    for (int shift = 0; shift < original_size; shift++)
    {
        results[shift] = simd_mul_sum(&in[0], &in[shift], original_size);
    }
    double t1 = omp_get_wtime();

    bench_results res =
        {
            .data = results,
            .megamults = megamults_from_time(t0, t1, original_size),
        };

    return res;
}

float simd_mul_sum(float *a, float *b, long unsigned int len)
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

bench_results opencl_parallel(vector<float> in, long unsigned int original_size)
{

    FILE *fp;
    fp = fopen(CL_FILE_NAME, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
        exit(1);
    }

    cl_int status; // returned status from opencl calls
                   // test against CL_SUCCESS

    // get the platform id:

    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (2)\n");

    // get the device id:

    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetDeviceIDs failed (2)\n");

    // 2. allocate the host memory buffers:

    // 3. create an opencl context:

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateContext failed\n");

    // 4. create an opencl command queue:

    cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateCommandQueue failed\n");

    // 5. allocate the device memory buffers:

    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * original_size * sizeof(cl_float), NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateBuffer failed (1)\n");

    cl_mem dSums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, original_size * sizeof(cl_float), NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateBuffer failed (3)\n");

    // 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

    float *hA = new float[2 * original_size];
    for (int i = 0; i < 2 * original_size; i++)
    {
        hA[i] = in[i];
    }
    float *hSums = new float[1 * original_size];

    status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, 2 * original_size * sizeof(cl_float), hA, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

    wait_for_cl_command_queue(cmdQueue);

    // 7. read the kernel code from a file:

    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *clProgramText = new char[fileSize + 1]; // leave room for '\0'
    size_t n = fread(clProgramText, 1, fileSize, fp);
    clProgramText[fileSize] = '\0';
    fclose(fp);
    if (n != fileSize)
        fprintf(stderr, "Expected to read %ld bytes read from '%s' -- actually read %ld.\n", fileSize, CL_FILE_NAME, n);

    // create the text for the kernel program:

    char *strings[1];
    strings[0] = clProgramText;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)strings, NULL, &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateProgramWithSource failed\n");
    delete[] clProgramText;

    // 8. compile and link the kernel code:

    const char *options = {""};
    status = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        size_t size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        cl_char *log = new cl_char[size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
        fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
        delete[] log;
    }

    // 9. create the kernel object:

    cl_kernel kernel = clCreateKernel(program, "AutoCorrelate", &status);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clCreateKernel failed\n");

    // 10. setup the arguments to the kernel object:

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clSetKernelArg failed (1)\n");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dSums);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clSetKernelArg failed (2)\n");

    // 11. enqueue the kernel object for execution:

    size_t globalWorkSize[3] = {original_size, 1, 1};
    size_t localWorkSize[3] = {LOCAL_SIZE, 1, 1};

    wait_for_cl_command_queue(cmdQueue);

    double t0 = omp_get_wtime();
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

    wait_for_cl_command_queue(cmdQueue);
    double t1 = omp_get_wtime();

    // 12. read the results buffer back from the device to the host:

    status = clEnqueueReadBuffer(cmdQueue, dSums, CL_TRUE, 0, original_size * sizeof(cl_float), hSums, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clEnqueueReadBuffer failed\n");
    wait_for_cl_command_queue(cmdQueue);

    vector<float> sums(original_size);
    for (int i = 0; i < original_size; i++)
    {
        sums[i] = hSums[i];
    }

    // 13. clean everything up:

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(dA);
    clReleaseMemObject(dSums);

    delete[] hA;
    delete[] hSums;

    bench_results res =
        {
            .data = sums,
            .megamults = megamults_from_time(t0, t1, original_size),
        };

    return res;
}

void wait_for_cl_command_queue(cl_command_queue queue)
{
    cl_event wait;
    cl_int status;

    status = clEnqueueMarker(queue, &wait);
    if (status != CL_SUCCESS)
        fprintf(stderr, "Wait: clEnqueueMarker failed\n");

    status = clWaitForEvents(1, &wait);
    if (status != CL_SUCCESS)
        fprintf(stderr, "Wait: clWaitForEvents failed\n");
}

float megamults_from_time(float time0, float time1, long unsigned int original_size)
{
    return (double)original_size / (time1 - time0) / 1000000.;
}
