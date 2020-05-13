#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdbool.h>

int NowYear = 2020; // 2020 - 2025
int NowMonth = 0;   // 0 - 11
int TotalMonths = 0;
bool IsPandemic = false;

float NowPrecip = 6;    // inches of rain per month
float NowTemp = 60;     // temperature this month
float NowHeight = 12.0; // grain height in inches
int NowNumDeer = 1;     // number of deer in the current population

unsigned int seed = 934905323;

const float GRAIN_GROWS_PER_MONTH = 9.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0; // average
const float AMP_PRECIP_PER_MONTH = 6.0; // plus or minus
const float RANDOM_PRECIP = 2.0;        // plus or minus noise

const float AVG_TEMP = 60.0;    // average
const float AMP_TEMP = 20.0;    // plus or minus
const float RANDOM_TEMP = 10.0; // plus or minus noise

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

float Ranf(unsigned int *seedp, float low, float high)
{
    float r = (float)rand_r(seedp); // 0 - RAND_MAX

    return (low + r * (high - low) / (float)RAND_MAX);
}

void set_temp_and_precip()
{
    float ang = (30.0 * (float)NowMonth + 15.0) * (M_PI / 180.);

    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
    if (NowPrecip < 0.)
        NowPrecip = 0.;
}

void GrainDeer()
{
    while (NowYear < 2026)
    {
        int nextNumDeer = NowNumDeer;
        if (NowNumDeer < NowHeight)
        {
            nextNumDeer++;
        }
        else if (NowNumDeer > NowHeight)
        {
            nextNumDeer--;
        }
        if (nextNumDeer < 0)
        {
            nextNumDeer = 0;
        }
// Done computing
#pragma omp barrier
        NowNumDeer = nextNumDeer;
// Done Assigning
#pragma omp barrier
// Done Post-Processing
#pragma omp barrier
// Done Printing
#pragma omp barrier
    }
}

float sqr(float a)
{
    return a * a;
}

void Grain()
{
    while (NowYear < 2026)
    {
        float tempFactor = exp(-sqr((NowTemp - MIDTEMP) / 10.0));
        float precipFactor = exp(-sqr((NowPrecip - MIDPRECIP) / 10.0));
        int nextHeight = NowHeight;
        // printf("TempFactor: %f\nPrecipFactor: %f\n", tempFactor, precipFactor);
        int newGrows = tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        int afterDeer = (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        nextHeight += newGrows;
        nextHeight -= afterDeer;
        // printf("New Grows: %d\nAfter Deer: %d\n", newGrows, afterDeer);
        if (nextHeight < 0.0)
        {
            nextHeight = 0.0;
        }
        // printf("Curr Height: %f\nNext Height: %d\n", NowHeight, nextHeight);
// Done computing
#pragma omp barrier
        NowHeight = nextHeight;
// Done Assigning
#pragma omp barrier
// Done Post-Processing
#pragma omp barrier
// Done Printing
#pragma omp barrier
    }
}

void Watcher()
{
    while (NowYear < 2026)
    {
// Done computing
// Done Assigning
#pragma omp barrier
// Done Post-Processing
#pragma omp barrier
#pragma omp barrier
        int panda = 20;
        if (IsPandemic == true)
        {
            panda = 35;
        }
        else
        {
            panda = 0;
        }
        printf("{\"year\": %d, \"month\": %d, \"grain_height\": %f, \"deer_count\": %d, \"precipitation\": %f, \"temperature\": %f, \"is_pandemic\": %d}", NowYear, NowMonth, NowHeight, NowNumDeer, NowPrecip, NowTemp, panda);
        if (NowMonth < 11)
        {
            NowMonth++;
        }
        else
        {
            NowYear++;
            NowMonth = 0;
        }
        TotalMonths++;

        if (!(NowMonth == 0 && NowYear == 2026))
        {
            printf(",\n");
        }
        else
        {
            printf("\n");
        }
        set_temp_and_precip();
// Done Printing
#pragma omp barrier
    }
}
void Pandemic()
{
    int pandemicStart = 0;
    bool covid_alert = false;
    int pandemic_length = 0;
    while (NowYear < 2026)
    {
        covid_alert = rand() % 30 == 0;
        if (covid_alert)
        {
            pandemicStart = TotalMonths;
            pandemic_length = (rand() % 4) + 8;
        }

// Done computing
#pragma omp barrier
// Done Assigning
#pragma omp barrier
        if (pandemic_length > TotalMonths - pandemicStart)
        {
            IsPandemic = true;
            NowNumDeer = 0;
            NowHeight = 0;
        }
        else
        {
            IsPandemic = false;
        }
// Done Post-Processing
#pragma omp barrier

// Done Printing
#pragma omp barrier
    }
}

int main()
{
    set_temp_and_precip();
    omp_set_num_threads(4); // same as # of sections
#pragma omp parallel sections
    {
#pragma omp section
        {
            GrainDeer();
        }

#pragma omp section
        {
            Grain();
        }

#pragma omp section
        {
            Watcher();
        }

#pragma omp section
        {
            Pandemic();
        }
    }

    return 0;
}
