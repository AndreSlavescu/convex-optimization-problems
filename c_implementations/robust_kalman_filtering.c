#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Function to compute the running sum of squares of an array
#pragma inline
int running_sum(int *a, int length)
{
    assert(a);
    assert(length > 0);

    int running_sum = 0;
#pragma unroll
    for (int i = 0; i < length; ++i)
    {
        running_sum += a[i] * a[i];
    }

    return running_sum;
}

// Function to compute the l2 norm of an array
double l2_norm(int *a, int length)
{
    int total_sum = running_sum(a, length);
    return sqrt(total_sum);
}

// Function to compute the Huber function as defined in the example
double huber_function(int *a, int length, double p)
{
    assert(a);
    assert(p > 0);
    assert(length > 0);

    double norm = l2_norm(a, length);

    if (norm <= p)
    {
        return running_sum(a, length);
    }
    return 2 * p * norm - p * p;
}

int main()
{
    int a[] = {1, 2, 3, 4};
    int length = 4;
    double p = 1.0;
    printf("Huber Function Value: %lf\n", huber_function(a, length, p));
    return 0;
}
