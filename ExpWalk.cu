/* Exp Walk
 * Written by Owen Welsh
 * Latest version dated August 27, 2012
 *
 * Simulates Pollard's multiplicative Rho algorithm using a walk on the
 * exponents, using a GPU to run many simulations at once.  Runs about
 * 6.666 times faster than an equivalent algoritm running on the CPU.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LOOPS 965 /* Number of loops each thread will execute */
#define THREADS 256 /* Must be a power of 2 */
#define BLOCKS 13 /* Same as the number of prime orders to be used */

typedef unsigned long long int uint64_t;
typedef unsigned int uint;

__device__ __constant__ uint seedszD[BLOCKS];
__device__ __constant__ uint seedswD[BLOCKS];

/* An RNG that uses the values of two seeds to generate output */
__device__ uint GetUInt(uint *m_z, uint *m_w)
{
        *m_z = 36969 * ((*m_z) & 65535) + ((*m_z) >> 16);
        *m_w = 18000 * ((*m_w) & 65535) + ((*m_w) >> 16);
        return ((*m_z) << 16) + *m_w;
}

/* Used so that seeds that start out with similar values don't
   generate similar output for the first few itterations of GetUInt */
__device__ void shuffle(uint *m_z, uint *m_w)
{
        GetUInt(m_z, m_w);
        GetUInt(m_z, m_w);
        GetUInt(m_z, m_w);
        GetUInt(m_z, m_w);
        GetUInt(m_z, m_w);
}

/* A hash function that uses a modified version of the Mersenne Twister
   algorithm to randomly map values to 0, 1, or 2 */
__device__ uint64_t case3(uint64_t x)
{
        x = 1812433253 * (x ^ (x >> 30)) + 1;
        x = x ^ (x >> 11);
        x = x ^ ((x << 7) & 2636928640);
        x = x ^ ((x << 15) & 4022730752);
        x = x ^ (x >> 18);
        return x % 3;
}

/* Runs one step of the exponent walk with x as the currect state.
   Returns the new state. */
__device__ uint64_t func(uint64_t x, uint64_t *a, uint64_t *b,
    uint64_t k, uint64_t N)
{
        switch(case3(x)) {
        case 0: (*a)++; break;
        case 1: (*b)++; break;
        case 2: (*a) *= 2; (*b) *= 2; break;
        }
        *a = (*a) % N;
        *b = (*b) % N;
        return ((*a) + (*b) * k) % N;
}

/* Uses Brent's cycle detection algorithm to determine how long until a
   self-intersection with x0 as the starting state, k as the solution, and N
   as the prime order.  Returns the number of steps until self-intersection. */
__device__ uint64_t brent(uint64_t x0, uint64_t k, uint64_t N)
{
        int power = 1;
        int i;
        uint64_t lambda = 1;
        uint64_t mu = 0;
        uint64_t tortoise = x0;
        uint64_t hare;
        uint64_t at, ah, bt, bh;

        ah = at = x0;
        bh = bt = 0;
        hare = func(x0, &ah, &bh, k, N);

/* Determines the length of the cycle (lambda, i.e. the head of the rho) */
        while (tortoise != hare) {
                if (power == lambda) {
                        tortoise = hare;
                        power *= 2;
                        lambda = 0;
                }
                hare = func(hare, &ah, &bh, k, N);
                lambda++;
        }

        tortoise = hare = x0;
        ah = at = x0;
        bh = bt = 0;

/* Determines the time until first intersection (mu, the tail of the rho)*/
        for (i = 0; i < lambda; i++)
                hare = func(hare, &ah, &bh, k, N);
        while (tortoise != hare) {
                tortoise = func(tortoise, &at, &bt, k, N);
                hare = func(hare, &ah, &bh, k, N);
                mu++;
        }

        return lambda + mu;
}

/* Uses brent() LOOPS times in THREADS threads to simulate Pollard's Rho and
   determine the average required number of steps until self-intersection.
   Uses random k and x0 values in each run. */
__global__ void simulate(float* stepsD, uint64_t *primesD)
{
        __shared__ float steps[THREADS];
        uint64_t N = primesD[blockIdx.x];
        uint64_t x0, k;
        uint64_t stepsI = 0;
        int i;
        uint id = threadIdx.x;
        uint stride, mz, mw;

        mz = seedszD[blockIdx.x] + id;
        mw = seedswD[blockIdx.x] + id;

        shuffle(&mz, &mw);

        for (i = 0; i < LOOPS; i++) {
                x0 = (uint64_t) GetUInt(&mz, &mw) % N;
                k = (((uint64_t) GetUInt(&mz, &mw)) % (N - 2)) + 2;

                stepsI += brent(x0, k, N);
        }
        __syncthreads();

        steps[id] = ((float) stepsI) / LOOPS;

/* Summates every thread's result to later find the average.
   Requires that THREADS be a power of 2. */
        for (stride = blockDim.x / 2; stride > 1; stride >>= 1) {
                __syncthreads();
                if (id < stride)
                        steps[id] += steps[id + stride];
        }
        __syncthreads();
        if (id < stride)
                steps[id] += steps[id + stride];

        if (id == 0)
                stepsD[blockIdx.x] = steps[0] / blockDim.x;
}

/* Determines I/O methods and which prime orders will be used during simulation,
   and handles the CUDA bookkeeping. */
int main(int argc, char *argv[])
{
        uint64_t primes[BLOCKS] = {251, 503, 1009, 2003,
                4001, 8009, 16001, 32003, 64007,
                128021, 255989, 511997, 1000003};
        uint64_t *primesD;
        float *stepsD;
        float steps[BLOCKS];
        const size_t size = sizeof(uint64_t) * BLOCKS;
        const size_t sizeF = sizeof(float) * BLOCKS;
        const size_t sizeU = sizeof(uint) * BLOCKS;
        uint seedsz[BLOCKS];
        uint seedsw[BLOCKS];
        int i;

        srand(time(NULL));
        cudaMalloc((void**) &stepsD, sizeF);
        cudaMalloc((void**) &primesD, size);
        cudaMemcpy(primesD, primes, size, cudaMemcpyHostToDevice);

/* If you wish the loop this process, the body of the loop should begin here */
        for(i = 0; i < BLOCKS; i++) {
                seedsz[i] = (uint) rand();
                seedsw[i] = (uint) rand();
        }

        cudaMemcpyToSymbol(seedszD, seedsz, sizeU, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(seedswD, seedsw, sizeU, cudaMemcpyHostToDevice);

        simulate<<<BLOCKS, THREADS>>>(stepsD, primesD);

        cudaMemcpy(steps, stepsD, sizeF, cudaMemcpyDeviceToHost);
/* And end here (with the requisit summing mechanism placed after the last call
   to cudaMemcpy()) */

        cudaFree(stepsD);
        cudaFree(primesD);

        for (i = 0; i < BLOCKS; i++)
                printf("N=%lu, steps: %f\n", primes[i], steps[i]);

        return 0;
}