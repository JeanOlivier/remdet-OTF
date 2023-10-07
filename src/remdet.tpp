#include "remdet.hpp"

using namespace std;

// Computes average signal on block of length *period* and removes it from
// buffer. Also returns the deterministic signal for a single period.
template<typename T> 
void remdet(T *buffer, uint64_t size, T *detpart, uint64_t period)
{
    getdet(buffer, size, detpart, period);  // detpart is now the det part
    deldet(buffer, size, detpart, period);  // detpart is removed from buffer
}

template<typename T>
void getdet(T *buffer, uint64_t size, T *detpart, uint64_t period)
{   
    // detsum should have same sign as T for performance
    typedef typename conditional<((T)-1>0), uint64_t, int64_t>::type detsum_t;
    detsum_t *detsum = new detsum_t [period](); // Initialized to 0x00s
    uint64_t numblocks = size/period;
    // Whole blocks
    #pragma omp parallel
    {
        manage_thread_affinity();
        #pragma omp for simd reduction(+:detsum[:period])
        for (uint64_t i=0; i<numblocks; i++){
            T *buff = buffer + i*period;
            detsum_func(buff, detsum, period);
        }
        // Remainder
        T *buff = buffer + numblocks*period;
        #pragma omp for simd reduction(+:detsum[:period])
        for (uint64_t j=0; j<size%period; j++){
            detsum[j] += buff[j];
        }
        #pragma omp for simd reduction(+:detpart[:period])
        for (uint64_t j=0; j<period; j++){
            // size%period first values are averaged on numblocks+1 samples
            uint64_t N = numblocks + (j < size%period);
            detpart[j] = (T)(double)mpfr::rint_round((mpreal)detsum[j]/(mpreal)N);
            // The above should improve average accuracy via rounding in mpfr
            // Converting to double first to avoid using T-specific functions
            // Ensuing loss of precision should be less than discret«ization
        }
    }
    delete [] detsum;   // Clearing memory
}

template<typename T>
void deldet(T *buffer, uint64_t size, T *detpart, uint64_t period)
{
    // detpart is the actual deterministic part
    uint64_t numblocks = size/period; // Don't care about the remainder for now
    // Whole blocks
    #pragma omp parallel
    {
        manage_thread_affinity();
        #pragma omp for simd
        for (uint64_t i=0; i<numblocks; i++){
            T *buff = buffer + i*period;
            detsub_func(buff, detpart, period);
        }
        /*#pragma omp for simd // Was much slower, prob bc of the mod
        for (uint64_t i=0; i<size; i++){    // No need for remainder
            buffer[i] -= detpart[i%period];
        }*/
    }
    // Remainder
    T *buff = buffer + numblocks*period;
    for (uint64_t j=0; j<size%period; j++){
        buff[j] -= detpart[j];
    }
}

// To allow for T-specific optimisations

template<typename T>
inline void detsum_func(T *buff, auto *detsum, uint64_t period){
    for (uint64_t j=0; j<period; j++){
        detsum[j] += buff[j];
    }
}

template<typename T>
inline void detsub_func(T *buff, T *detpart, uint64_t period){
    for (uint64_t j=0; j<period; j++){
        buff[j] -= detpart[j];
    }
}


// This also didn't speed things up, but uses a single core for same performance!
// This likely shows that we're memory-bandwidth limited. 
//
// A single-threaded version that uses AVX also for getdet could be more
// "efficient" in terms of perf/watt, even if not faster at the wall clock.


#include <immintrin.h>
#include <smmintrin.h>

template<>
void deldet(int16_t *buffer, uint64_t size, int16_t *detpart, uint64_t period) {
    if (period==8){  
        // Here we double the detpart to use efficient SIMD
        int16_t *detpart_twice = new int16_t [16];
        for (int16_t i=0; i<16; i++){
            detpart_twice[i] = detpart[i%8];
        }

        deldet_16_int16(buffer, size, detpart_twice);  // Will use AVX2 or not according to platform

        // handle left-over
        for (uint64_t j = size-size%16; j < size; j++) {
            buffer[j] -= detpart_twice[j%16];
        }
        delete [] detpart_twice;
    }
    else if (period==16){ // If period is a multiple of 16
        deldet_16_int16(buffer, size, detpart);  // Will use AVX2 or not according to platform

        // handle left-over
        for (uint64_t i = size-size%16; i < size; i++) {
            buffer[i] -= detpart[i%16];
        }
    }
    else{
        // detpart is the actual deterministic part
        uint64_t numblocks = size/period; // Don't care about the remainder for now
        // Whole blocks
        #pragma omp parallel
        {
        manage_thread_affinity();
            #pragma omp for simd
            for (uint64_t i=0; i<numblocks; i++){
                int16_t *buff = buffer + i*period;
                detsub_func(buff, detpart, period);
            }
        }
        // Remainder
        int16_t *buff = buffer + numblocks*period;
        for (uint64_t j=0; j<size%period; j++){
            buff[j] -= detpart[j];
        }
    }
}

__attribute__ ((target("avx2")))  // int16 specialization using AVX2
void deldet_16_int16(int16_t *buffer, uint64_t size, int16_t *detpart) {
    __m256i vec_detpart = _mm256_loadu_si256((__m256i*) detpart);
    for (uint64_t i = 0; i < size-size%16; i+=16) {
        // load 256-bit chunks of each array
        __m256i vec_buffer = _mm256_loadu_si256((__m256i*) &buffer[i]);
        // subs each pair of 16-bit integers in the 128-bit chunks
        vec_buffer = _mm256_sub_epi16(vec_buffer, vec_detpart);
        // store 256-bit chunk to first array
        _mm256_storeu_si256((__m256i*) &buffer[i], vec_buffer);
    }
}

__attribute__ ((target("default")))  // non-SIMD version
void deldet_16_int16(int16_t *buffer, uint64_t size, int16_t *detpart) {
    for (uint64_t i = 0; i < size; i+=16){
        detsub_func(buffer + i, detpart, (uint64_t) 16);
    }
}
