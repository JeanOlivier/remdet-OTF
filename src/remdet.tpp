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


// SPECIALIZATIONS //
// This didn't actually help at all; same performance.
/*
template <>
inline void detsum_func(int16_t *buff, int64_t *detsum, uint64_t period){
    // detsum is int64_t if T is int16_t
    uint64_t *buff_64 = (uint64_t *) buff;
    uint64_t tmp;
    // Reading block of 64 bits
    #pragma omp simd
    for (uint64_t j=0; j<period/4; j++){
        tmp = buff_64[j];
        detsum[j*4+0] += (int16_t)(tmp       & 0xFFFF);
        detsum[j*4+1] += (int16_t)(tmp >> 16 & 0xFFFF);
        detsum[j*4+2] += (int16_t)(tmp >> 32 & 0xFFFF);
        detsum[j*4+3] += (int16_t)(tmp >> 48 & 0xFFFF);
    }
    // Remainder
    for (uint64_t j=period-period%4; j<period; j++){
        detsum[j] += buff[j];
    }
}
*/

// This also didn't speed things up; kept as an AVX example.
/*
#include <immintrin.h>
#include <smmintrin.h>
template <>
void deldet(int16_t *buffer, uint64_t size, int16_t *detpart, uint64_t period) {
    if (not period%16){ // If period is a multiple of 16
        #pragma omp parallel
        {
            manage_thread_affinity();
            #pragma omp for
            for (uint64_t i = 0; i < size; i+=16) {
                // load 256-bit chunks of each array
                __m256i first_values = _mm256_load_si256((__m256i*) &buffer[i]);
                __m256i second_values = _mm256_load_si256((__m256i*) &detpart[i%period]);

                // subs each pair of 16-bit integers in the 128-bit chunks
                first_values = _mm256_sub_epi16(first_values, second_values);
                
                // store 256-bit chunk to first array
                _mm256_store_si256((__m256i*) &buffer[i], first_values);
            }
        }
        // handle left-over
        for (uint64_t i = size-size%16; i < size; i++) {
            buffer[i] -= detpart[i%period];
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
*/