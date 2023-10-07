#ifndef remdet_H
#define remdet_H

#include "common.hpp"

// Computes average signal on block of length *period* and removes it from
// buffer. Also returns the deterministic signal for a single period.

// All-in-one, initial detpart should be filled with 0x00
template<typename T> void remdet(T *buffer, uint64_t size, T *detpart, uint64_t period);
// Just get the det part, detpart should be zeroed
template<typename T> void getdet(T *buffer, uint64_t size, T *detpart, uint64_t period);
// Deletes the det part in-place inside buffer, detpart should be the real thing
template<typename T> void deldet(T *buffer, uint64_t size, T *detpart, uint64_t period);

// For T-specialized optimisations
template<typename T> inline void detsum_func(T *buff, auto *detsum, uint64_t period);
template<typename T> inline void detsub_func(T *buff, T *detpart, uint64_t period);
// AVX2 specialization associated function
void deldet_16_int16(int16_t *buffer, uint64_t size, int16_t *detpart);

#endif // remdet_H
