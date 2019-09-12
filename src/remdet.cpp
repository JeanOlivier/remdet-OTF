#include "remdet.tpp"

// Instantiating the types we want to support
#define declare_F_for_U(F,U) template void F<U##_t>(U##_t *buffer, uint64_t size, U##_t *detpart, uint64_t period);
#define declare_allF_for_U(U) \
    declare_F_for_U(getdet, U);\
    declare_F_for_U(deldet, U);\
    declare_F_for_U(remdet, U);

declare_allF_for_U(int8);
declare_allF_for_U(uint8);
declare_allF_for_U(int16);
declare_allF_for_U(uint16);