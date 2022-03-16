#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_VL_UTILS_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_VL_UTILS_HPP

#include <immintrin.h>

#ifndef AVX512VL
#define MOVEMASKi64(x) _mm256_movemask_pd(_mm256_castsi256_pd(x))
#define MOVEMASKi32(x) _mm256_movemask_ps(_mm256_castsi256_ps(x))
#define CMPEQi64(a,b) MOVEMASKi64(_mm256_cmpeq_epi64(a,b))
#define CMPEQi32(a,b) MOVEMASKi32(_mm256_cmpeq_epi32(a,b))
#define CMPRESSSTOREi64(ptr, mask, val) *ptr = val[ 0 ]; \
                                        ptr += ( mask & 0b1 ); \
                                        *ptr = val[ 1 ]; \
                                        ptr += ( ( mask >> 1 ) & 0b1 ); \
                                        *ptr = val[ 2 ]; \
                                        ptr += ( ( mask >> 2 ) & 0b1 ); \
                                        *ptr = val[ 3 ]; \
                                        ptr += ( ( mask >> 3 ) & 0b1 );
//#define CMPRESSSTOREi32(ptr, mask, val) *ptr = val[ 0 ]; \
//                                        ptr += ( mask & 0b1 ); \
//                                        *ptr = val[ 1 ]; \
//                                        ptr += ( ( mask >> 1 ) & 0b1 ); \
//                                        *ptr = val[ 2 ]; \
//                                        ptr += ( ( mask >> 2 ) & 0b1 ); \
//                                        *ptr = val[ 3 ]; \
//                                        ptr += ( ( mask >> 3 ) & 0b1 ); \
//                                        *ptr = val[ 4 ]; \
//                                        ptr += ( ( mask >> 4 ) & 0b1 ); \
//                                        *ptr = val[ 5 ]; \
//                                        ptr += ( ( mask >> 5 ) & 0b1 ); \
//                                        *ptr = val[ 6 ]; \
//                                        ptr += ( ( mask >> 6 ) & 0b1 ); \
//                                        *ptr = val[ 7 ]; \
//                                        ptr += ( ( mask >> 7 ) & 0b1 );
#define CMPRESSSTOREi32(ptr, mask, val) [&ptr](int m, __m256i v) -> void { \
   alignas(32) std::array< uint32_t, 8 > tmp;                                                             \
   _mm256_store_si256( reinterpret_cast< __m256i * >( tmp.data() ), v );                                  \
   *ptr = tmp[ 0 ]; \
   ptr += ( m & 0b1 );                                                                                      \
   *ptr = tmp[ 1 ]; \
   ptr += ( ( m >> 1) & 0b1 );                                                                                      \
   *ptr = tmp[ 2 ]; \
   ptr += ( ( m >> 2) & 0b1 ); \
   *ptr = tmp[ 3 ]; \
   ptr += ( ( m >> 3) & 0b1 ); \
   *ptr = tmp[ 4 ]; \
   ptr += ( ( m >> 4) & 0b1 ); \
   *ptr = tmp[ 5 ]; \
   ptr += ( ( m >> 5) & 0b1 ); \
   *ptr = tmp[ 6 ]; \
   ptr += ( ( m >> 6) & 0b1 ); \
   *ptr = tmp[ 7 ]; \
   ptr += ( ( m >> 7) & 0b1 );                                                                              \
}( mask, val )
#define MASKADDi64(mask, a, b) \
                                        _mm256_add_epi64(a, \
                                         _mm256_andnot_si256( \
                                          _mm256_sub_epi64( \
                                           _mm256_and_si256(\
                                            _mm256_srlv_epi64(\
                                             _mm256_set1_epi64x(mask&0xF), \
                                             _mm256_set_epi64x(3,2,1,0)    \
                                            ),              \
                                            _mm256_set1_epi64x(1)          \
                                           ),               \
                                           _mm256_set1_epi64x(1)           \
                                          ),                \
                                          b                 \
                                         )                  \
                                        )
#define MASKADDi32(mask, a, b) \
                                        _mm256_add_epi32(a, \
                                         _mm256_andnot_si256( \
                                          _mm256_sub_epi32( \
                                           _mm256_and_si256(\
                                            _mm256_srlv_epi32(\
                                             _mm256_set1_epi32(mask&0xFF), \
                                             _mm256_set_epi32(7,6,5,4,3,2,1,0)    \
                                            ),              \
                                            _mm256_set1_epi32(1)          \
                                           ),               \
                                           _mm256_set1_epi32(1)           \
                                          ),                \
                                          b                 \
                                         )                  \
                                        )
#define COMPARE_N_ADDi64(r,a,p,b) _mm256_add_epi64(r, _mm256_and_si256( _mm256_cmpeq_epi64( a, p ), b ) )
#define COMPARE_N_ADDi32(r,a,p,b) _mm256_add_epi32(r, _mm256_and_si256( _mm256_cmpeq_epi32( a, p ), b ) )

#else
#define MOVEMASKi64(x) x
#define MOVEMASKi32(x) x
#define CMPEQi64(a, b) _mm256_cmpeq_epi64_mask(a, b)
#define CMPEQi32(a, b) _mm256_cmpeq_epi32_mask(a, b)
#define CMPRESSSTOREi64(ptr, mask, val) _mm256_mask_compressstoreu_epi64(reinterpret_cast< void * >( ptr ), mask, val ); ptr += __builtin_popcount( mask & 0xF )
#define CMPRESSSTOREi32(ptr, mask, val) _mm256_mask_compressstoreu_epi32(reinterpret_cast< void * >( ptr ), mask, val ); ptr += __builtin_popcount( mask & 0xFF )
#define MASKADDi64(mask,a,b) _mm256_mask_add_epi64(a,mask,a,b)
#define MASKADDi32(mask,a,b) _mm256_mask_add_epi32(a,mask,a,b)
#define COMPARE_N_ADDi64(r,a,p,b) _mm256_mask_add_epi64(r, _mm256_cmpeq_epi64_mask(a,p), r, b)
#define COMPARE_N_ADDi32(r,a,p,b) _mm256_mask_add_epi32(r, _mm256_cmpeq_epi32_mask(a,p), r, b)
#endif
#define REDUCEADDi64(a) a[0]+a[1]+a[2]+a[3]
#define REDUCEADDi32(a) [](__m256i x) -> uint32_t { alignas(32) std::array< uint32_t, 8 > tmp; _mm256_store_si256( reinterpret_cast< __m256i * >( tmp.data() ), x ); return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]; }( a )
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_VL_UTILS_HPP
