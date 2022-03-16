#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_FILTER_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_FILTER_HPP
#include <cstdint>
#include <immintrin.h>

#include <operators/vl_utils.hpp>

void filter_eq_bitmask_sequential_avx2_64(
   uint8_t * __restrict__ result,
   uint64_t const * __restrict__ data,
   std::size_t element_count,
   uint64_t value
) {
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto const value_vec = _mm256_set1_epi64x( value );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
      *result++ = CMPEQi64( _mm256_load_si256( data_vec_ptr ), value_vec);
      ++data_vec_ptr;
   }
}

void filter_eq_bitmask_strided_avx2_64(
   uint8_t * __restrict__ result,
   uint64_t const * __restrict__ data,
   std::size_t element_count,
   uint64_t value,
   std::size_t stride_size
) {
   std::size_t stride = stride_size >> 3;
   auto const vidx = _mm256_set_epi64x(
      3*stride,2*stride, stride, 0
   );
   auto const value_vec = _mm256_set1_epi64x( value );
   auto const chunk_count = element_count / (stride*4);
   auto const chunk_inc = 3*stride;

   auto base_addr = reinterpret_cast< long long int const * >( data );

   for( auto j = 0; j < chunk_count; ++j ) {
      for( auto i = 0; i < stride; ++i ) {
         *result++ = CMPEQi64( _mm256_i64gather_epi64( base_addr, vidx, 8 ), value_vec);
         ++base_addr;
      }
      base_addr += chunk_inc;
   }
}

std::size_t filter_eq_poslist_sequential_avx2_64(
   uint64_t * __restrict__ result,
   uint64_t const * __restrict__ data,
   std::size_t element_count,
   uint64_t value
) {
   auto result_current = result;
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto const value_vec = _mm256_set1_epi64x( value );
   auto pos_vec = _mm256_set_epi64x(3,2,1,0);
   auto const inc_vec = _mm256_set1_epi64x( 4 );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
      auto result_mask = CMPEQi64( _mm256_load_si256( data_vec_ptr ), value_vec);
      CMPRESSSTOREi64(result_current, result_mask, pos_vec );
      pos_vec = _mm256_add_epi64( pos_vec, inc_vec );
      ++data_vec_ptr;
   }
   return result_current - result;
}
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_FILTER_HPP
