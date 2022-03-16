#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_FILTER_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_FILTER_HPP
#include <cstdint>
#include <immintrin.h>

#include <operators/vl_utils.hpp>

void filter_eq_bitmask_sequential_avx2_32(
   uint8_t * __restrict__ result,
   uint32_t const * __restrict__ data,
   std::size_t element_count,
   uint32_t value
) {
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto const value_vec = _mm256_set1_epi32( value );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
      *result++ = CMPEQi32(_mm256_load_si256( data_vec_ptr ), value_vec );
      ++data_vec_ptr;
   }
}


void filter_eq_bitmask_strided_avx2_32(
   uint8_t * __restrict__ result,
   uint32_t const * __restrict__ data,
   std::size_t element_count,
   uint32_t value,
   std::size_t stride_size
) {
   std::size_t stride = stride_size >> 2;
   auto const vidx = _mm256_set_epi32(
      7*stride, 6*stride, 5*stride, 4*stride, 3*stride,2*stride, stride, 0
   );
   auto const value_vec = _mm256_set1_epi32( value );
   auto const chunk_count = element_count / (stride*8);
   auto const chunk_inc = 7*stride;
   auto base_addr = reinterpret_cast< int const * >( data );
   for( auto j = 0; j < chunk_count; ++j ) {
      for( auto i = 0; i < stride; ++i ) {
         *result++ = CMPEQi32( _mm256_i32gather_epi32( base_addr, vidx, 4 ), value_vec );
         ++base_addr;
      }
      base_addr += chunk_inc;
   }
}

std::size_t filter_eq_poslist_sequential_avx2_32(
   uint32_t * __restrict__ result,
   uint32_t const * __restrict__ data,
   std::size_t element_count,
   uint32_t value
) {
   auto result_current = result;
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto const value_vec = _mm256_set1_epi32( value );
   auto pos_vec = _mm256_set_epi32( 7,6,5,4,3,2,1,0);
   auto const inc_vec = _mm256_set1_epi32( 8 );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
      auto result_mask = CMPEQi32( _mm256_load_si256( data_vec_ptr ), value_vec );
      CMPRESSSTOREi32( result_current, result_mask, pos_vec );
      pos_vec = _mm256_add_epi32( pos_vec, inc_vec );
      ++data_vec_ptr;
   }
   return result_current - result;
}
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_FILTER_HPP
