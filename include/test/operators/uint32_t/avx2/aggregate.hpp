#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_AGGREGATE_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_AGGREGATE_HPP


#include <cstdint>
#include <cstddef>
#include <array>
#include <immintrin.h>
#include <utils.hpp>

__m256i aggregate_sum_bitmask_sequential_avx2_32(
   uint8_t const * __restrict__ valid_elements,
   uint32_t const * __restrict__ data,
   std::size_t element_count,
   __m256i result_vec
) {
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
#ifdef USE_BRANCH
         auto mask = *valid_elements;
         if( mask != 0 ) {
               result_vec = MASKADDi32( mask, result_vec, _mm256_load_si256( data_vec_ptr ));
            }
#else
         result_vec = MASKADDi32(*valid_elements, result_vec, _mm256_load_si256( data_vec_ptr ) );
#endif
         ++data_vec_ptr;
         ++valid_elements;
      }
   return result_vec;
}

__m256i aggregate_sum_bitmask_strided_avx2_32(
   uint8_t const * __restrict__ valid_elements,
   uint32_t const * __restrict__ data,
   std::size_t element_count,
   __m256i result_vec,
   std::size_t stride_size
) {
   std::size_t stride = stride_size >> 2;
   auto const vidx = _mm256_set_epi32(
      7*stride, 6*stride, 5*stride, 4*stride, 3*stride,2*stride, stride, 0
   );
   auto const chunk_count = element_count / (stride*8);
   auto const chunk_inc = 7*stride;
   auto base_addr = reinterpret_cast< int const * >( data );

   for( auto j = 0; j < chunk_count; ++j ) {
         for( auto i = 0; i < stride; ++i ) {
#ifdef USE_BRANCH
               auto mask = *valid_elements;
               if( mask != 0 ) {
                  result_vec = MASKADDi32(mask, result_vec, _mm256_i32gather_epi32( base_addr, vidx, 4 ));
               }
#else
               result_vec = MASKADDi32(*valid_elements, result_vec, _mm256_i32gather_epi32( base_addr, vidx, 4 ));
#endif
               ++base_addr;
               ++valid_elements;
            }
         base_addr += chunk_inc;
      }
   return result_vec;
}

__m256i aggregate_sum_poslist_sequential_avx2_32(
   uint32_t const * __restrict__ valid_data,
   std::size_t element_count,
   __m256i result_vec
) {
   auto const [vectorized_end, data_end] = get_end_pos( valid_data, element_count, 3 );
   auto const valid_data_end_ptr = reinterpret_cast< __m256i const * >( vectorized_end );
   auto valid_data_vec_ptr = reinterpret_cast< __m256i const * >( valid_data );

   while( valid_data_vec_ptr != valid_data_end_ptr ) {
         result_vec = _mm256_add_epi32(
            result_vec,
            _mm256_load_si256(
               valid_data_vec_ptr
            )
         );
         ++valid_data_vec_ptr;
      }
   valid_data = reinterpret_cast< uint32_t const * >( valid_data_vec_ptr );
   if( valid_data != data_end ) {
         alignas( 64 ) std::array< uint32_t, 8 > tmp = {};
         auto i = 0;
         while( valid_data != data_end ) {
               tmp[ i++ ] = *valid_data++;
            }
         result_vec = _mm256_add_epi32(
            result_vec,
            _mm256_load_si256( reinterpret_cast< __m256i const * >( tmp.data() ) )
         );
      }
   return result_vec;
}
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_AGGREGATE_HPP
