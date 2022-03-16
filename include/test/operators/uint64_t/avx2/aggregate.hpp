#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_AGGREGATE_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_AGGREGATE_HPP

#include <cstdint>
#include <cstddef>
#include <array>
#include <immintrin.h>
#include <utils.hpp>

__m256i aggregate_sum_bitmask_sequential_avx2_64(
   uint8_t const * __restrict__ valid_elements,
   uint64_t const * __restrict__ data,
   std::size_t element_count,
   __m256i result_vec
) {
   auto const data_end = reinterpret_cast< __m256i const * >( data + element_count );
   auto data_vec_ptr = reinterpret_cast< __m256i const * >( data );
   while( data_vec_ptr != data_end ) {
#ifdef USE_BRANCH
         auto mask = *valid_elements;
         if( mask != 0 ) {
               result_vec = MASKADDi64( mask, result_vec, _mm256_load_si256( data_vec_ptr ));
            }
#else
      result_vec = MASKADDi64(*valid_elements, result_vec, _mm256_load_si256( data_vec_ptr ) );
#endif
      ++data_vec_ptr;
      ++valid_elements;
   }
   return result_vec;
}

__m256i aggregate_sum_bitmask_strided_avx2_64(
   uint8_t const * __restrict__ valid_elements,
   uint64_t const * __restrict__ data,
   std::size_t element_count,
   __m256i result_vec,
   std::size_t stride_size
) {
   std::size_t stride = stride_size >> 3;
   auto const vidx = _mm256_set_epi64x(
      3*stride,2*stride, stride, 0
   );
   auto const chunk_count = element_count / (stride*4);
   auto const chunk_inc = 3*stride;
   auto base_addr = reinterpret_cast< long long int const * >( data );

   for( auto j = 0; j < chunk_count; ++j ) {
      for( auto i = 0; i < stride; ++i ) {
#ifdef USE_BRANCH
         auto mask = *valid_elements;
         if( mask != 0 ) {
            result_vec = MASKADDi64(mask, result_vec, _mm256_i64gather_epi64( base_addr, vidx, 8 ));
         }
#else
            result_vec = MASKADDi64(*valid_elements, result_vec, _mm256_i64gather_epi64( base_addr, vidx, 8 ));
#endif
         ++base_addr;
         ++valid_elements;
      }
         base_addr += chunk_inc;
   }
   return result_vec;
}

__m256i aggregate_sum_poslist_sequential_avx2_64(
   uint64_t const * __restrict__ valid_data,
   std::size_t element_count,
   __m256i result_vec
) {
   auto const [vectorized_end, data_end] = get_end_pos( valid_data, element_count, 2 );
   auto const valid_data_end_ptr = reinterpret_cast< __m256i const * >( vectorized_end );
   auto valid_data_vec_ptr = reinterpret_cast< __m256i const * >( valid_data );

   while( valid_data_vec_ptr != valid_data_end_ptr ) {
      result_vec = _mm256_add_epi64(
         result_vec,
         _mm256_load_si256(
            valid_data_vec_ptr
         )
      );
      ++valid_data_vec_ptr;
   }
   valid_data = reinterpret_cast< uint64_t const * >( valid_data_vec_ptr );
   if( valid_data != data_end ) {
         alignas( 64 ) std::array< uint64_t, 4 > tmp = {};
         auto i = 0;
         while( valid_data != data_end ) {
            tmp[ i++ ] = *valid_data++;
         }
         result_vec = _mm256_add_epi64(
            result_vec,
            _mm256_load_si256( reinterpret_cast< __m256i const * >( tmp.data() ) )
         );
      }
   return result_vec;
}
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_AGGREGATE_HPP
