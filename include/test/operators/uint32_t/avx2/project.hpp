#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_PROJECT_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_PROJECT_HPP

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include <utils.hpp>

void project_sequential_avx2_32(
   uint32_t * __restrict__ result,
   uint32_t const * __restrict__ pos_data,
   std::size_t pos_count,
   uint32_t const * __restrict__ data
) {

   auto const [pos_vectorized_end, pos_data_end] = get_end_pos( pos_data, pos_count, 3 );
   auto const pos_vectorized_end_ptr = reinterpret_cast< __m256i const * >( pos_vectorized_end );
   auto base_addr = reinterpret_cast< int const * >( data );
   auto pos_vec_ptr = reinterpret_cast< __m256i const * >( pos_data );
   auto result_vec_ptr = reinterpret_cast< __m256i * >( result );

   while( pos_vec_ptr != pos_vectorized_end_ptr ) {
         _mm256_store_si256(
            result_vec_ptr,
            _mm256_i32gather_epi32(
               base_addr,
               _mm256_load_si256( pos_vec_ptr ),
               4
            )
         );
         ++result_vec_ptr;
         ++pos_vec_ptr;
      }
   pos_data = reinterpret_cast< uint32_t const * >( pos_vec_ptr );
   auto result_remainder = reinterpret_cast< uint32_t * >( result_vec_ptr );
   while( pos_data != pos_data_end ) {
      *result_remainder++ = data[ *pos_data++ ];
   }
}


#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT32_T_AVX2_PROJECT_HPP
