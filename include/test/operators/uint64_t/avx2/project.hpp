#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PROJECT_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PROJECT_HPP

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include <utils.hpp>

void project_sequential_avx2_64(
   uint64_t * __restrict__ result,
   uint64_t const * __restrict__ pos_data,
   std::size_t pos_count,
   uint64_t const * __restrict__ data
) {

   auto const [pos_vectorized_end, pos_data_end] = get_end_pos( pos_data, pos_count, 2 );
   auto const pos_vectorized_end_ptr = reinterpret_cast< __m256i const * >( pos_vectorized_end );
   auto base_addr = reinterpret_cast< long long int const * >( data );
   auto pos_vec_ptr = reinterpret_cast< __m256i const * >( pos_data );
   auto result_vec_ptr = reinterpret_cast< __m256i * >( result );

   while( pos_vec_ptr != pos_vectorized_end_ptr ) {
      _mm256_store_si256(
         result_vec_ptr,
         _mm256_i64gather_epi64(
            base_addr,
            _mm256_load_si256( pos_vec_ptr ),
            8
         )
      );
      ++result_vec_ptr;
      ++pos_vec_ptr;
   }
   auto pos_data_remainder = reinterpret_cast< uint64_t const * >( pos_vec_ptr );
   auto result_remainder = reinterpret_cast< uint64_t * >( result_vec_ptr );
   while( pos_data_remainder != pos_data_end ) {
      *result_remainder++ = data[ *pos_data_remainder++ ];
   }
}




#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PROJECT_HPP
