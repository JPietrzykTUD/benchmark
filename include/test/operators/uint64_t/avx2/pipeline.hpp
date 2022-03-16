#ifndef TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PIPELINE_HPP
#define TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PIPELINE_HPP

#include <cstdint>
#include <immintrin.h>

#include <operators/vl_utils.hpp>

uint64_t query_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred
) {
   auto result = _mm256_setzero_si256();
   auto const pred_vec = _mm256_set1_epi64x( pred );
   auto const [to_filter_vectorized_end, to_filter_end ] = get_end_pos( to_filter_data, element_count, 2 );
   auto to_filter_vec_ptr = reinterpret_cast< __m256i const * >( to_filter_data );
   auto to_agg_vec_ptr = reinterpret_cast< __m256i const * >( to_agg_data );
   auto to_filter_vec_end_ptr = reinterpret_cast< __m256i const * >( to_filter_vectorized_end );
   while( to_filter_vec_ptr != to_filter_vec_end_ptr ) {
#ifdef USE_BRANCH
      auto mask = CMPEQi64(_mm256_load_si256( to_filter_vec_ptr), pred_vec);
      if( mask != 0 ) {
         result = MASKADDi64( mask, result, _mm256_load_si256( to_agg_vec_ptr ));
      }
#else
         result = COMPARE_N_ADDi64(result, _mm256_load_si256( to_filter_vec_ptr ), pred_vec, _mm256_load_si256( to_agg_vec_ptr ));
#endif
         ++to_filter_vec_ptr;
         ++to_agg_vec_ptr;
      }
   return REDUCEADDi64( result );
}


uint64_t query_strided_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred,
   std::size_t stride_size
) {
   auto result = _mm256_setzero_si256();
   std::size_t stride = stride_size >> 3;
   auto const vidx = _mm256_set_epi64x(
      3*stride,2*stride, stride, 0
   );
   auto const value_vec = _mm256_set1_epi64x( pred );
   auto const chunk_count = element_count / (stride*4);
   auto const chunk_inc = 3*stride;

   auto base_addr_filter = reinterpret_cast< long long int const * >( to_filter_data );
   auto base_addr_agg = reinterpret_cast< long long int const * >( to_agg_data );


   for( auto j = 0; j < chunk_count; ++j ) {
         for( auto i = 0; i < stride; ++i ) {
#ifdef USE_BRANCH
               auto mask = CMPEQi64(_mm256_i64gather_epi64( base_addr_filter, vidx, 8 ), value_vec);
               if( mask != 0 ) {
                     result = MASKADDi64( mask, result, _mm256_i64gather_epi64( base_addr_agg, vidx, 8 ));
                  }
#else
               result = COMPARE_N_ADDi64(
                  result,
                  _mm256_i64gather_epi64( base_addr_filter, vidx, 8 ),
                  value_vec,
                  _mm256_i64gather_epi64( base_addr_agg, vidx, 8 )
               );
#endif
               ++base_addr_filter;
               ++base_addr_agg;
            }
         base_addr_filter += chunk_inc;
         base_addr_agg += chunk_inc;
      }
   return REDUCEADDi64( result );
}
#endif //TOGATHERORNOT_INCLUDE_OPERATORS_UINT64_T_AVX2_PIPELINE_HPP
