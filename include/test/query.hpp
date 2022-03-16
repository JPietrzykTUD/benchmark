#ifndef TOGATHERORNOT_INCLUDE_QUERY_HPP
#define TOGATHERORNOT_INCLUDE_QUERY_HPP

#include <operators/filter.hpp>
#include <operators/project.hpp>
#include <operators/aggregate.hpp>
#include <operators/pipelines.hpp>

#include <utils.hpp>
#include <operators/vl_utils.hpp>


#include <immintrin.h>

template< typename T >
T query_scalar(
   T const * __restrict__ to_filter_data,
   T const * __restrict__ to_agg_data,
   std::size_t element_count,
   T pred
) {
   T result = (T)0;
   auto const end = to_filter_data + element_count;
   while( to_filter_data != end ) {
      result += ( *to_filter_data == pred ) ? *to_agg_data : (T)0;
      ++to_filter_data;
      ++to_agg_data;
   }
   return result;
}

//Query using bitmasks as intermediates, Operator at a time
uint64_t query_bitmask_oat_sequential_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 2 ) * sizeof( uint8_t ) ) );

   filter_eq_bitmask_sequential_avx2_64(
      filter_result, to_filter_data, element_count, pred
   );

   auto result = aggregate_sum_bitmask_sequential_avx2_64(
      filter_result, to_agg_data, element_count, _mm256_setzero_si256( )
   );

   free( filter_result );
   return REDUCEADDi64( result );
}
//Query using bitmasks as intermediates with Strided Access, Operator at a time
uint64_t query_bitmask_oat_strided_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred,
   std::size_t stride_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 2 ) * sizeof( __mmask8 ) ) );

   filter_eq_bitmask_strided_avx2_64(
      filter_result, to_filter_data, element_count, pred, stride_size
   );

   auto result = aggregate_sum_bitmask_strided_avx2_64(
      filter_result, to_agg_data, element_count, _mm256_setzero_si256( ), stride_size
   );

   free( filter_result );
   return REDUCEADDi64( result );
}
//Query using bitmasks as intermediates, Operator at a time
uint32_t query_bitmask_oat_sequential_avx2_32(
   uint32_t const * __restrict__ to_filter_data,
   uint32_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint32_t pred
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 3 ) * sizeof( uint8_t ) ) );

   filter_eq_bitmask_sequential_avx2_32(
      filter_result, to_filter_data, element_count, pred
   );

   auto result = aggregate_sum_bitmask_sequential_avx2_32(
      filter_result, to_agg_data, element_count, _mm256_setzero_si256( )
   );

   free( filter_result );
   return REDUCEADDi32( result );
}
//Query using bitmasks as intermediates with Strided Access, Operator at a time
uint32_t query_bitmask_oat_strided_avx2_32(
   uint32_t const * __restrict__ to_filter_data,
   uint32_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint32_t pred,
   std::size_t stride_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 3 ) * sizeof( uint8_t ) ) );

   filter_eq_bitmask_strided_avx2_32(
      filter_result, to_filter_data, element_count, pred, stride_size
   );

   auto result = aggregate_sum_bitmask_strided_avx2_32(
      filter_result, to_agg_data, element_count, _mm256_setzero_si256( ), stride_size
   );
   free( filter_result );
   return REDUCEADDi32( result );
}
uint64_t query_poslist_oat_sequential_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred
) {
   auto filter_result = reinterpret_cast< uint64_t * >( _mm_malloc( element_count * sizeof( uint64_t ), 64 ) );


   auto valid_element_count = filter_eq_poslist_sequential_avx2_64(
      filter_result, to_filter_data, element_count, pred
   );
//   std::cerr << "[256] VALID ENTRIES: " << valid_element_count << "\n";
   auto project_result = reinterpret_cast< uint64_t * >( _mm_malloc( valid_element_count * sizeof( uint64_t ), 64 ) );

   project_sequential_avx2_64(
      project_result, filter_result, valid_element_count, to_agg_data
   );

   auto result = aggregate_sum_poslist_sequential_avx2_64(
      project_result, valid_element_count, _mm256_setzero_si256( )
   );

   _mm_free( project_result );
   _mm_free( filter_result );
   return REDUCEADDi64( result );
}
//Query using poslists as intermediates, Operator at a time
uint32_t query_poslist_oat_sequential_avx2_32(
   uint32_t const * __restrict__ to_filter_data,
   uint32_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint32_t pred
) {
   auto filter_result = reinterpret_cast< uint32_t * >( _mm_malloc( element_count * sizeof( uint32_t ), 64 ) );


   auto valid_element_count = filter_eq_poslist_sequential_avx2_32(
      filter_result, to_filter_data, element_count, pred
   );

   auto project_result = reinterpret_cast< uint32_t * >( _mm_malloc( valid_element_count * sizeof( uint32_t ), 64 ) );

   project_sequential_avx2_32(
      project_result, filter_result, valid_element_count, to_agg_data
   );

   auto result = aggregate_sum_poslist_sequential_avx2_32(
      project_result, valid_element_count, _mm256_setzero_si256( )
   );

   _mm_free( project_result );
   _mm_free( filter_result );
   return REDUCEADDi32( result );
}
//Query using bitmasks as intermediates, Vector at a time
uint64_t query_bitmask_vat_sequential_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred,
   std::size_t vector_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( vector_size >> 2 ) * sizeof( uint8_t ) ) );

   auto const vector_stepwidth = vector_size >> 3;
   auto const vector_count = element_count / vector_stepwidth;
   auto const remainder = element_count % vector_stepwidth;

   auto result = _mm256_setzero_si256();

   for( auto vector = 0; vector < vector_count; ++vector ) {
         filter_eq_bitmask_sequential_avx2_64(
            filter_result, to_filter_data, vector_stepwidth, pred
         );
         result = aggregate_sum_bitmask_sequential_avx2_64(
            filter_result, to_agg_data, vector_stepwidth, result
         );
         to_filter_data += vector_stepwidth;
         to_agg_data += vector_stepwidth;
      }
   if( remainder != 0 ) {
         filter_eq_bitmask_sequential_avx2_64(
            filter_result, to_filter_data, remainder, pred
         );
         result = aggregate_sum_bitmask_sequential_avx2_64(
            filter_result, to_agg_data, remainder, result
         );
      }

   free( filter_result );
   return REDUCEADDi64( result );
}

//Query using bitmasks as intermediates, Vector at a time
uint64_t query_bitmask_vat_strided_avx2_64(
   uint64_t const * __restrict__ to_filter_data,
   uint64_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint64_t pred,
   std::size_t vector_size,
   std::size_t stride_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( vector_size >> 2 ) * sizeof( uint8_t ) ) );

   auto const vector_stepwidth = vector_size >> 3;
   auto const vector_count = element_count / vector_stepwidth;
   auto const remainder = element_count % vector_stepwidth;

   auto result = _mm256_setzero_si256();

   for( auto vector = 0; vector < vector_count; ++vector ) {
         filter_eq_bitmask_strided_avx2_64(
            filter_result, to_filter_data, vector_stepwidth, pred, stride_size
         );
         result = aggregate_sum_bitmask_strided_avx2_64(
            filter_result, to_agg_data, vector_stepwidth, result, stride_size
         );
         to_filter_data += vector_stepwidth;
         to_agg_data += vector_stepwidth;
      }
   if( remainder != 0 ) {
         filter_eq_bitmask_strided_avx2_64(
            filter_result, to_filter_data, remainder, pred, stride_size
         );
         result = aggregate_sum_bitmask_strided_avx2_64(
            filter_result, to_agg_data, remainder, result, stride_size
         );
      }

   free( filter_result );
   return REDUCEADDi64( result );
}

//Query using bitmasks as intermediates, Vector at a time
uint32_t query_bitmask_vat_sequential_avx2_32(
   uint32_t const * __restrict__ to_filter_data,
   uint32_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint32_t pred,
   std::size_t vector_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 3 ) * sizeof( uint8_t ) ) );

   auto const vector_stepwidth = vector_size >> 2;
   auto const vector_count = element_count / vector_stepwidth;
   auto const remainder = element_count % vector_stepwidth;

   auto result = _mm256_setzero_si256();

   for( auto vector = 0; vector < vector_count; ++vector ) {
         filter_eq_bitmask_sequential_avx2_32(
            filter_result, to_filter_data, vector_stepwidth, pred
         );

         result = aggregate_sum_bitmask_sequential_avx2_32(
            filter_result, to_agg_data, vector_stepwidth, result
         );
         to_filter_data += vector_stepwidth;
         to_agg_data += vector_stepwidth;
      }
   if( remainder != 0 ) {
         filter_eq_bitmask_sequential_avx2_32(
            filter_result, to_filter_data, remainder, pred
         );
         result = aggregate_sum_bitmask_sequential_avx2_32(
            filter_result, to_agg_data, remainder, result
         );
      }

   free( filter_result );
   return REDUCEADDi32( result );
}
//Query using bitmasks as intermediates, Vector at a time
uint32_t query_bitmask_vat_strided_avx2_32(
   uint32_t const * __restrict__ to_filter_data,
   uint32_t const * __restrict__ to_agg_data,
   std::size_t element_count,
   uint32_t pred,
   std::size_t vector_size,
   std::size_t stride_size
) {
   auto filter_result = reinterpret_cast< uint8_t * >( malloc( ( element_count >> 3 ) * sizeof( uint8_t ) ) );

   auto const vector_stepwidth = vector_size >> 2;
   auto const vector_count = element_count / vector_stepwidth;
   auto const remainder = element_count % vector_stepwidth;

   auto result = _mm256_setzero_si256();

   for( auto vector = 0; vector < vector_count; ++vector ) {
         filter_eq_bitmask_strided_avx2_32(
            filter_result, to_filter_data, vector_stepwidth, pred, stride_size
         );

         result = aggregate_sum_bitmask_strided_avx2_32(
            filter_result, to_agg_data, vector_stepwidth, result, stride_size
         );
         to_filter_data += vector_stepwidth;
         to_agg_data += vector_stepwidth;
      }
   if( remainder != 0 ) {
         filter_eq_bitmask_strided_avx2_32(
            filter_result, to_filter_data, remainder, pred, stride_size
         );
         result = aggregate_sum_bitmask_strided_avx2_32(
            filter_result, to_agg_data, remainder, result, stride_size
         );
      }

   free( filter_result );
   return REDUCEADDi32( result );
}


#endif //TOGATHERORNOT_INCLUDE_QUERY_HPP
