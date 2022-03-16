
#ifndef TOGATHERORNOT_INCLUDE_UTILS_HPP
#define TOGATHERORNOT_INCLUDE_UTILS_HPP


#include <cstdint>
#include <cstddef>
#include <tuple>
#include <array>
#include <immintrin.h>


template< typename T >
auto get_end_pos(
   T const * __restrict__ data,
   std::size_t element_count,
   std::size_t vector_shift
) {
   return std::make_tuple(
      data + ( ( element_count >> vector_shift ) << vector_shift ), data + element_count
   );
}


void print(__m256i i ) {
   alignas(32) std::array< uint32_t, 8 > tmp{};
   _mm256_store_si256( reinterpret_cast< __m256i* >( tmp.data() ), i );
   for( std::size_t i = 0; i < 8; ++i ) {
         std::cout << tmp[ 7-i ] << "\t";
      }
   std::cout << "\n";
}


#endif //TOGATHERORNOT_INCLUDE_UTILS_HPP
