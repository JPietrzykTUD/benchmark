#include <iostream>

#include <core/benchmark.hpp>
#include <test/query.hpp>
#include <random>
#include <immintrin.h>

template< typename T >
class data_generator {
private:
  std::vector< T* > _data_ptr;
  std::mt19937 _engine;
public:
  data_generator(): _data_ptr{}, _engine{0x401C0FFEE401BABE}{}
  ~data_generator() {
    for( auto it = _data_ptr.rbegin(); it != _data_ptr.rend(); ++it ) {
      _mm_free( *it );
    }
  }

  void reset() {
    for( auto it = _data_ptr.rbegin(); it != _data_ptr.rend(); ++it ){
      _mm_free( *it );
    }
    _data_ptr.clear();
  }

  T const * generate_uniform( std::size_t bytecount, T min_inclusive, T max_inclusive ) {
    auto const element_count = bytecount / sizeof( T );
    std::cerr << "Generating " << bytecount << " bytes data (" << element_count << " Elements).\n";
    auto result = reinterpret_cast< T * >( _mm_malloc( bytecount, 64 ) );
    _data_ptr.push_back( result );
    using dist_t =
    std::conditional_t<
        std::is_integral_v< T >,
        std::uniform_int_distribution< T >,
        std::uniform_real_distribution< T >
    >;
    dist_t dist( min_inclusive, max_inclusive );
    std::generate( result, result + element_count, [&, this]{ return dist(this->_engine);});
    return result;
  }

  T const * generate_spiked(
      std::size_t bytecount,
      T spike_range_min_inclusive,
      T spike_range_max_inclusive,
      double proportion
  ) {
    auto const element_count = bytecount / sizeof( T );
    auto const range_count = (std::size_t) ((double)element_count * proportion);
    std::cerr << "Generating " << bytecount << " bytes data (" << element_count << " Elements).\n";
    auto result = reinterpret_cast< T * >( _mm_malloc( bytecount, 64 ) );
    _data_ptr.push_back( result );

    using dist_t =
    std::conditional_t<
        std::is_integral_v< T >,
        std::uniform_int_distribution< T >,
        std::uniform_real_distribution< T >
    >;
    dist_t dist_spike( spike_range_min_inclusive, spike_range_max_inclusive );
    dist_t dist_remainder( spike_range_max_inclusive, std::numeric_limits< T >::max() - 1 );
    std::generate( result, result + range_count, [&, this]{ return dist_spike(this->_engine);});
    std::generate( result + range_count, result + element_count, [&, this]{ return dist_remainder(this->_engine);});
    std::shuffle(result, result+element_count, this->_engine);
    //todo: SHUFFLE
    return result;
  }
};
#define WOLO(x) std::string y = #x

int main() {
  auto const data_size = 1<<29;
  data_generator< uint64_t > gen{};
  auto agg_data = gen.generate_uniform( data_size, 1, 10 );

  for( auto sel = 0.1; sel < 0.5; sel += 0.1) {
    auto filter_data = gen.generate_spiked(data_size, 10, 10, 0.1);
    auto const data_count = data_size >> 3;
    BENCHMARK(query_scalar<uint64_t>, BENCHMARK_PARAMETERS("filter_data", filter_data, "aggregation_data", agg_data, "element_count", data_count, "predicate", 10), BENCHMARK_SPECS("selectivity", sel));
//    _mm_free( filter_data );
  }
//  WOLO([](){std::cout << "Hello";});
//  std::cout << y << "\n";
  return 0;

}
