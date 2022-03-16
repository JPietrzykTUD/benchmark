// ------------------------------------------------------------------- //
/*
   This file is part of the benchmark Project.
   Copyright (c) 2022 Johannes Pietrzyk.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, version 3.
 
   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.
 
   You should have received a copy of the GNU General Public License 
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// ------------------------------------------------------------------- //

/*
 * @file print.hpp
 * @author jpietrzyk
 * @date 15.03.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef BENCHMARK_INCLUDE_UTILS_PRINT_HPP
#define BENCHMARK_INCLUDE_UTILS_PRINT_HPP
#include <cstddef>
#include <type_traits>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <string>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace tuddbs { namespace benchmark {
template< class T >
std::string type_name( ) {
  typedef typename std::remove_reference< T >::type TR;
  std::unique_ptr< char, void ( * )( void * ) > own (
      abi::__cxa_demangle( typeid( TR ).name( ), nullptr,nullptr, nullptr ),
      std::free
  );
  std::string r = own != nullptr ? own.get( ) : typeid( TR ).name( );
  if( std::is_const< TR >::value ) {
    r += " const";
  }
  if( std::is_volatile< TR >::value ) {
    r += " volatile";
  }
  if( std::is_lvalue_reference< T >::value ) {
    r += "&";
  } else if( std::is_rvalue_reference< T >::value ) {
    r += "&&";
  }
  return r;
}
}
std::string now_to_string() {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm = *std::localtime(&now_c);
  std::ostringstream oss;
  oss << std::put_time(&now_tm, "%d.%m.%Y %H:%M:%S");
  return oss.str();
}
template< typename T, typename PrintLambda >
void print_vector(std::ostream & oss, std::string const & name, std::vector< T > const & data, PrintLambda&& fun ) {
  oss << name << ": [";
  bool start = true;
  for(auto const d: data) {
    if(start) {
      start = false;
    } else {
      oss << ", ";
    }
    oss << fun(d);
  }
  oss << "]\n";
}
}
#define TYPENAME( x ) tuddbs::benchmark::type_name< decltype( x ) >( )

#endif //BENCHMARK_INCLUDE_UTILS_PRINT_HPP
