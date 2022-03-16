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
 * @file clock.hpp
 * @author jpietrzyk
 * @date 15.03.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef BENCHMARK_INCLUDE_UTILS_CLOCK_HPP
#define BENCHMARK_INCLUDE_UTILS_CLOCK_HPP

#include <chrono>
#include <ostream>

namespace tuddbs{ namespace benchmark {

template<typename DurationType = long double, typename Clock = std::chrono::high_resolution_clock, typename Resolution = std::nano>
class stopwatch_t {
public:
  using duration_type = DurationType;
  using time_point_type = std::chrono::time_point< Clock >;
  using resolution_type = Resolution;
private:
  time_point_type m_start;
  time_point_type m_end;
public:
  void start() {
    m_start = Clock::now();
  }
  void stop() {
    m_end = Clock::now();
  }
  duration_type time_elapsed() const {
    return std::chrono::duration<duration_type, resolution_type>(m_end-m_start).count();
  }
  std::string resolution_to_str() const noexcept {
    if constexpr( std::is_same_v< Resolution, std::nano > ) {
      return "ns";
    }  else if constexpr( std::is_same_v< Resolution, std::micro > ) {
      return "us";
    } else if constexpr( std::is_same_v< Resolution, std::milli > ) {
      return "ms";
    } else if constexpr( std::is_same_v< Resolution, std::ratio< 1 > > ) {
      return "s";
    }
    return "";
  }
  template< typename DT, typename C, typename R >
  friend std::ostream & operator<<(std::ostream &, stopwatch_t< DT, C, R > const &);
};
template<typename DurationType = long double, typename Clock = std::chrono::high_resolution_clock, typename Resolution = std::nano>
std::ostream & operator<<(std::ostream & oss, stopwatch_t<DurationType, Clock, Resolution> const & watch) {
  oss << watch.time_elapsed();
  return oss;
}
}
}

#endif //BENCHMARK_INCLUDE_UTILS_CLOCK_HPP
