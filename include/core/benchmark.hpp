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
 * @file benchmark.hpp
 * @author jpietrzyk
 * @date 15.03.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef BENCHMARK_INCLUDE_CORE_BENCHMARK_HPP
#define BENCHMARK_INCLUDE_CORE_BENCHMARK_HPP

#include <tuple>
#include <cstddef>
#include <type_traits>
#include <string>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <utils/print.hpp>
#include <utils/clock.hpp>


#include <iostream>
namespace tuddbs { namespace benchmark {

class benchmark_ctx {
public:
  static benchmark_ctx& getInstance() {
    static benchmark_ctx instance;
    return instance;
  }
  using stopwatch_type = tuddbs::benchmark::stopwatch_t< long double, std::chrono::high_resolution_clock, std::nano >;
  stopwatch_type create_watch() const {
    return stopwatch_type{};
  }
private:
  benchmark_ctx() = default;
  ~benchmark_ctx() = default;
  benchmark_ctx(benchmark_ctx const &)=delete;
  benchmark_ctx& operator=(benchmark_ctx const &)=delete;
};

template< typename T >
struct variable_t {
  using type = T;
  std::string name;
  T value;
  template< typename U >
    friend std::ostream & operator<<(std::ostream&, variable_t< U > const &);
};
template< typename T >
std::ostream &operator<<(std::ostream & out, variable_t< T > const & parameter) {
  out << parameter.name << ": { type: " << type_name< T >() << ", value: " << parameter.value << " }";
  return out;
}

namespace details {
  template< typename FirstValueType, typename... RemainderTypes >
  auto unpack_to_variables(const std::string& n, FirstValueType v, RemainderTypes&& ... args) {
    static_assert(sizeof...(RemainderTypes) % 2 == 0, "Number of arguments must be even!");
    if constexpr( sizeof...(RemainderTypes) > 0 ) {
      return std::tuple_cat(
          std::make_tuple(variable_t<FirstValueType>{.name=(n), .value=(v)}),
          unpack_to_variables(args...)
      );
    } else {
      return std::make_tuple(variable_t<FirstValueType>{.name=(n), .value=(v)});
    }
  }
  template< typename Tuple, std::size_t Idx = 0 >
  auto extract_variable_values(Tuple&& types) {
    if constexpr(std::tuple_size_v< Tuple> > 0 ) {
      if constexpr(Idx<(std::tuple_size_v< Tuple >-1)) {
        return std::tuple_cat(std::make_tuple(std::get<Idx>(types).value), extract_variable_values<Tuple, Idx+1>(std::move(types)));
      } else {
        return std::make_tuple(std::get<Idx>(types).value);
      }
    } else {
      return std::make_tuple();
    }
  }
  template<typename BenchmarkFunction,typename VariableTupleType, typename FinalizeFunction = std::nullptr_t>
  struct benchmark_types_helper_t {
    template<typename FF, class Enable = void, typename... Args> struct finalizer_impl{
      using invoke_result_type = void;
    };
    template<typename FF, typename... Args>
    struct finalizer_impl<FF, typename std::enable_if_t<!std::is_null_pointer_v<FF>>, Args...> {
      using invoke_result_type = std::invoke_result_t<FF, Args... >;
    };

    template<typename=std::make_index_sequence<std::tuple_size_v<VariableTupleType>>>
    struct params_type_impl;
    template<std::size_t... Idx>
    struct params_type_impl<std::index_sequence<Idx...>> {
      template<std::size_t I>
      using wrap = typename std::tuple_element_t<I,VariableTupleType>::type;
      using parameters_type = std::tuple<wrap<Idx>...>;
      using invoke_result_type = std::invoke_result_t<BenchmarkFunction, wrap<Idx>... >;
      using finalizer_result_type =
          std::conditional_t<
            std::is_void_v<invoke_result_type>,
            typename finalizer_impl<FinalizeFunction, wrap<Idx>...>::invoke_result_type,
            typename finalizer_impl<FinalizeFunction, invoke_result_type, wrap<Idx>...>::invoke_result_type
          >;
    };
    using parameters_type = typename params_type_impl<>::parameters_type;
    using function_return_type = typename params_type_impl<>::invoke_result_type;
    static_assert(
        !(std::is_void_v<function_return_type> && std::is_null_pointer_v<FinalizeFunction>),
        "Finalizer Function has to be specified if benchmark function does not return a result."
    );

    using finalizer_return_type = typename params_type_impl<>::finalizer_result_type;
    static_assert(
        (!std::is_void_v<function_return_type> || !std::is_void_v<finalizer_return_type>),
        "Either the benchmark function or the finalizer must return *something*."
    );
    using benchmark_return_type =
        typename
        std::conditional_t<
          std::is_void_v<finalizer_return_type>,
          function_return_type,
          finalizer_return_type
        >;


  };


template<
    typename BenchmarkReturnType,
    typename BenchmarkFunction,
    typename BenchmarkParamsTuple,
    typename FinalizeFunction = std::nullptr_t
>
auto benchmark_execute_single_with_pcm(std::ostream & oss, BenchmarkFunction && fun, BenchmarkParamsTuple&& params, FinalizeFunction && fun_finalize = nullptr) {
  auto watch = benchmark_ctx::getInstance().create_watch();
  if constexpr(std::is_null_pointer_v<FinalizeFunction>) {
    watch.start();
    //PCM CODE GOES HERE
    volatile auto result = std::apply(fun, params);
    //PCM CODE GOES HERE
    watch.stop();
    oss << "initial_run_time: " << watch << "\n";
    oss << "initial_result: " << result << "\n";
    return result;
  } else {
    if constexpr(std::is_void_v<BenchmarkReturnType>) {
      watch.start();
      //PCM CODE GOES HERE
      std::apply(fun, params);
      //PCM CODE GOES HERE
      watch.stop();
      auto result = std::apply(fun_finalize, params);
      oss << "initial_run_time: " << watch << "\n";
      oss << "initial_result: " << result << "\n";
      return result;
    } else {
      watch.start();
      //PCM CODE GOES HERE
      volatile auto result = std::apply(fun, params);
      //PCM CODE GOES HERE
      watch.stop();
      auto result_final = std::apply(fun_finalize, std::tuple_cat(std::make_tuple(result),params));
      oss << "initial_run_time: " << watch << "\n";
      oss << "initial_result: " << result_final << "\n";
      return result_final;
    }
  }
}
template<
    typename BenchmarkReturnType,
    typename BenchmarkFunction,
    typename BenchmarkParamsTuple,
    typename FinalizeFunction = std::nullptr_t
>
auto benchmark_execute_single(BenchmarkFunction && fun, BenchmarkParamsTuple&& params, FinalizeFunction && fun_finalize = nullptr) {
  auto watch = benchmark_ctx::getInstance().create_watch();
  if constexpr(std::is_null_pointer_v<FinalizeFunction>) {
    watch.start();
    volatile auto result = std::apply(fun, params);
    watch.stop();
    return std::make_tuple(watch, result);
  } else {
    if constexpr(std::is_void_v<BenchmarkReturnType>) {
      watch.start();
      std::apply(fun, params);
      watch.stop();
      return std::make_tuple(watch,std::apply(fun_finalize, params));
    } else {
      watch.start();
      volatile auto result = std::apply(fun, params);
      watch.stop();
      return std::make_tuple(watch, std::apply(fun_finalize, std::tuple_cat(std::make_tuple(result),params)));
    }
  }
}
}

template< typename BenchmarkFunction, typename... BenchmarkParams, typename... BenchmarkSpecs, typename FinalizeFunction = std::nullptr_t >
std::string benchmark_impl(const std::string& function_name, BenchmarkFunction && fun, std::tuple< BenchmarkParams... >&& params, std::tuple< BenchmarkSpecs... >&& specs, FinalizeFunction && fun_finalize = nullptr ) {
  //Sanity checking
  using type_helper = details::benchmark_types_helper_t<BenchmarkFunction, std::tuple< BenchmarkParams... >, FinalizeFunction>;

  auto function_parameters = details::extract_variable_values(std::move(params));
  std::cout << "--- !Benchmark\n";
  std::cout << "start_time: " << tuddbs::now_to_string() << "\n";
  std::cout << "name: " << function_name << "\n"
               "type: " << TYPENAME(fun) << "\n"
               "parameters:\n";
  std::apply([](auto&&... args) {((std::cout << "   " << args << "\n"), ...);}, params);
  std::cout << "specifics: \n";
  std::apply([](auto&&... args) {((std::cout << "   " << args << "\n"), ...);}, specs);
  std::cout << "time_measurement_unit: " << benchmark_ctx::getInstance().create_watch().resolution_to_str() << "\n";
  std::cout << "number_of_threads: 1\n";

  details::benchmark_execute_single_with_pcm<typename type_helper::function_return_type, BenchmarkFunction, typename type_helper::parameters_type, FinalizeFunction>(
      std::cout,
      std::forward<BenchmarkFunction>(fun),
      std::forward<decltype(function_parameters)>(function_parameters),
      std::forward<FinalizeFunction>(fun_finalize)
  );

  std::vector<benchmark_ctx::stopwatch_type > measurements{};
  std::vector<typename type_helper::benchmark_return_type> results{};

  for(auto rep = 0; rep < 20; ++rep) {
    auto result =
      details::benchmark_execute_single<typename type_helper::function_return_type, BenchmarkFunction, typename type_helper::parameters_type, FinalizeFunction>(
          std::forward<BenchmarkFunction>(fun),
          std::forward<decltype(function_parameters)>(function_parameters),
          std::forward<FinalizeFunction>(fun_finalize)
      );
    measurements.emplace_back(std::move(std::get<0>(result)));
    results.emplace_back(std::move(std::get<1>(result)));
  }
  print_vector(std::cout, "measurements", measurements, [](auto const watch){ return watch.time_elapsed();});
  print_vector(std::cout, "results", results, [](auto const result){ return result;});

  std::cout << "end_time: " << tuddbs::now_to_string() << "\n";
  std::cout << "...\n";
  return "";  //return value can be used to calculate improvements to baseline otf.
}
template< typename BenchmarkFunction, typename... BenchmarkParams, typename... BenchmarkSpecs, typename FinalizeFunction = std::nullptr_t >
  std::string benchmark_multithread_impl(const std::string& function_name, BenchmarkFunction && fun, std::tuple< BenchmarkParams... >&& params, std::tuple< BenchmarkSpecs... >&& specs, FinalizeFunction && fun_finalize = nullptr ) {
    //Sanity checking
    using type_helper = details::benchmark_types_helper_t<BenchmarkFunction, std::tuple< BenchmarkParams... >, FinalizeFunction>;

    auto function_parameters = details::extract_variable_values(std::move(params));
    std::cout << "--- !Benchmark\n";
    std::cout << "start_time: " << tuddbs::now_to_string() << "\n";
    std::cout << "name: " << function_name << "\n"
                                              "type: " << TYPENAME(fun) << "\n"
                                                                           "parameters:\n";
    std::apply([](auto&&... args) {((std::cout << "   " << args << "\n"), ...);}, params);
    std::cout << "specifics: \n";
    std::apply([](auto&&... args) {((std::cout << "   " << args << "\n"), ...);}, specs);
    std::cout << "time_measurement_unit: " << benchmark_ctx::getInstance().create_watch().resolution_to_str() << "\n";
    std::cout << "number_of_threads: 1\n";

    details::benchmark_execute_single_with_pcm<typename type_helper::function_return_type, BenchmarkFunction, typename type_helper::parameters_type, FinalizeFunction>(
        std::cout,
        std::forward<BenchmarkFunction>(fun),
        std::forward<decltype(function_parameters)>(function_parameters),
        std::forward<FinalizeFunction>(fun_finalize)
    );

    std::vector<benchmark_ctx::stopwatch_type > measurements{};
    std::vector<typename type_helper::benchmark_return_type> results{};

    for(auto rep = 0; rep < 20; ++rep) {
      auto result =
          details::benchmark_execute_single<typename type_helper::function_return_type, BenchmarkFunction, typename type_helper::parameters_type, FinalizeFunction>(
              std::forward<BenchmarkFunction>(fun),
              std::forward<decltype(function_parameters)>(function_parameters),
              std::forward<FinalizeFunction>(fun_finalize)
          );
      measurements.emplace_back(std::move(std::get<0>(result)));
      results.emplace_back(std::move(std::get<1>(result)));
    }
    print_vector(std::cout, "measurements", measurements, [](auto const watch){ return watch.time_elapsed();});
    print_vector(std::cout, "results", results, [](auto const result){ return result;});

    std::cout << "end_time: " << tuddbs::now_to_string() << "\n";
    std::cout << "...\n";
    return "";  //return value can be used to calculate improvements to baseline otf.
  }


}}
#define BENCHMARK_PARAMETERS(...) (tuddbs::benchmark::details::unpack_to_variables(__VA_ARGS__))
#define BENCHMARK_SPECS(...) (tuddbs::benchmark::details::unpack_to_variables(__VA_ARGS__))
#define BENCHMARK(fun, ...) tuddbs::benchmark::benchmark_impl(#fun, fun, __VA_ARGS__)

#endif //BENCHMARK_INCLUDE_CORE_BENCHMARK_HPP
