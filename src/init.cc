#include "ccbench.h"
#include "Primes.h"
#include "Sort.h"
#include <iostream>

template<typename T>
void clean_single(T& arg) {
    if constexpr (std::is_pointer_v<T>) {
        arg = nullptr;
    } else if constexpr (requires { arg.clear(); }) {
        arg.clear();
    } else if constexpr (std::is_fundamental_v<T>) {
        (void)static_cast<T>(0);
    }
}

template<typename... Args>
void cleanup(Args&... args) {
    (clean_single(args), ...);
}

int main() {
    // String
    constexpr ccbench::ConstexprString hello("Hello, World!");
    // Factorial
    constexpr auto factorial_15 = ccbench::Factorial<4096>::value;
    constexpr auto fib_400 = ccbench::Fibonacci<4096>::value;
    // Matrix multiplication
    constexpr auto m1 = [] {
        ccbench::Matrix<int, 3, 3> m;
        m.at(0, 0) = 1; m.at(0, 1) = 2; m.at(0, 2) = 3;
        m.at(1, 0) = 4; m.at(1, 1) = 5; m.at(1, 2) = 6;
        m.at(2, 0) = 7; m.at(2, 1) = 8; m.at(2, 2) = 9;
        return m;
    }();
    constexpr auto m2 = [] {
        ccbench::Matrix<int, 3, 3> m;
        m.at(0, 0) = 9; m.at(0, 1) = 8; m.at(0, 2) = 7;
        m.at(1, 0) = 6; m.at(1, 1) = 5; m.at(1, 2) = 4;
        m.at(2, 0) = 3; m.at(2, 1) = 2; m.at(2, 2) = 1;
        return m;
    }();
    constexpr auto m3 = m1 * m2;
    // Primes
    constexpr auto prime_1000 = ccbench::NthPrime<4096>::value;
    // Sorting
    constexpr std::array unsorted_arr = {9, 5, 2, 7, 1, 8, 4, 6, 3, 0};
    constexpr auto sorted_arr = ccbench::bubble_sort(unsorted_arr);

    cleanup(hello, factorial_15, fib_400, m1, m2, m3, prime_1000, unsorted_arr, sorted_arr);

    return 0;
}
