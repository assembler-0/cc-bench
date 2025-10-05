#ifndef CCBENCH_H
#define CCBENCH_H

#include <array>
#include <string_view>

namespace ccbench {

// Compile-time string
template <size_t N>
struct ConstexprString {
    std::array<char, N> data{};

    constexpr explicit ConstexprString(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }

    [[nodiscard]] constexpr auto operator<=>(const ConstexprString&) const = default;
};

// Factorial
template <size_t N>
struct Factorial {
    static constexpr size_t value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    static constexpr size_t value = 1;
};

// Fibonacci
template <size_t N>
struct Fibonacci {
    static constexpr size_t value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template <>
struct Fibonacci<0> {
    static constexpr size_t value = 0;
};

template <>
struct Fibonacci<1> {
    static constexpr size_t value = 1;
};

// TypeList for metaprogramming
template <typename...>
struct TypeList {};

// Compile-time Matrix
template <typename T, size_t Rows, size_t Cols>
class Matrix {
public:
    std::array<std::array<T, Cols>, Rows> data{};

    constexpr Matrix() = default;

    constexpr T& at(size_t r, size_t c) {
        return data[r][c];
    }

    [[nodiscard]] constexpr const T& at(size_t r, size_t c) const {
        return data[r][c];
    }

    template <size_t OtherCols>
    constexpr auto operator*(const Matrix<T, Cols, OtherCols>& other) const {
        Matrix<T, Rows, OtherCols> result{};
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < OtherCols; ++j) {
                for (size_t k = 0; k < Cols; ++k) {
                    result.at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return result;
    }
};

}

#endif // CCBENCH_H
