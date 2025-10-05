#ifndef SORT_H
#define SORT_H

#include <array>

namespace ccbench {

// Compile-time bubble sort
template <typename T, size_t N>
constexpr std::array<T, N> bubble_sort(std::array<T, N> arr) {
    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = 0; j < N - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                T temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

}

#endif // SORT_H
