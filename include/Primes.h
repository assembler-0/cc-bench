#ifndef PRIMES_H
#define PRIMES_H

#include <cstddef>

namespace ccbench {

// Function to check if a number is prime at compile time
constexpr bool is_prime(const size_t n) {
    if (n <= 1) return false;
    for (size_t i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// Get the Nth prime number
template <size_t N>
struct NthPrime {
    static constexpr size_t value = []() {
        size_t count = 0;
        size_t num = 1;
        while (count < N + 1) {
            num++;
            if (is_prime(num)) {
                count++;
            }
        }
        return num;
    }();
};

} // namespace CompilerBurner

#endif // PRIMES_H
