#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cinttypes>
#include <random>
#include <cmath>
#include <codecvt>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <cwctype>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <regex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <valarray>
#include <vector>

template<int N> struct Fib { static constexpr int value = Fib<N-1>::value + Fib<N-2>::value; };
template<> struct Fib<0> { static constexpr int value = 0; };
template<> struct Fib<1> { static constexpr int value = 1; };

template<int N> struct Fact { static constexpr long long value = N * Fact<N-1>::value; };
template<> struct Fact<0> { static constexpr long long value = 1; };

template<typename T, int N> struct PowerTemplate { static constexpr T value = T(N) * PowerTemplate<T, N-1>::value; };
template<typename T> struct PowerTemplate<T, 0> { static constexpr T value = T(1); };

template<int N, int K> struct Binomial { static constexpr int value = Fact<N>::value / (Fact<K>::value * Fact<N-K>::value); };

template<typename T, T... Vals> struct Sum { static constexpr T value = (Vals + ...); };

template<int N> struct IsPrime {
    template<int I> struct Helper { static constexpr bool value = (N % I != 0) && Helper<I-1>::value; };
    template<> struct Helper<1> { static constexpr bool value = true; };
    static constexpr bool value = (N > 1) && Helper<N-1>::value;
};

template<typename...> struct TypeList {};

template<typename List> struct Length;
template<typename... Types> struct Length<TypeList<Types...>> { static constexpr int value = sizeof...(Types); };

template<int N, typename List> struct At;
template<int N, typename Head, typename... Tail>
struct At<N, TypeList<Head, Tail...>> { using type = At<N-1, TypeList<Tail...>>::type; };
template<typename Head, typename... Tail>
struct At<0, TypeList<Head, Tail...>> { using type = Head; };

template<typename T, typename List> struct Contains;
template<typename T, typename... Types>
struct Contains<T, TypeList<Types...>> { static constexpr bool value = (std::is_same_v<T, Types> || ...); };

template<typename List1, typename List2> struct Concat;
template<typename... Types1, typename... Types2>
struct Concat<TypeList<Types1...>, TypeList<Types2...>> { using type = TypeList<Types1..., Types2...>; };

template<template<typename> class F, typename List> struct Map;
template<template<typename> class F, typename... Types>
struct Map<F, TypeList<Types...>> { using type = TypeList<F<Types>...>; };

template<template<typename> class P, typename List> struct Filter;
template<template<typename> class P, typename Head, typename... Tail>
struct Filter<P, TypeList<Head, Tail...>> {
    using rest = Filter<P, TypeList<Tail...>>::type;
    using type = std::conditional_t<P<Head>::value, 
        typename Concat<TypeList<Head>, rest>::type, rest>;
};
template<template<typename> class P>
struct Filter<P, TypeList<>> { using type = TypeList<>; };

template<typename T> struct IsPointer { static constexpr bool value = false; };
template<typename T> struct IsPointer<T*> { static constexpr bool value = true; };

template<typename T> struct RemovePointer { using type = T; };
template<typename T> struct RemovePointer<T*> { using type = T; };

template<int N> struct StringLiteral { char data[N]; constexpr explicit StringLiteral(const char (&str)[N]) { std::copy_n(str, N, data); } };

template<StringLiteral S> struct CompileTimeString { static constexpr auto value = S; };

template<typename T, T V> struct IntegralConstant { static constexpr T value = V; using type = T; };

template<bool B> using BoolConstant = IntegralConstant<bool, B>;
using TrueType = BoolConstant<true>;
using FalseType = BoolConstant<false>;

template<typename T> struct TypeIdentity { using type = T; };

template<bool, typename T, typename> struct Conditional { using type = T; };
template<typename T, typename F> struct Conditional<false, T, F> { using type = F; };

template<typename> struct AlwaysTrue : TrueType {};
template<typename> struct AlwaysFalse : FalseType {};

template<typename From, typename To> struct IsConvertible {
    template<typename T> static auto test(T*) -> decltype(static_cast<To>(std::declval<T>()), TrueType{});
    template<typename> static FalseType test(...);
    static constexpr bool value = decltype(test<From>(nullptr))::value;
};

template<typename Base, typename Derived> struct IsBaseOf {
    template<typename T> static auto test(T*) -> decltype(static_cast<Base*>(static_cast<Derived*>(nullptr)), TrueType{});
    template<typename> static FalseType test(...);
    static constexpr bool value = decltype(test<Derived>(nullptr))::value;
};

template<typename T> struct HasIterator {
    template<typename U> static auto test(U*) -> decltype(std::declval<U>().begin(), std::declval<U>().end(), TrueType{});
    template<typename> static FalseType test(...);
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template<typename T> struct HasSize {
    template<typename U> static auto test(U*) -> decltype(std::declval<U>().size(), TrueType{});
    template<typename> static FalseType test(...);
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template<typename T> struct HasPushBack {
    template<typename U> static auto test(U*) -> decltype(std::declval<U>().push_back(std::declval<typename U::value_type>()), TrueType{});
    template<typename> static FalseType test(...);
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template<typename T, typename = void> struct VoidT { using type = void; };
template<typename T> using void_t = VoidT<T>::type;

template<typename T, typename = void> struct HasValueType : FalseType {};
template<typename T> struct HasValueType<T, void_t<typename T::value_type>> : TrueType {};

template<typename T, typename = void> struct HasKeyType : FalseType {};
template<typename T> struct HasKeyType<T, void_t<typename T::key_type>> : TrueType {};

template<typename T, typename = void> struct HasMappedType : FalseType {};
template<typename T> struct HasMappedType<T, void_t<typename T::mapped_type>> : TrueType {};

template<typename T> struct IsContainer : BoolConstant<HasIterator<T>::value && HasSize<T>::value> {};

template<typename T> struct IsAssociative : BoolConstant<HasKeyType<T>::value> {};

template<typename T> struct IsMap : BoolConstant<HasKeyType<T>::value && HasMappedType<T>::value> {};

template<typename T> struct IsSequence : BoolConstant<IsContainer<T>::value && !IsAssociative<T>::value> {};

template<typename T> constexpr bool is_container_v = IsContainer<T>::value;
template<typename T> constexpr bool is_associative_v = IsAssociative<T>::value;
template<typename T> constexpr bool is_map_v = IsMap<T>::value;
template<typename T> constexpr bool is_sequence_v = IsSequence<T>::value;

template<typename T> struct FunctionTraits;
template<typename R, typename... Args> struct FunctionTraits<R(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    template<size_t N> using arg_type = std::tuple_element_t<N, args_tuple>;
};

template<typename R, typename... Args> struct FunctionTraits<R(*)(Args...)> : FunctionTraits<R(Args...)> {};

template<typename C, typename R, typename... Args> struct FunctionTraits<R(C::*)(Args...)> : FunctionTraits<R(Args...)> {
    using class_type = C;
};

template<typename F> struct LambdaTraits;
template<typename C, typename R, typename... Args> struct LambdaTraits<R(C::*)(Args...) const> : FunctionTraits<R(Args...)> {};

template<typename F> struct FunctionTraits<F> : LambdaTraits<decltype(&F::operator())> {};

template<typename T, size_t N> struct ArrayWrapper { T data[N]; };

template<typename T, T... Values> struct ValueSequence {
    static constexpr size_t size = sizeof...(Values);
    static constexpr T values[sizeof...(Values)] = {Values...};
};

template<size_t... Is> using IndexSequence = ValueSequence<size_t, Is...>;

template<size_t N> struct MakeIndexSequence {
    template<size_t... Is> static IndexSequence<Is..., N-1> make(IndexSequence<Is...>);
    using type = decltype(make(typename MakeIndexSequence<N-1>::type{}));
};
template<> struct MakeIndexSequence<0> { using type = IndexSequence<>; };

template<size_t N> using make_index_sequence = MakeIndexSequence<N>::type;

template<typename... Ts> using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

template<typename Tuple, size_t... Is> constexpr auto tuple_to_array_impl(const Tuple& t, IndexSequence<Is...>) {
    using T = std::tuple_element_t<0, Tuple>;
    return std::array<T, sizeof...(Is)>{std::get<Is>(t)...};
}

template<typename... Ts> constexpr auto tuple_to_array(const std::tuple<Ts...>& t) {
    static_assert((std::is_same_v<Ts, std::tuple_element_t<0, std::tuple<Ts...>>> && ...));
    return tuple_to_array_impl(t, index_sequence_for<Ts...>{});
}

template<typename T, size_t N> struct Matrix {
    std::array<std::array<T, N>, N> data;
    constexpr Matrix() : data{} {}
    constexpr T& operator()(size_t i, size_t j) { return data[i][j]; }
    constexpr const T& operator()(size_t i, size_t j) const { return data[i][j]; }
};

template<typename T, size_t N> constexpr Matrix<T, N> identity_matrix() {
    Matrix<T, N> m;
    for (size_t i = 0; i < N; ++i) m(i, i) = T(1);
    return m;
}

template<typename T, size_t N> constexpr Matrix<T, N> multiply(const Matrix<T, N>& a, const Matrix<T, N>& b) {
    Matrix<T, N> result;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < N; ++k)
                result(i, j) += a(i, k) * b(k, j);
    return result;
}

template<int N> struct CompileTimeLoop {
    template<typename F> static constexpr void execute(F&& f) {
        f(IntegralConstant<int, N-1>{});
        CompileTimeLoop<N-1>::execute(std::forward<F>(f));
    }
};
template<> struct CompileTimeLoop<0> {
    template<typename F> static constexpr void execute(F&&) {}
};

template<typename T> struct Singleton {
    static T& instance() {
        static T inst;
        return inst;
    }
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
protected:
    Singleton() = default;
    ~Singleton() = default;
};

template<typename T> class LazyInitialized {
    mutable std::optional<T> value_;
    mutable std::function<T()> initializer_;
public:
    template<typename F> explicit LazyInitialized(F&& f) : initializer_(std::forward<F>(f)) {}
    const T& get() const {
        if (!value_) value_ = initializer_();
        return *value_;
    }
};

template<typename T, typename Deleter = std::default_delete<T>> class UniqueResource {
    T* resource_;
    Deleter deleter_;
public:
    explicit UniqueResource(T* r, Deleter d = Deleter{}) : resource_(r), deleter_(d) {}
    ~UniqueResource() { if (resource_) deleter_(resource_); }
    UniqueResource(const UniqueResource&) = delete;
    UniqueResource& operator=(const UniqueResource&) = delete;
    UniqueResource(UniqueResource&& other) noexcept : resource_(other.release()), deleter_(std::move(other.deleter_)) {}
    UniqueResource& operator=(UniqueResource&& other) noexcept {
        reset(other.release());
        deleter_ = std::move(other.deleter_);
        return *this;
    }
    T* get() const noexcept { return resource_; }
    T* release() noexcept { T* r = resource_; resource_ = nullptr; return r; }
    void reset(T* r = nullptr) { if (resource_) deleter_(resource_); resource_ = r; }
    T& operator*() const { return *resource_; }
    T* operator->() const { return resource_; }
    explicit operator bool() const noexcept { return resource_ != nullptr; }
};

template<typename... Fs> struct Overload : Fs... { using Fs::operator()...; };
template<typename... Fs> Overload(Fs...) -> Overload<Fs...>;

template<typename Variant, typename... Fs> constexpr decltype(auto) visit_overload(Variant&& v, Fs&&... fs) {
    return std::visit(Overload{std::forward<Fs>(fs)...}, std::forward<Variant>(v));
}

template<typename T> struct Tag { using type = T; };
template<typename T> constexpr Tag<T> tag{};

template<typename T> constexpr auto type_name() {
    std::string_view name = __PRETTY_FUNCTION__;
    const auto start = name.find('=') + 2;
    auto end = name.find(';', start);
    if (end == std::string_view::npos) end = name.find(']', start);
    return name.substr(start, end - start);
}

template<auto V> constexpr auto value_name() {
    std::string_view name = __PRETTY_FUNCTION__;
    const auto start = name.find('=') + 2;
    auto end = name.find(';', start);
    if (end == std::string_view::npos) end = name.find(']', start);
    return name.substr(start, end - start);
}

struct CompileTimeCounter {
    template<int N> struct Flag { friend constexpr int adl_flag(Flag); };
    template<int N> struct Writer { friend constexpr int adl_flag(Flag<N>) { return N; } static constexpr int value = N; };
    template<int N = 0> static constexpr int reader(int, Flag<N>) { return N; }
    template<int N = 0> static constexpr int reader(float, Flag<N>,
                                const int R = reader(0, Flag<N+1>{})) { return R; }
    template<int N = 0> static constexpr int reader(double, Flag<N>) { return N - 1; }
};

#define COUNTER CompileTimeCounter::next()

template<typename T> struct DebugType;

template<int N> struct ExponentialInstantiation {
    using type = ExponentialInstantiation<N-1>::type;
    static constexpr int value = ExponentialInstantiation<N-1>::value * 2;
};
template<> struct ExponentialInstantiation<0> { using type = int; static constexpr int value = 1; };

template<int N> struct NestedTemplateHell {
    template<int M> struct Inner {
        template<int K> struct DeepInner {
            template<int J> struct DeeperInner {
                template<int I> struct DeepestInner {
                    static constexpr int value = N + M + K + J + I;
                    using type = std::tuple<std::integral_constant<int, N>, 
                                           std::integral_constant<int, M>,
                                           std::integral_constant<int, K>,
                                           std::integral_constant<int, J>,
                                           std::integral_constant<int, I>>;
                };
            };
        };
    };
};

template<typename... Types> struct VariadicNightmare {
    template<typename... MoreTypes> struct Combine {
        template<typename... EvenMoreTypes> struct DeepCombine {
            using type = std::tuple<Types..., MoreTypes..., EvenMoreTypes...>;
            static constexpr size_t size = sizeof...(Types) + sizeof...(MoreTypes) + sizeof...(EvenMoreTypes);
            template<size_t N> using at = std::tuple_element_t<N, type>;
        };
    };
};

template<typename T> struct RecursiveWrapper { T value;
  explicit RecursiveWrapper(T v) : value(v) {} };

template<int N> struct SFINAEHell {
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    static auto test1(T) -> std::true_type;
    static auto test1(...) -> std::false_type;
    
    template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    static auto test2(T) -> std::true_type;
    static auto test2(...) -> std::false_type;
    
    template<typename T, typename = std::enable_if_t<std::is_pointer_v<T>>>
    static auto test3(T) -> std::true_type;
    static auto test3(...) -> std::false_type;
    
    template<typename T> using is_integral = decltype(test1(std::declval<T>()));
    template<typename T> using is_floating = decltype(test2(std::declval<T>()));
    template<typename T> using is_pointer = decltype(test3(std::declval<T>()));
    
    static constexpr int value = N;
};

using MegaType = VariadicNightmare<int, double, float, char, long, short, unsigned, signed char, 
                                  std::string, std::vector<int>, std::map<int, std::string>,
                                  std::unique_ptr<int>, std::shared_ptr<double>, std::weak_ptr<float>,
                                  std::function<int(double)>, std::tuple<int, double, std::string>,
                                  std::array<int, 100>, std::deque<std::string>, std::list<double>,
                                  std::set<int>, std::unordered_set<std::string>, std::queue<int>,
                                  std::stack<double>, std::priority_queue<int>, std::bitset<64>,
                                  std::complex<double>, std::valarray<int>, std::pair<int, std::string>,
                                  std::optional<int>, std::variant<int, double, std::string>,
                                  std::any, std::type_info, std::exception, std::runtime_error>::
                 Combine<std::logic_error, std::domain_error, std::invalid_argument, std::length_error,
                        std::out_of_range, std::future_error, std::system_error, std::ios_base::failure,
                        std::bad_alloc, std::bad_array_new_length, std::bad_cast, std::bad_exception,
                        std::bad_function_call, std::bad_typeid, std::bad_weak_ptr, std::regex_error,
                        std::thread, std::mutex, std::recursive_mutex, std::timed_mutex,
                        std::recursive_timed_mutex, std::shared_mutex, std::condition_variable,
                        std::condition_variable_any, std::once_flag, std::atomic<int>,
                        std::atomic<double>, std::atomic<bool>, std::atomic_flag, std::memory_order>::
                 DeepCombine<std::chrono::nanoseconds, std::chrono::microseconds, std::chrono::milliseconds,
                            std::chrono::seconds, std::chrono::minutes, std::chrono::hours,
                            std::chrono::system_clock, std::chrono::steady_clock, std::chrono::high_resolution_clock,
                            std::random_device, std::mt19937, std::mt19937_64, std::uniform_int_distribution<>,
                            std::uniform_real_distribution<>, std::normal_distribution<>,
                            std::bernoulli_distribution, std::binomial_distribution<>, std::geometric_distribution<>,
                            std::negative_binomial_distribution<>, std::poisson_distribution<>,
                            std::exponential_distribution<>, std::gamma_distribution<>,
                            std::weibull_distribution<>, std::extreme_value_distribution<>,
                            std::chi_squared_distribution<>, std::cauchy_distribution<>,
                            std::fisher_f_distribution<>, std::student_t_distribution<>,
                            std::piecewise_constant_distribution<>,
                            std::piecewise_linear_distribution<>, std::seed_seq, std::random_device>;

template<int... Is> constexpr auto make_mega_instantiation() {
    return std::make_tuple(
        ExponentialInstantiation<Is>::value...,
        NestedTemplateHell<Is>::template Inner<Is>::template DeepInner<Is>::template DeeperInner<Is>::template DeepestInner<Is>::value...,
        SFINAEHell<Is>::value...,
        Fib<Is>::value...,
        Fact<Is>::value...,
        PowerTemplate<double, Is>::value...,
        IsPrime<Is>::value...
    );
}


template<typename T>
void CleanOnce(T& arg) {
    if constexpr (std::is_pointer_v<T>) {
        arg = nullptr;
    } else if constexpr (requires { arg.clear(); }) {
        arg.clear();
    } else if constexpr (std::is_fundamental_v<T>) {
        arg = static_cast<T>(0);
    }
}

template<typename... Args>
void Clean(Args&... args) {
    (CleanOnce(args), ...);
}

int main() {
    constexpr auto fib_values = make_mega_instantiation<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30>();
    constexpr auto more_values = make_mega_instantiation<31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60>();
    constexpr auto even_more = make_mega_instantiation<61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90>();

    constexpr auto identity = identity_matrix<double, 10>();
    constexpr auto multiplied = multiply(identity, identity);
    
    CompileTimeLoop<50>::execute([]<typename T0>() {
        constexpr int val = T0::value;
        [[maybe_unused]] constexpr auto nested = NestedTemplateHell<val>::template Inner<val>::template DeepInner<val>::template DeeperInner<val>::template DeepestInner<val>::value;
    });

    Clean(fib_values, more_values, even_more, identity, multiplied);

    return 0;
}
