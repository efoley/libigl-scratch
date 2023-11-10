#pragma once

#include <stdexcept>

// debugging stuff
// need 2 macros so that any macro passed as argument will be expanded
// before being stringified
#define _STRINGIZE_DETAIL(x) #x
#define _STRINGIZE(x) _STRINGIZE_DETAIL(x)

#define CHECK(expr)                                                            \
  _check(expr,                                                                 \
         "(CHECK failed at " __FILE__ ":" _STRINGIZE(__LINE__) "): " #expr)

#define DEBUG_CHECK(expr) CHECK(expr)

#define FAIL(msg) _fail("(FAIL at " __FILE__ ":" _STRINGIZE(__LINE__) "): " msg)

#define UNREACHABLE()                                                          \
  _unreachable("(UNREACHABLE at " __FILE__ ":" _STRINGIZE(__LINE__) ")")

namespace edf {
inline void _check(bool b, const char *msg) {
  if (!b) {
    throw std::runtime_error(msg);
  }
}

[[noreturn]] inline void _fail(const char *msg) {
  throw std::runtime_error(msg);
}

[[noreturn]] inline void _unreachable(const char *msg) {
  throw std::runtime_error(msg);
}

template <typename Iterable, typename Callable>
void enumerate(const Iterable &iterable, Callable func) {
  size_t index = 0;
  for (const auto &element : iterable) {
    func(index, element);
    ++index;
  }
}

} // namespace edf  