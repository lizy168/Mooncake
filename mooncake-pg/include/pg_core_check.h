#ifndef MOONCAKE_PG_CORE_CHECK_H
#define MOONCAKE_PG_CORE_CHECK_H

#include <sstream>
#include <stdexcept>
#include <utility>

namespace mooncake {

template <typename... Args>
inline void coreCheck(bool condition, Args&&... args) {
    if (condition) return;
    std::ostringstream message;
    (message << ... << std::forward<Args>(args));
    throw std::runtime_error(message.str());
}

}  // namespace mooncake

#define MOONCAKE_CORE_CHECK(condition, ...) \
    ::mooncake::coreCheck((condition), __VA_ARGS__)

#endif  // MOONCAKE_PG_CORE_CHECK_H
