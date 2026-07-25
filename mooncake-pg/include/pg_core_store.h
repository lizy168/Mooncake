#ifndef MOONCAKE_PG_CORE_STORE_H
#define MOONCAKE_PG_CORE_STORE_H

#include <cstdint>
#include <string>
#include <vector>

namespace mooncake {

class CoreStore {
   public:
    virtual ~CoreStore() = default;

    virtual bool check(const std::vector<std::string>& keys) = 0;
    virtual std::vector<uint8_t> get(const std::string& key) = 0;
    virtual std::string getString(const std::string& key) = 0;
    virtual void set(const std::string& key, const std::vector<uint8_t>& value) = 0;
    virtual void set(const std::string& key, const std::string& value) = 0;
    virtual void deleteKey(const std::string& key) = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_CORE_STORE_H
