#pragma once

#include "rational_sparse_json.hpp"
#include <memory>

namespace atlas::monodromy_weyl {

// Reuse the Briançon–Maisonobe annihilator of f^s across root queries.
class Annihilator {
public:
    Annihilator(const sparse_json::Terms& f, std::size_t width);
    ~Annihilator();
    bool local_root(const mpq_class& alpha) const;

private:
    struct Data;
    std::unique_ptr<Data> data;
};

} // namespace atlas::monodromy_weyl
