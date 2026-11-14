#pragma once
#include "local_io.hpp"
#include <json/value.h>
#include <numeric>
#include <set>

namespace atlas::chart_rows {
using local::Basis;
using local::Polynomial;
using local::Powers;
using V = Json::Value;
using Axes = std::vector<size_t>;
inline V array() {
    return V(Json::arrayValue);
}
inline V rows(const Basis &b) {
    V out = array();
    for (const auto &f : b)
        out.append(local::encode_row(f));
    return out;
}
inline Basis parse(const V &data, size_t n) {
    Basis out;
    for (const auto &f : data)
        out.push_back(local::decode_row(f.asString(), n));
    return out;
}
inline Polynomial var(size_t k, size_t n) {
    Powers p(n);
    p.at(k) = 1;
    return {{p, 1}};
}
inline Polynomial one(size_t n) {
    return {{Powers(n), 1}};
}
inline Axes ids(size_t n) {
    Axes a(n);
    std::iota(a.begin(), a.end(), 0);
    return a;
}
inline V indices(const Axes &axes) {
    V a = array();
    for (auto k : axes)
        a.append(Json::UInt(k));
    return a;
}
inline Axes support(const Basis &rows, size_t n) {
    std::set<size_t> found;
    for (const auto &f : rows)
        for (const auto &[p, c] : f)
            for (size_t j = 0; j < n; ++j)
                if (p[j] != 0)
                    found.insert(j);
    return {found.begin(), found.end()};
}
// Project and reorder only when all omitted coordinates are known to be absent.
inline Polynomial project(const Polynomial &f, const Axes &cols) {
    Polynomial g;
    for (const auto &[p, c] : f) {
        Powers q;
        for (auto k : cols)
            q.push_back(p.at(k));
        g.emplace(q, c);
    }
    return g;
}
inline Polynomial lift(const Polynomial &f, const Axes &cols, size_t n) {
    Polynomial g;
    for (const auto &[p, c] : f) {
        Powers q(n);
        for (size_t i = 0; i < p.size(); ++i)
            q[cols.at(i)] = p[i];
        g.emplace(q, c);
    }
    return g;
}
inline Basis project(const Basis &f, const Axes &cols) {
    Basis g;
    for (const auto &x : f)
        g.push_back(project(x, cols));
    return g;
}
inline Basis lift(const Basis &f, const Axes &cols, size_t n) {
    Basis g;
    for (const auto &x : f)
        g.push_back(lift(x, cols, n));
    return g;
}
} // namespace atlas::chart_rows
