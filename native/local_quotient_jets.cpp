#include "local_quotient_jets.hpp"
#include <algorithm>

namespace atlas::local {
namespace {
constexpr size_t max_columns = 96;
constexpr size_t max_cells = 12000;

struct Column {
    Powers powers;
    unsigned degree;
};

// Generate exactly the monomials of this degree; ordering on ties must agree
// with the leading term of the requested local order.
bool append_degree(std::vector<Column>& columns, size_t width, unsigned degree, bool lex) {
    const auto first = columns.size();
    Powers e(width);
    auto visit = [&](auto&& self, size_t axis, unsigned left) -> bool {
        if (axis+1 == width) {
            if (columns.size() == max_columns) return false;
            e[axis] = left;
            columns.push_back({e,degree});
            return true;
        }
        for (unsigned i=0; i<=left; ++i) {
            e[axis] = i;
            if (!self(self,axis+1,left-i)) return false;
        }
        return true;
    };
    if (!visit(visit,0,degree)) return false;
    std::sort(columns.begin()+first,columns.end(),[lex](const Column& a, const Column& b) {
        return global_less(b.powers,a.powers,lex);
    });
    return true;
}
}

std::optional<std::vector<Powers>> finite_point_jet_heads(const Basis& ideal, const PointRow& point, bool lex) {
    const auto width = point.size();
    if (!width || width >= max_columns || ideal.size() > max_columns) return std::nullopt;
    // This probe materializes only small truncated vector spaces. Larger
    // spaces continue through the full point-local standard basis engine.
    std::vector<Column> columns{{Powers(width),0}};
    std::map<Powers,size_t> indices{{columns.front().powers,0}};
    for (unsigned d=1;; ++d) {
        const auto border = columns.size();
        if (!append_degree(columns,width,d,lex)) return std::nullopt;
        for (size_t j=border; j<columns.size(); ++j) indices.emplace(columns[j].powers,j);

        const auto jets = symbolic_point_jets(ideal,point,d+1);
        std::vector<std::vector<Polynomial>> rows;
        for (const auto& jet : jets) {
            if (jet.empty()) continue;
            unsigned order = d;
            for (const auto& [e,c] : jet) order = std::min(order,columns[indices.at(e)].degree);
            for (const auto& multiplier : columns) {
                if (multiplier.degree + order > d) break;
                if (rows.size() >= max_cells/columns.size()) return std::nullopt;
                auto& row = rows.emplace_back(columns.size());
                for (const auto& [e,c] : jet) {
                    if (multiplier.degree + columns[indices.at(e)].degree > d) continue;
                    Powers target = e;
                    for (size_t i=0; i<width; ++i) target[i] += multiplier.powers[i];
                    row[indices.at(target)] = c;
                }
            }
        }
        // In R/m^(d+1) the displayed multiples span the ideal. A pivot
        // at a degree-d column is a relation with that leading monomial.
        // If every degree-d column pivots, m^d ⊂ I+m^(d+1), hence m^d ⊂ I
        // in the local ring by Nakayama. Every shorter pivot then lifts to
        // the same initial monomial, and all remaining columns are standard.
        if (rows.size() < columns.size()-border) continue;
        const auto pivots = symbolic_pivot_columns(point,std::move(rows));
        std::vector<bool> leading(columns.size());
        for (auto j : pivots) leading[j] = true;
        if (std::find(leading.begin()+border,leading.end(),false) != leading.end()) continue;
        std::vector<Powers> minimal;
        for (auto column : pivots) {
            auto e = columns[column].powers;
            bool multiple = false;
            for (size_t i=0; i<width && !multiple; ++i) if (e[i] != 0) {
                --e[i];
                multiple = leading[indices.at(e)];
                ++e[i];
            }
            if (!multiple) minimal.push_back(std::move(e));
        }
        return minimal;
    }
}
}
