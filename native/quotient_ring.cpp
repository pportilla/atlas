// Arithmetic in QQ[x]/I, using the reduced global basis in the declared order.
#include "quotient_ring.hpp"
#include "local_monomial.hpp"
#include "quotient_inverse.hpp"
#include "quotient_finite_inverse.hpp"
#include "quotient_packed.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>
#include <sstream>

namespace atlas::quotient_ring {
namespace {
bool constant(const Powers& e) {
    return std::all_of(e.begin(),e.end(),[](const auto& k) { return k == 0; });
}

Polynomial one(size_t width) { return {{Powers(width),1}}; }

void add(Polynomial& f, Powers powers, mpq_class coefficient) {
    if (coefficient == 0) return;
    auto [it,inserted] = f.try_emplace(std::move(powers),std::move(coefficient));
    if (!inserted) {
        it->second += coefficient;
        if (it->second == 0) f.erase(it);
    }
}

Polynomial combine(Polynomial a, const Polynomial& b, bool subtract) {
    for (const auto& [e,c] : b) add(a,e,subtract ? -c : c);
    return a;
}

void validate(const Basis& rows, size_t width) {
    for (const auto& f : rows) for (const auto& [e,c] : f) {
        if (e.size() != width || std::any_of(e.begin(),e.end(),[](const auto& k) { return k < 0; }))
            throw InvalidInput("Invalid exponent vector in quotient-ring input.");
        if (c == 0) throw InvalidInput("A sparse term must have nonzero coefficient.");
    }
}

Basis defining_basis(const Basis& rows, size_t width, bool lex) {
    validate(rows,width);
    Basis result;
    std::vector<Powers> heads;
    for (const auto& f : rows) {
        if (f.empty()) continue;
        auto head = local::global_head(f,lex);
        if (constant(head)) return {one(width)};
        result.push_back(f);
        const auto& lc = f.at(head);
        if (lc != 1) for (auto& term : result.back()) term.second /= lc;
        heads.push_back(std::move(head));
    }
    if (result.size() <= 1) return result;

    if (std::all_of(result.begin(),result.end(),[](const auto& f) { return f.size() == 1; })) {
        Basis minimal;
        for (size_t i = 0; i < heads.size(); ++i) {
            bool redundant = false;
            for (size_t j = 0; j < heads.size(); ++j)
                if (j != i && local::monomial_divides(heads[j],heads[i]) && (heads[j] != heads[i] || j < i)) {
                    redundant = true;
                    break;
                }
            if (!redundant) minimal.push_back(std::move(result[i]));
        }
        result = std::move(minimal);
    } else {
        // Buchberger's product criterion certifies a basis with coprime heads.
        // An interreduced such basis is already the unique reduced basis.
        bool coprime = true;
        for (size_t i = 0; i < heads.size(); ++i) for (size_t j = 0; j < i; ++j) {
            if (local::monomial_divides(heads[i],heads[j]) || local::monomial_divides(heads[j],heads[i])) return primary::reduced(rows,width,lex);
            for (size_t k = 0; k < width; ++k) if (heads[i][k] && heads[j][k]) coprime = false;
        }
        for (size_t i = 0; i < result.size(); ++i)
            for (const auto& [e,c] : result[i]) if (e != heads[i])
                for (const auto& h : heads) if (local::monomial_divides(h,e)) return primary::reduced(rows,width,lex);
        if (!coprime) {
            const auto packed = PackedReducer::make(result,heads,width,lex);
            if (!packed || !packed->pairs_reduce_to_zero()) return primary::reduced(rows,width,lex);
        }
    }
    std::sort(result.begin(),result.end(),[lex](const auto& a, const auto& b) {
        return local::global_less(local::global_head(b,lex),local::global_head(a,lex),lex);
    });
    return result;
}

mpq_class scalar_power(mpq_class a, mpz_class n) {
    if (a == 1 || n == 0) return 1;
    if (a == -1) return mpz_odd_p(n.get_mpz_t()) ? -1 : 1;
    mpq_class result = 1;
    while (n > 0) {
        if (mpz_odd_p(n.get_mpz_t())) result *= a;
        n >>= 1;
        if (n > 0) a *= a;
    }
    return result;
}
}

Ring::Ring(const Basis& relations, size_t width, bool lex)
    : Ring(defining_basis(relations,width,lex),width,lex,Reduced{}) {}

Ring::Ring(Basis basis, size_t width, bool lex, Reduced)
    : width_(width), lex_(lex), basis_(std::move(basis)), zero_(false), binomial_(true) {
    for (const auto& f : basis_) {
        heads_.push_back(local::global_head(f,lex_));
        zero_ |= constant(heads_.back());
        binomial_ &= f.size() <= 2;
    }
    if (!zero_) packed_ = PackedReducer::make(basis_,heads_,width_,lex_);
}

Ring::~Ring() = default;
Ring::Ring(Ring&&) noexcept = default;

Ring Ring::from_basis(Basis basis, size_t width, bool lex) {
    validate(basis,width);
    for (const auto& f : basis)
        if (f.empty() || f.at(local::global_head(f,lex)) != 1)
            throw InvalidInput("Expected a monic reduced basis of the declared quotient ring.");
    return Ring(std::move(basis),width,lex,Reduced{});
}

bool Ring::standard(const Powers& e) const {
    return std::none_of(heads_.begin(),heads_.end(),[&](const auto& h) { return local::monomial_divides(h,e); });
}

Polynomial Ring::direct(Polynomial f) const {
    if (zero_) return {};
    if (basis_.empty()) return f;
    Polynomial remainder;
    // Any reducible term can be replaced. Each replacement decreases the
    // declared order, and a Gröbner basis makes the final remainder unique.
    while (!f.empty()) {
        auto node = f.extract(std::prev(f.end()));
        auto& e = node.key();
        auto& c = node.mapped();
        const auto divisor = std::find_if(heads_.begin(),heads_.end(),[&](const auto& h) { return local::monomial_divides(h,e); });
        if (divisor == heads_.end()) {
            add(remainder,std::move(e),std::move(c));
            continue;
        }
        const auto& h = *divisor;
        const auto& row = basis_[divisor-heads_.begin()];
        if (row.size() == 1) continue;
        if (row.size() == 2) {
            auto tail = row.begin();
            if (tail->first == h) ++tail;
            // Apply the same rewrite in bulk, even for exponents wider than
            // a machine word. At least one coordinate drops at each step.
            mpz_class copies = 0;
            for (size_t i = 0; i < width_; ++i) if (h[i] > tail->first[i]) {
                mpz_class bound = 1 + (e[i]-h[i])/(h[i]-tail->first[i]);
                if (copies == 0 || bound < copies) copies = std::move(bound);
            }
            for (size_t i = 0; i < width_; ++i) e[i] += copies*(tail->first[i]-h[i]);
            c *= scalar_power(-tail->second,copies);
            add(f,std::move(e),std::move(c));
        } else {
            for (size_t i = 0; i < width_; ++i) e[i] -= h[i];
            for (const auto& [tail,coefficient] : row) if (tail != h) {
                Powers p(e);
                for (size_t i = 0; i < width_; ++i) p[i] += tail[i];
                add(f,std::move(p),-c*coefficient);
            }
        }
    }
    return remainder;
}

Polynomial Ring::normal(const Polynomial& f) const {
    if (zero_) return {};
    if (basis_.empty() || std::all_of(f.begin(),f.end(),[&](const auto& term) { return standard(term.first); })) return f;
    if (packed_) if (auto result = packed_->reduce(f,nullptr,false,false)) return std::move(*result);
    if (binomial_) return direct(f);
    Polynomial remaining, expanded;
    for (const auto& [e,c] : f) {
        if (standard(e) || std::none_of(e.begin(),e.end(),[](const auto& k) { return k > 128; })) {
            remaining.emplace(e,c);
            continue;
        }
        // Binary modular powering avoids a degree-by-degree rewrite of
        // e.g. x^(2^80) modulo x^2+x+1.
        Polynomial term{{Powers(width_),c}};
        for (size_t i = 0; i < width_; ++i) if (e[i] != 0) {
            Powers axis(width_);
            axis[i] = 1;
            term = product(term,power(direct({{std::move(axis),1}}),e[i]));
        }
        expanded = combine(std::move(expanded),term,false);
    }
    return combine(direct(std::move(remaining)),expanded,false);
}

Polynomial Ring::product(const Polynomial& a, const Polynomial& b) const {
    return direct(primary::multiply(a,b));
}

Polynomial Ring::power(Polynomial a, mpz_class exponent) const {
    if (zero_) return {};
    if (!exponent) return one(width_);
    if (a.empty()) return {};
    if (exponent == 1) return a;
    if (a.size() == 1 && binomial_) {
        auto e = a.begin()->first;
        for (auto& k : e) k *= exponent;
        auto monomial = direct({{std::move(e),1}});
        if (!monomial.empty()) monomial.begin()->second *= scalar_power(a.begin()->second,exponent);
        return monomial;
    }
    if (width_ == 1) if (auto f = univariate_power(a,basis_,exponent)) return std::move(*f);
    Polynomial result;
    bool initialized = false;
    while (exponent > 0) {
        if (mpz_odd_p(exponent.get_mpz_t())) {
            result = initialized ? product(result,a) : a;
            initialized = true;
        }
        exponent >>= 1;
        if (exponent > 0) a = product(a,a);
        if (a.empty()) return {};
    }
    return result;
}

std::optional<Polynomial> Ring::inverse(const Polynomial& a) const {
    if (zero_) return Polynomial{};
    if (a.empty()) return std::nullopt;
    const Powers origin(width_);
    const auto c = a.find(origin);
    if (a.size() == 1 && c != a.end()) return Polynomial{{origin,1/c->second}};
    if (basis_.empty()) return std::nullopt;
    if (c == a.end() && std::all_of(basis_.begin(),basis_.end(),[&](const auto& g) { return !g.count(origin); }))
        return std::nullopt; // evaluation at the origin maps this element to zero

    if (a.size() == 1) {
        // In a binomial relation m=d (d a nonzero rational), every factor of
        // m is invertible. Choose m^k divisible by this entire monomial.
        const auto& [e,coefficient] = *a.begin();
        for (size_t j = 0; j < basis_.size(); ++j) {
            const auto& g = basis_[j];
            if (g.size() != 2 || !g.count(origin)) continue;
            const auto& h = heads_[j];
            mpz_class k = 0;
            bool covered = true;
            for (size_t i = 0; i < width_; ++i) if (e[i]) {
                if (!h[i]) { covered = false; break; }
                mpz_class need = (e[i]+h[i]-1)/h[i];
                if (need > k) k = std::move(need);
            }
            if (covered) {
                Powers complement(width_);
                for (size_t i = 0; i < width_; ++i) complement[i] = k*h[i]-e[i];
                return normal({{std::move(complement),1/(coefficient*scalar_power(-g.at(origin),k))}});
            }
        }
    }

    std::optional<Polynomial> finite;
    if (finite_inverse(basis_,heads_,a,width_,finite)) return finite;
    if (std::all_of(basis_.begin(),basis_.end(),[](const auto& p) { return p.size() == 1; })) {
        // In a monomial quotient the nonconstant part of a unit is nilpotent.
        // Its support must lie in the radical of this monomial ideal.
        if (c == a.end()) return std::nullopt;
        for (const auto& [e,coefficient] : a) if (e != origin) {
            const bool nilpotent = std::any_of(heads_.begin(),heads_.end(),[&](const auto& h) {
                for (size_t i = 0; i < width_; ++i) if (h[i] && !e[i]) return false;
                return true;
            });
            if (!nilpotent) return std::nullopt;
        }
        Polynomial tail = a;
        tail.erase(origin);
        for (auto& term : tail) term.second /= -c->second;
        Polynomial result{{origin,1/c->second}};
        while (!tail.empty()) {
            result = combine(result,product(result,tail),false);
            tail = product(tail,tail);
        }
        return result;
    }

    if (independent_graph(basis_,width_)) return std::nullopt;

    if (auto result = fast_inverse(a,basis_,width_,lex_)) return normal(*result);
    // Invert a in the graph ideal (I, t*a-1), with t first in lex order.
    // A unit has a unique relation t-g(x) in the reduced elimination basis.
    // The product check excludes elements that invert only on a component.
    Basis graph;
    for (const auto& g : basis_) {
        Polynomial row;
        for (const auto& [e,coefficient] : g) {
            Powers lifted{0};
            lifted.insert(lifted.end(),e.begin(),e.end());
            row.emplace(std::move(lifted),coefficient);
        }
        graph.push_back(std::move(row));
    }
    Polynomial relation{{Powers(width_+1),-1}};
    Powers t(width_+1);
    t[0] = 1;
    for (const auto& [e,coefficient] : a) {
        Powers lifted{1};
        lifted.insert(lifted.end(),e.begin(),e.end());
        relation.emplace(std::move(lifted),coefficient);
    }
    graph.push_back(std::move(relation));
    for (const auto& row : primary::reduced(graph,width_+1,true)) if (row.rbegin()->first == t) {
        Polynomial candidate;
        for (const auto& [e,coefficient] : row) if (e != t)
            candidate.emplace(Powers(e.begin()+1,e.end()),-coefficient);
        candidate = normal(candidate);
        if (product(a,candidate) == one(width_)) return candidate;
        break;
    }
    return std::nullopt;
}

bool independent_graph(const Basis& rows, size_t geometric_width) {
    std::vector<size_t> appearances(geometric_width);
    for (const auto& g : rows) for (const auto& [e,c] : g)
        for (size_t j=0; j<geometric_width; ++j) if (e[j]) ++appearances[j];
    return std::all_of(rows.begin(),rows.end(),[&](const auto& g) {
        for (const auto& [e,c] : g) {
            const auto end=e.begin()+geometric_width;
            const auto it=std::find_if(e.begin(),end,[](const auto& k){return k!=0;});
            if (it!=end && *it==1 && appearances[it-e.begin()]==1 &&
                std::all_of(it+1,e.end(),[](const auto& k){return k==0;})) return true;
        }
        return false;
    });
}

Value Ring::evaluate(const std::string& action, const Basis& args, const mpz_class& exponent) const {
    const bool unary = action == "reduce" || action == "negate" || action == "power" ||
                       action == "unit" || action == "inverse";
    const bool binary = action == "add" || action == "subtract" || action == "multiply" ||
                        action == "equal" || action == "divide";
    if (!unary && !binary && action != "declare") throw InvalidInput("Unknown quotient-ring operation: "+action);
    if (args.size() != (unary ? 1u : binary ? 2u : 0u)) throw InvalidInput("Wrong number of operands for "+action+".");
    if (action == "declare") return std::monostate{};
    validate(args,width_);
    if (packed_ && binary && action != "divide") {
        auto f = packed_->reduce(args[0],&args[1],action == "subtract" || action == "equal", action == "multiply");
        if (f) {
            if (action == "equal") return f->empty();
            return std::move(*f);
        }
    }
    if (action == "add" || action == "subtract" || action == "equal") {
        auto difference = normal(combine(args[0],args[1],action != "add"));
        if (action == "equal") return difference.empty();
        return difference;
    }
    auto a = normal(args[0]);
    if (action == "reduce") return a;
    if (action == "negate") {
        for (auto& term : a) term.second = -term.second;
        return a;
    }
    if (action == "multiply") return product(a,normal(args[1]));
    if (action == "power" && exponent >= 0) return power(std::move(a),exponent);
    auto reciprocal = inverse(action == "divide" ? normal(args[1]) : a);
    if (action == "unit") return reciprocal.has_value();
    if (!reciprocal) throw InvalidInput("The element is not a unit in this quotient ring.");
    if (action == "divide") return product(a,*reciprocal);
    if (action == "power") return power(std::move(*reciprocal),-exponent);
    return std::move(*reciprocal);
}

Json::Value Ring::render(const Value& value, const std::vector<std::string>& names) const {
    Json::Value out, variables(Json::arrayValue), relations(Json::arrayValue);
    for (const auto& name : names) variables.append(name);
    for (const auto& f : basis_) relations.append(local::format_polynomial(f,names));
    out["ring"]["field"] = "QQ";
    out["ring"]["variables"] = std::move(variables);
    out["ring"]["order"] = lex_ ? "lex" : "grevlex";
    out["relations"] = std::move(relations);
    if (const auto* f = std::get_if<Polynomial>(&value)) {
        out["type"] = "quotient-element";
        out["representative"] = local::format_polynomial(*f,names);
    } else if (const auto* predicate = std::get_if<bool>(&value)) {
        out["type"] = "boolean";
        out["value"] = *predicate;
    } else out["type"] = "quotient-ring";
    return out;
}

mpz_class parse_exponent(const std::string& text) {
    const size_t first = !text.empty() && (text[0] == '-' || text[0] == '+') ? 1 : 0;
    if (text.size() <= first || !std::all_of(text.begin()+first,text.end(),[](char c) { return c >= '0' && c <= '9'; }))
        throw InvalidInput("Expected an integer quotient-ring exponent.");
    return mpz_class(text[0] == '+' ? text.substr(1) : text,10);
}

Json::Value compute(const Basis& relations, const std::vector<std::string>& names, bool lex,
                    const std::string& action, const Basis& operands, const mpz_class& exponent) {
    Ring ring(relations,names.size(),lex);
    return ring.render(ring.evaluate(action,operands,exponent),names);
}

std::string operation(std::istream& in, bool prepared) {
    Json::Value result;
    try {
        std::string coordinates,order,action,exponent,count,row;
        if (!std::getline(in,coordinates) || !std::getline(in,order) || !std::getline(in,action) ||
            !std::getline(in,exponent) || !std::getline(in,count) || (order != "lex" && order != "grevlex"))
            throw InvalidInput("Expected quotient-ring coordinates, order, operation, exponent and relation count.");
        const auto n = parse_exponent(count);
        if (n < 0 || !n.fits_ulong_p()) throw InvalidInput("Invalid quotient-ring relation count.");
        std::vector<std::string> names;
        std::istringstream vars(coordinates);
        for (std::string name; vars >> name;) names.push_back(std::move(name));
        Basis relations, operands;
        for (unsigned long i = 0; i < n.get_ui(); ++i) {
            if (!std::getline(in,row)) throw InvalidInput("Missing quotient-ring relation.");
            relations.push_back(local::decode_row(row,names.size()));
        }
        while (std::getline(in,row)) operands.push_back(local::decode_row(row,names.size()));
        const auto k = parse_exponent(exponent);
        if (prepared) {
            const auto ring = Ring::from_basis(std::move(relations),names.size(),order == "lex");
            result = ring.render(ring.evaluate(action,operands,k),names);
        } else result = compute(relations,names,order == "lex",action,operands,k);
    } catch (const std::invalid_argument& error) {
        result["error"]["code"] = "invalid_request";
        result["error"]["message"] = error.what();
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer,result)+"\n";
}
}
