// Sparse local algebra operations shared by the dispatcher and the native CLI.
#include "local_io.hpp"
#include "ae_codimension.hpp"
#include <json/writer.h>
#include "local_monomial.hpp"
#include "local_point_quotient.hpp"
#include "local_dimension.hpp"
#include "local_division.hpp"
#include "local_point_basis.hpp"
#include "local_point_normal_form.hpp"
#include "groebner.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <array>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace atlas::local {
using TraceScreenHook = int (*)(uint32_t, uint32_t, uint32_t, const uint32_t*, const uint32_t*,
                                const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" void atlas_groebner_swap_cpu_scope(ReplayBatch*, int*, TraceScreenHook*, uint64_t*) noexcept;

namespace {
std::string basis_rows(const Basis& basis) {
    std::string out;
    for (const auto& f : basis) out += encode_row(f) + "\n";
    return out;
}

std::string quotient_operation(const std::string& wire) {
    if (wire == "local_quotient_basis") return "local-quotient-basis";
    if (wire == "local_jacobian_algebra") return "local-jacobian-algebra";
    if (wire == "local_milnor_number") return "milnor-number";
    if (wire == "local_tjurina_number") return "tjurina-number";
    return {};
}

std::string quotient_rows(const PointQuotient& result) {
    std::ostringstream out;
    out << (result.dimension ? result.dimension->get_str() : "infinite") << '\n'
        << result.derivatives.size() << '\n' << basis_rows(result.derivatives);
    for (const auto& monomial : result.monomials) {
        for (size_t i = 0; i < monomial.size(); ++i) {
            if (i) out << ' ';
            out << monomial[i];
        }
        out << '\n';
    }
    return out.str();
}

std::string line(std::istream& in) {
    std::string value;
    if (!std::getline(in, value)) throw std::invalid_argument("Incomplete local operation request.");
    return value;
}

std::vector<std::string> read_names(std::istream& in) {
    std::istringstream fields(line(in));
    std::vector<std::string> names;
    for (std::string name; fields >> name;) names.push_back(std::move(name));
    return names;
}

std::vector<mpq_class> read_point(std::istream& in, size_t width) {
    std::istringstream fields(line(in));
    std::vector<mpq_class> result(width);
    for (auto& coordinate : result) {
        if (!(fields >> coordinate) || coordinate.get_den() == 0)
            throw std::invalid_argument("Invalid local point coordinate.");
        coordinate.canonicalize();
    }
    std::string extra;
    if (fields >> extra) throw std::invalid_argument("Incorrect local point dimension.");
    return result;
}

struct LocalInput {
    std::vector<std::string> names;
    std::vector<mpq_class> point;
    Basis generators;
    bool needs_scope = false;
};

class CpuScope {
    ReplayBatch batch_ = nullptr;
    TraceScreenHook trace_ = nullptr;
    int active_ = 1;
    std::array<uint64_t, 8> start_{};
    uint64_t* activity_;
public:
    explicit CpuScope(uint64_t* activity) noexcept : activity_(activity) {
        atlas_groebner_swap_cpu_scope(&batch_, &active_, &trace_, start_.data());
    }
    CpuScope(const CpuScope&) = delete;
    CpuScope& operator=(const CpuScope&) = delete;
    ~CpuScope() {
        std::array<uint64_t, 8> end;
        atlas_groebner_swap_cpu_scope(&batch_, &active_, &trace_, end.data());
        if (activity_)
            for (size_t i = 0; i < end.size(); ++i) activity_[i] = end[i] - start_[i];
    }
};
}

// The packed device order supports total degrees through 32767.
extern "C" int atlas_monomials_fit_device(const int32_t* rows, size_t count, size_t width) noexcept {
    for (size_t i = 0; i < count; ++i) {
        int remaining = 32767;
        for (size_t j = 0; j < width; ++j) {
            const auto power = rows[i * width + j];
            if (power < 0 || power > remaining) return 0;
            remaining -= power;
        }
    }
    return 1;
}

// Validate untrusted device witnesses before applying them to exact QQ terms.
extern "C" int atlas_monomial_witnesses_valid(const int32_t* terms, size_t count,
                                              const int32_t* heads, size_t divisors, size_t width,
                                              const int32_t* witnesses) noexcept {
    for (size_t i = 0; i < count; ++i) {
        const auto witness = witnesses[i];
        const auto term = terms + i * width;
        const auto divides = [&](size_t j) {
            const auto head = heads + j * width;
            return std::equal(head, head + width, term, std::less_equal<>());
        };
        if (witness == -1) {
            for (size_t j = 0; j < divisors; ++j)
                if (divides(j)) return 0;
        } else if (witness < 0 || static_cast<size_t>(witness) >= divisors || !divides(witness)) {
            return 0;
        }
    }
    return 1;
}

// The handle owns only parsed input. Solves may run concurrently on it with
// different orders, and each solve performs its own incidence and rank checks.
extern "C" void* atlas_local_input_prepare(const char* data, size_t size) noexcept {
    try {
        if (!data) throw std::invalid_argument("Missing local input.");
        std::istringstream in(std::string(data, size));
        auto input = std::make_unique<LocalInput>();
        input->names = read_names(in);
        input->point = read_point(in, input->names.size());
        for (std::string row; std::getline(in, row);)
            input->generators.push_back(decode_row(row, input->names.size()));
        // Monomial ideals and principal equations return before the modular
        // solver. This records only an input shape; every solve recomputes
        // incidence, rank and dimension at the supplied point.
        input->needs_scope = input->generators.size() > 1 &&
            std::any_of(input->generators.begin(), input->generators.end(), [](const auto& f) { return f.size() > 1; });
        return input.release();
    } catch (const std::exception& ex) {
        set_error(ex.what());
    } catch (...) {
        set_error("Could not prepare local input.");
    }
    return nullptr;
}

extern "C" void atlas_local_input_free(void* ptr) noexcept {
    delete static_cast<LocalInput*>(ptr);
}

extern "C" int atlas_local_input_needs_scope(const void* ptr) noexcept {
    return ptr && static_cast<const LocalInput*>(ptr)->needs_scope;
}

// For the empty ideal at a validated point only the number of coordinates
// crosses the ABI. There are no rows, point powers or modular stages to prepare.
extern "C" void atlas_local_ambient_invariants(int width, int* out) noexcept {
    const auto result = ambient_space_at_point(width);
    out[0] = result.dimension;
    out[1] = result.embedding_dimension;
    out[2] = result.jacobian_rank;
    out[3] = result.smooth;
}

extern "C" int atlas_local_cpu_invariants_scoped(const void* ptr, int lex, int smoothness, int* out,
                                                 uint64_t* activity) noexcept {
    try {
        if (!ptr || !out) throw std::invalid_argument("Missing local input or output.");
        const auto& input = *static_cast<const LocalInput*>(ptr);
        std::optional<CpuScope> cpu;
        if (input.needs_scope) cpu.emplace(activity);
        else if (activity) std::fill_n(activity, 8, uint64_t(0));
        const auto result = dimension_at_point(input.generators, input.names, input.point, lex != 0, smoothness != 0);
        out[0] = result.dimension;
        out[1] = result.embedding_dimension;
        out[2] = result.jacobian_rank;
        out[3] = result.smooth;
        return 0;
    } catch (const PointOutsideScheme&) {
        return 1;
    } catch (const std::exception& ex) {
        set_error(ex.what());
    } catch (...) {
        set_error("Could not compute local dimension.");
    }
    return -1;
}

extern "C" int atlas_local_cpu_invariants(const void* ptr, int lex, int smoothness, int* out) noexcept {
    return atlas_local_cpu_invariants_scoped(ptr, lex, smoothness, out, nullptr);
}

std::string operations(std::istream& in, const std::string& op) {
    const auto names = read_names(in);
    const auto mode = line(in);
    const bool lex = mode == "lex";
    const auto width = names.size();
    const bool gpu = op.size() > 4 && op.compare(op.size()-4,4,"_gpu") == 0;
    const auto quotient = quotient_operation(gpu ? op.substr(0,op.size()-4) : op);
    const bool dimension = op == "local_dimension" || op == "local_dimension_gpu";
    const bool smoothness = op == "local_smoothness" || op == "local_smoothness_gpu";
    const bool homogeneous = op == "local_homogeneous_normal_form" || op == "local_homogeneous_normal_form_gpu";
    const bool point_basis = op == "local_point_basis" || op == "local_point_basis_gpu";
    const bool point_nf = op == "local_point_normal_form" || op == "local_point_normal_form_gpu";
    const auto point = (op == "local_project" || op == "local_translate" || op == "local_query_jet" || homogeneous || dimension || smoothness ||
                        point_basis || point_nf || !quotient.empty()) ?
        read_point(in, width) : std::vector<mpq_class>();
    int rank = -1;
    if (smoothness || dimension) rank = std::stoi(line(in));
    const bool known = point_nf && line(in) == "1";
    Basis rows;
    if (op == "local_ae_expressions") {
        for (std::string source; std::getline(in, source);) {
            try {
                rows.push_back(parse_expression(names, source));
            } catch (const std::runtime_error& error) {
                throw std::invalid_argument("Invalid map component: " + std::string(error.what()));
            }
        }
    } else {
        for (std::string row; std::getline(in, row);) rows.push_back(decode_row(row, width));
    }
    if (op == "local_ae_expressions" || op == "local_ae_codimension") {
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, ae_codimension(rows, names)) + "\n";
    }
    if (!quotient.empty())
        return quotient_rows(quotient_at_point(quotient,rows,names,point,lex,gpu));
    if (op == "local_translate") return basis_rows(translate_unbounded(rows, point));
    if (point_basis) {
        if (!use_sparse_point_basis(rows, point)) return "";
        return basis_rows(standard_basis_at_point(rows, names, point, lex, op == "local_point_basis_gpu"));
    }
    if (point_nf) {
        if (rows.empty()) throw std::invalid_argument("A query polynomial is required.");
        auto query = std::move(rows.back());
        rows.pop_back();
        if (!use_sparse_point_basis(rows, point) && !use_sparse_point_basis({query}, point)) return "";
        auto answer = normal_form_at_point(rows, query, names, point, lex, op == "local_point_normal_form_gpu", known);
        return std::string(answer.weak ? "weak\n" : "standard\n") +
               encode_row(answer.unit) + "\n" + encode_row(answer.remainder) + "\n";
    }
    if (smoothness || dimension) {
        const bool gpu = op == "local_smoothness_gpu" || op == "local_dimension_gpu";
        const auto result = dimension_at_point(rows, names, point, lex, smoothness, gpu, rank);
        return std::to_string(result.dimension) + " " + std::to_string(result.embedding_dimension) + " " +
               std::to_string(result.jacobian_rank) + " " + (result.smooth ? "1\n" : "0\n");
    }
    if (op == "local_project") return basis_rows(project_monomials(rows, point));
    if (op == "local_basis" || op == "local_basis_gpu")
        return basis_rows(standard_basis(rows, names, lex, op == "local_basis_gpu"));
    if (homogeneous) {
        if (rows.empty()) throw std::invalid_argument("A query polynomial is required.");
        auto query = std::move(rows.back());
        rows.pop_back();
        std::ostringstream reply;
        using Clock = std::chrono::steady_clock;
        auto previous = Clock::now();
        auto stage = [&](const char* operation, const char*) {
            const auto now = Clock::now();
            reply << operation << ' ' << std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous).count() << '\n';
            previous = now;
        };
        std::optional<Polynomial> remainder;
        try {
            remainder = homogeneous_normal_form(rows, query, point, lex, op == "local_homogeneous_normal_form_gpu", stage);
        } catch (const std::overflow_error&) {
            // Large coordinate translations use the worker's general pullback.
            return "";
        }
        if (!remainder) return "";
        reply << encode_row(*remainder) << '\n';
        return reply.str();
    }
    if (op == "local_query_jet" || op == "local_normal_form") {
        if (rows.empty()) throw std::invalid_argument("A query polynomial is required.");
        auto query = std::move(rows.back());
        rows.pop_back();
        if (op == "local_query_jet") {
            auto jet = finite_query_jet(rows, query, point, lex);
            return jet ? encode_row(*jet) + "\n" : "";
        }
        auto answer = normal_form(rows, query, width, lex);
        return std::string(answer.weak ? "weak\n" : "standard\n") +
               encode_row(answer.unit) + "\n" + encode_row(answer.remainder) + "\n";
    }
    if (op == "local_partials") {
        if (rows.size() != 1) throw std::invalid_argument("Expected one polynomial for derivatives.");
        return basis_rows(partial_derivatives(rows.front(),width));
    }
    if (op == "local_heads" || op == "local_global_heads" || op == "local_associate_heads") {
        Basis result;
        for (const auto& f : rows) {
            if (f.empty()) continue;
            auto head = op == "local_global_heads" ? global_head(f, lex) : local_head(f, lex);
            if (op == "local_associate_heads")
                for (const auto& term : f)
                    if (!monomial_divides(head, term.first))
                        return "";
            result.push_back({{std::move(head), 1}});
        }
        return basis_rows(result);
    }
    throw std::invalid_argument("Unknown local operation.");
}
}
