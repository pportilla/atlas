// Translate marked trigonometric literals into the QQbar evaluator's program.
#include <gmpxx.h>
#include "groebner.hpp"
#include <map>
#include <sstream>

namespace atlas {
namespace {

mpq_class rational(std::istream& input) {
    std::string token;
    if (!(input >> token)) throw std::invalid_argument("Expected a rational coefficient.");
    mpq_class q(token, 10);
    if (q.get_den() == 0) throw std::domain_error("Invalid rational coefficient.");
    q.canonicalize();
    return q;
}

struct Angle {
    mpq_class q = 1;
    mpz_class pi = 0;
    int imaginary = 0;  // Exponent of I modulo four.
};

std::string trig_program(const std::string& function, const Angle& angle) {
    const bool exponential = function == "exp";
    if (angle.q == 0) {
        if (function == "cot" || function == "csc") throw std::domain_error("Undefined trigonometric scalar.");
        return exponential || function == "sec" || function == "cos" ? "q 1" : "q 0";
    }
    if (angle.pi != 1 || (angle.imaginary % 2 != static_cast<int>(exponential)))
        throw std::invalid_argument("Expected a rational multiple of pi in the algebraic scalar.");
    const mpq_class q = angle.imaginary >= 2 ? -angle.q : angle.q;
    const mpq_class negative = -q;
    const std::string z = "q -1 pow " + q.get_str();
    if (exponential) return z;
    const std::string inverse = "q -1 pow " + negative.get_str();
    const auto cosine = z + " " + inverse + " + q 1/2 *";
    const auto sine = z + " " + inverse + " q -1 * + i * q -1/2 *";
    if (function == "cos") return cosine;
    if (function == "sin") return sine;
    if (function == "tan") return sine + " " + cosine + " pow -1 *";
    if (function == "cot") return cosine + " " + sine + " pow -1 *";
    if (function == "sec") return cosine + " pow -1";
    if (function == "csc") return sine + " pow -1";
    throw std::invalid_argument("Unknown trigonometric scalar.");
}

Angle angle_expression(std::istream& in) {
    std::vector<Angle> values;
    for (std::string op; in >> op;) {
        if (op == "pi" || op == "I" || op == "q") {
            Angle value;
            if (op == "q") value.q = rational(in);
            else if (op == "pi") value.pi = 1;
            else value.imaginary = 1;
            values.push_back(std::move(value));
        } else if (op == "neg" && !values.empty()) {
            values.back().q = -values.back().q;
        } else if (values.size() >= 2) {
            Angle b = std::move(values.back());
            values.pop_back();
            auto& a = values.back();
            if (op == "*") {
                a.q *= b.q;
                a.pi += b.pi;
                a.imaginary = (a.imaginary + b.imaginary) % 4;
            } else if (op == "/") {
                if (!b.q) throw std::domain_error("Invalid marked algebraic coefficient.");
                a.q /= b.q;
                a.pi -= b.pi;
                a.imaginary = (a.imaginary - b.imaginary + 4) % 4;
            } else if ((op == "+" || op == "-") && a.pi == b.pi && a.imaginary == b.imaginary) {
                if (op == "+") a.q += b.q; else a.q -= b.q;
            } else throw std::invalid_argument("Expected a rational algebraic exponent or angle.");
        } else throw std::invalid_argument("Incomplete marked angle.");
    }
    if (values.size() != 1) throw std::invalid_argument("Incomplete marked angle.");
    return std::move(values.front());
}

std::string indexed_program(std::istream& in, bool flint) {
    mpq_class position = rational(in);
    if (position.get_den() != 1) throw std::invalid_argument("Index outside the root polynomial.");
    mpz_class index = position.get_num();
    std::map<size_t, mpq_class> polynomial;
    if (flint) {
        size_t power = 0;
        while (in >> std::ws && !in.eof()) polynomial.emplace(power++, rational(in));
        --index;  // Stored FLINT root indices are one-based.
    } else {
        for (size_t power; in >> power;) polynomial[power] = rational(in);
    }
    if (polynomial.empty() || !polynomial.rbegin()->second)
        throw std::invalid_argument("Invalid index or polynomial for the marked root.");
    const size_t degree = polynomial.rbegin()->first;
    if (index >= degree || (flint ? index < 0 : index < -mpz_class(degree)))
        throw std::invalid_argument("Index outside the root polynomial.");
    std::ostringstream out;
    out << (flint ? "flintroot " : "indexedroot ") << degree;
    for (size_t j = 0; j <= degree; ++j) {
        const auto term = polynomial.find(j);
        out << ' ' << (term == polynomial.end() ? "0" : term->second.get_str());
    }
    out << ' ' << index;
    return out.str();
}

std::string marked_scalar(std::istream& in) {
    std::string op, name;
    in >> op;
    if (op == "flintroot" || op == "indexedroot")
        return indexed_program(in, op == "flintroot");
    if (op == "trig" && in >> name) return trig_program(name, angle_expression(in));
    if (op == "same") {
        size_t length;
        if (!(in >> length) || !length) throw std::invalid_argument("Missing marked polynomial.");
        const auto lead = rational(in), other = rational(in);
        if (!lead || !other) return "0";
        for (size_t i = 1; i < length; ++i) {
            const auto a = rational(in), b = rational(in);
            if (a * other != b * lead) return "0";
        }
        return "1";
    }
    throw std::invalid_argument("Unknown marked scalar operation.");
}

} // namespace
} // namespace atlas

extern "C" char* atlas_marked_scalar(const char* input, size_t size) noexcept {
    try {
        std::istringstream in(std::string(input, size));
        return atlas::copy_text(atlas::marked_scalar(in));
    } catch (const std::exception& error) { atlas::set_error(error.what()); }
    catch (...) { atlas::set_error("Invalid marked algebraic coefficient."); }
    return nullptr;
}
