// Fresh-process CPU frontend for local and global ideal operations.
#include "local_io.hpp"
#include "ae_codimension.hpp"
#include "local_division.hpp"
#include "local_dimension.hpp"
#include "local_point_basis.hpp"
#include "local_point_normal_form.hpp"
#include "local_point_quotient.hpp"
#include "primary_decomposition.hpp"
#include "normalization.hpp"
#include "localization.hpp"
#include "ideal_powers.hpp"
#include "quotient_ring.hpp"
#include <json/json.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace atlas::local;
using Strings = std::vector<std::string>;
using Clock = std::chrono::steady_clock;

struct Request {
    std::string operation, order = "grevlex", compute = "auto";
    Strings names, generators, primes;
    std::vector<mpq_class> point;
    std::optional<std::string> query, other_query, numerator, denominator, exponent;
    std::string action = "declare";
    bool json = false;
    bool point_given = false;
};

bool power_query(const std::string& op) {
    return op == "ideal-power" || op == "symbolic-power";
}

bool global_ideal_query(const std::string& op) {
    return op == "primary-decomposition" || op == "normalization" || op == "localization" || op == "localize" || power_query(op) || op == "quotient-ring";
}

bool hypersurface_query(const std::string& op) {
    return op == "milnor-number" || op == "tjurina-number" || op == "local-jacobian-algebra";
}

bool local_quotient(const std::string& op) {
    return hypersurface_query(op) || op == "local-quotient-basis";
}

bool dimension_query(const std::string& operation) {
    return operation == "local-dimension" || operation == "local-smoothness";
}

bool needs_device_dispatch() {
    // Explicit device settings and custom profiles use the shared dispatch policy.
    for (auto name : {"RESOLUTION_GPU_BACKEND", "RESOLUTION_GPU_INDEX", "ATLAS_TRACE_PROFILE"})
        if (std::getenv(name)) return true;
    const char* cache = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    if ((!cache || !*cache) && !home) return true;
    const std::string directory = cache && *cache ? cache : std::string(home) + "/.cache";
    for (auto backend : {"triton", "opencl"}) {
        std::ifstream file(directory + "/resolution-atlas/groebner-trace-" + backend + "-0.json",
                           std::ios::binary | std::ios::ate);
        if (!file || file.tellg() > (4 << 20)) continue;
        file.seekg(0);
        Json::Value profile;
        Json::CharReaderBuilder reader;
        if (!Json::parseFromStream(reader, file, &profile, nullptr) || !profile["rows"].isArray()) continue;
        // A positive separation is required by trace_policy._savings. The
        // shared dispatcher makes the full eligibility and device decisions.
        for (const auto& row : profile["rows"])
            if (row["cpu_low_ms"].isNumeric() && row["gpu_high_ms"].isNumeric() &&
                row["cpu_low_ms"].asDouble() > row["gpu_high_ms"].asDouble()) return true;
    }
    return false;
}

std::string join(const Strings& values, const std::string& separator) {
    std::string result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) result += separator;
        result += values[i];
    }
    return result;
}

Strings comma_list(const std::string& arg) {
    const char* space = " \t\n\r\v\f";
    if (arg.find_first_not_of(space) == arg.npos) return {};
    Strings out;
    for (size_t i = 0;;) {
        const auto end = arg.find(',', i);
        auto item = arg.substr(i, end-i);
        const auto begin = item.find_first_not_of(space);
        out.push_back(begin == item.npos ? "" : item.substr(begin, item.find_last_not_of(space)-begin+1));
        if (end == arg.npos) return out;
        i = end+1;
    }
}

mpz_class integer(std::string s) {
    bool negative = !s.empty() && s.front() == '-';
    if (negative || (!s.empty() && s.front() == '+')) s.erase(0, 1);
    if (s.empty() || s.find_first_not_of("0123456789") != s.npos)
        throw std::invalid_argument("Invalid point coordinate.");
    mpz_class n(s, 10);
    return negative ? -n : n;
}

mpq_class coordinate(const std::string& s) {
    const auto slash = s.find('/');
    auto numerator = integer(s.substr(0, slash));
    auto denominator = slash == s.npos ? mpz_class(1) : integer(s.substr(slash+1));
    if (!denominator) throw std::invalid_argument("Zero denominator.");
    mpq_class q(numerator, denominator);
    q.canonicalize();
    return q;
}

bool is_option(const std::string& s) {
    static const std::set<std::string> flags = {
        "--field", "--vars", "--order", "--compute", "--gpu", "--gen", "--poly",
        "--component", "--prime", "--numerator", "--denominator", "--exponent", "--action", "--other-poly", "--point", "--json", "--help", "-h", "--"
    };
    return flags.count(s.substr(0, s.find('=')));
}

Request arguments(int argc, char** argv) {
    Request r;
    r.operation = argv[3];
    if (r.operation == "localize") r.operation = "localization";
    std::optional<Strings> point;
    bool vars = false;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") { r.json = true; continue; }
        const auto eq = arg.find('=');
        auto flag = arg.substr(0, eq);
        if (flag == "--component" && r.operation == "ae-codimension") flag = "--gen";
        if (flag != "--gen" && flag != "--poly" && flag != "--vars" && flag != "--point" &&
            flag != "--field" && flag != "--order" && flag != "--compute" &&
            !(power_query(r.operation) && flag == "--exponent") &&
            !(r.operation == "quotient-ring" && (flag == "--action" || flag == "--other-poly" || flag == "--exponent")) &&
            !(r.operation == "localization" && (flag == "--prime" || flag == "--numerator" || flag == "--denominator")))
            throw std::invalid_argument("Use the general command parser.");
        if (eq == arg.npos && (++i == argc || is_option(argv[i])))
            throw std::invalid_argument("Missing argument.");
        auto value = eq == arg.npos ? std::string(argv[i]) : arg.substr(eq+1);
        // Unicode whitespace and argparse abbreviations are accepted by the general frontend.
        if (std::any_of(value.begin(), value.end(), [](unsigned char c){ return c >= 128; }))
            throw std::invalid_argument("Use the Unicode command parser.");
        if (flag == "--gen") r.generators.push_back(value);
        else if (flag == "--exponent") r.exponent = value;
        else if (flag == "--prime") r.primes.push_back(value);
        else if (flag == "--numerator") r.numerator = value;
        else if (flag == "--denominator") r.denominator = value;
        else if (flag == "--poly") r.query = value;
        else if (flag == "--other-poly") r.other_query = value;
        else if (flag == "--action") r.action = value;
        else if (flag == "--vars") { r.names = comma_list(value); vars = true; }
        else if (flag == "--point") point = comma_list(value);
        else if (flag == "--order") r.order = value;
        else if (flag == "--compute") r.compute = value;
        else if (value != "QQ") throw std::invalid_argument("Unsupported field.");
    }
    if (!vars || (r.names.empty() && !power_query(r.operation) && r.operation != "quotient-ring" && r.operation != "local-quotient-basis") ||
        (power_query(r.operation) && !r.exponent) || (r.order != "lex" && r.order != "grevlex") ||
        (r.compute != "auto" && r.compute != "cpu") ||
        (r.operation != "quotient-ring" && (r.operation == "local-normal-form" || hypersurface_query(r.operation)) != r.query.has_value()) ||
        (r.operation == "quotient-ring" && (bool(r.exponent) != (r.action == "power") || (r.other_query && !r.query))) ||
        (hypersurface_query(r.operation) && !r.generators.empty()))
        throw std::invalid_argument("Use the general algebra handler.");
    std::set<std::string> distinct;
    for (const auto& name : r.names)
        if (name.empty() || name.size()>24 || !std::isalpha(name.front()) ||
            !std::all_of(name.begin(), name.end(), [](char c){ return std::isalnum(c) || c=='_'; }) ||
            !distinct.insert(name).second) throw std::invalid_argument("Invalid variables.");
    if (global_ideal_query(r.operation) || r.operation == "ae-codimension") {
        if (point) throw std::invalid_argument("Unexpected point.");
        return r;
    }
    r.point_given = point.has_value();
    if (!point) point = Strings(r.names.size(), "0");
    if (point->size() != r.names.size()) throw std::invalid_argument("Invalid point.");
    for (const auto& s : *point) r.point.push_back(coordinate(s));
    return r;
}

Json::Value array(const Strings& values) {
    Json::Value a(Json::arrayValue);
    for (const auto& s : values) a.append(s);
    return a;
}

std::string json_text(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    return Json::writeString(builder, value);
}

std::string shell_quote(const std::string& s) {
    if (!s.empty() && s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-") == s.npos)
        return s;
    std::string out = "'";
    for (char c : s) out += c == '\'' ? "'\"'\"'" : std::string(1,c);
    return out + "'";
}

Json::Value replay(const Request& r, const Json::Value& ring, const Strings& point) {
    Strings args{"./atlas", "algebra", r.operation, "--vars", join(r.names, ","), "--field", "QQ",
                 "--order", r.order, "--compute", r.compute};
    auto operand = [&](const std::string& flag, const std::string& s) {
        if (!s.empty() && s.front()=='-') args.push_back(flag+"="+s);
        else { args.push_back(flag); args.push_back(s); }
    };
    for (const auto& g : r.generators) operand(r.operation == "ae-codimension" ? "--component" : "--gen",g);
    for (const auto& g : r.primes) operand("--prime",g);
    if (r.numerator) operand("--numerator",*r.numerator);
    if (r.denominator) operand("--denominator",*r.denominator);
    if (r.query) operand("--poly",*r.query);
    if (r.other_query) operand("--other-poly", *r.other_query);
    if (r.operation == "quotient-ring") operand("--action", r.action);
    if (r.exponent) operand("--exponent",*r.exponent);
    if (r.point_given || !point.empty()) args.insert(args.end(), {"--point", join(point, ",")});
    args.push_back("--json");
    for (auto& arg : args) arg = shell_quote(arg);
    Json::Value request;
    request["operation"] = r.operation;
    request["ring"] = ring;
    request["compute"] = r.compute;
    if (r.operation == "ae-codimension") request["operands"]["components"] = array(r.generators);
    else if (!hypersurface_query(r.operation)) request["operands"]["ideal"] = array(r.generators);
    if (r.operation == "localization") request["operands"]["prime"] = array(r.primes);
    if (r.numerator) request["operands"]["numerator"] = *r.numerator;
    if (r.denominator) request["operands"]["denominator"] = *r.denominator;
    if (r.point_given || !point.empty()) request["operands"]["point"] = array(point);
    if (r.exponent) request["operands"]["exponent"] = *r.exponent;
    if (r.query) request["operands"]["polynomial"] = *r.query;
    if (r.other_query) request["operands"]["other_polynomial"] = *r.other_query;
    if (r.operation == "quotient-ring") request["operands"]["action"] = r.action;
    Json::Value out;
    out["command"] = join(args, " ");
    // algebra_request accepts serialized JSON. Quoting it as a string again
    // gives an Atlas string literal, even for names reserved by the session.
    out["script"] = "result = algebra_request("+json_text(json_text(request))+")\nprint(result.json)\n";
    return out;
}

struct Stages {
    Clock::time_point start = Clock::now(), checkpoint = start;
    Json::Value values{Json::arrayValue};

    void operator()(const char* name, const char* label = nullptr) {
        const auto now = Clock::now();
        Json::Value s;
        s["operation"] = name;
        s["backend"] = "cpu";
        s["elapsed_ns"] = Json::Int64(std::chrono::duration_cast<std::chrono::nanoseconds>(now-checkpoint).count());
        if (label) s["label"] = label;
        values.append(s);
        checkpoint = now;
    }
};

Json::Value local_result(const Request& r, const Basis& input, Stages& stage) {
    const bool lex = r.order == "lex";
    Polynomial query;
    if (r.query) query = parse_expression(r.names, *r.query);
    stage("parse_local_input");
    const Powers zero(r.names.size(),0);
    const Polynomial one{{zero,1}};
    Basis output;
    bool weak = false;
    const bool shifted = std::any_of(r.point.begin(), r.point.end(), [](const auto& v){ return v != 0; });
    if (r.query && (query.empty() || input.empty())) {
        output = {one, query};
    } else if (shifted && !input.empty() &&
               std::all_of(input.begin(), input.end(), [](const auto& f){ return f.size() == 1; })) {
        auto basis = project_monomials(input, r.point);
        stage("project_point_monomials");
        if (r.query) {
            auto nf = normal_form(basis, query, r.names.size(), lex);
            output = {std::move(nf.unit), std::move(nf.remainder)};
            stage("local_normal_form");
        } else output = std::move(basis);
    } else if (shifted && (use_sparse_point_basis(input, r.point) ||
                           (r.query && use_sparse_point_basis({query}, r.point)))) {
        if (r.query) {
            auto nf = normal_form_at_point(input, query, r.names, r.point, lex, false, false);
            weak = nf.weak;
            output = {std::move(nf.unit), std::move(nf.remainder)};
            stage("local_point_normal_form", "local_normal_form");
        } else {
            output = standard_basis_at_point(input, r.names, r.point, lex);
            stage("local_point_basis", "local_standard_basis");
        }
    } else if (shifted && has_point_unit(input, r.point)) {
        output = r.query ? Basis{one, {}} : Basis{one};
        stage("inspect_point_ideal");
    } else {
        Basis centered = translate(input, r.point);
        stage("center_at_point");
        auto remainder = r.query ? homogeneous_normal_form(centered, query, r.point, lex, false, std::ref(stage))
                                 : std::nullopt;
        if (remainder) output = {one, std::move(*remainder)};
        else {
            bool used_finite_jet;
            Basis basis = standard_basis(centered, r.names, lex, false, used_finite_jet);
            stage(used_finite_jet ? "finite_jet_standard_basis" : "local_standard_basis", "local_standard_basis");
            if (!r.query) output = std::move(basis);
            else if (basis == Basis{one}) output = {one, {}};
            else {
                auto jet = finite_query_jet(basis, query, r.point, lex);
                Polynomial f = jet ? std::move(*jet) : translate({query}, r.point).front();
                stage("center_query_at_point");
                auto nf = normal_form(basis, f, r.names.size(), lex);
                weak = nf.weak;
                output = {std::move(nf.unit), std::move(nf.remainder)};
                stage("local_normal_form");
            }
        }
        auto back = r.point;
        for (auto& q : back) q = -q;
        output = translate_unbounded(output, back);
        stage("restore_point_coordinates");
    }
    Strings polynomials;
    for (const auto& f : output) polynomials.push_back(format_polynomial(f, r.names));
    Json::Value result;
    result["scope"] = "point";
    if (r.query) {
        result["unit"] = polynomials.at(0);
        result["remainder"] = polynomials.at(1);
        result["reduction"] = weak ? "weak" : "standard";
    } else result["generators"] = array(polynomials);
    return result;
}

Json::Value decomposition_result(const Request& r, const Basis& input, Stages& stage) {
    stage("parse_input");
    const auto components = atlas::primary::decompose(input, r.names.size(), r.order == "lex");
    stage("primary_decomposition");
    stage.values[stage.values.size()-1]["generators"] = Json::UInt64(r.generators.size());
    auto ideal = [&](const Basis& rows) {
        Json::Value generators(Json::arrayValue);
        for (const auto& f : rows) generators.append(format_polynomial(f, r.names));
        return generators;
    };
    Json::Value result;
    result["components"] = Json::Value(Json::arrayValue);
    for (const auto& component : components) {
        Json::Value pair;
        pair["primary"] = ideal(component.primary);
        pair["prime"] = ideal(component.prime);
        result["components"].append(std::move(pair));
    }
    return result;
}

std::string centered_monomial(const Powers& powers, const Request& r) {
    const bool product = std::count_if(powers.begin(), powers.end(), [](const auto& e) { return e != 0; }) > 1;
    Strings factors;
    for (size_t j = 0; j < powers.size(); ++j) {
        if (powers[j] == 0) continue;
        std::string axis = r.names[j];
        if (r.point[j] != 0) {
            const mpq_class magnitude = abs(r.point[j]);
            axis += (r.point[j] > 0 ? "-" : "+") + magnitude.get_str();
            if (product || powers[j] > 1) axis = "(" + axis + ")";
        }
        if (powers[j] > 1) axis += "^" + powers[j].get_str();
        factors.push_back(std::move(axis));
    }
    return factors.empty() ? "1" : join(factors, "*");
}

Json::Value local_quotient_result(const Request& r, const Basis& input, Stages& stages) {
    stages("parse_local_input");
    auto q = quotient_at_point(r.operation, input, r.names, r.point, r.order == "lex");
    auto label = r.operation;
    std::replace(label.begin(), label.end(), '-', '_');
    stages(label.c_str());
    const bool number = r.operation == "milnor-number" || r.operation == "tjurina-number";
    Json::Value out;
    out["type"] = number ? "integer" : r.operation == "local-quotient-basis" ? "quotient-basis" : r.operation;
    out["scope"] = r.point_given ? "point" : "origin";
    out["finite"] = q.dimension.has_value();
    auto& length = out[number ? "value" : "dimension"];
    if (q.dimension) {
        if (q.dimension->fits_ulong_p()) length = Json::UInt64(q.dimension->get_ui());
        else length = q.dimension->get_str(); // The output writer emits the unlimited JSON integer.
    }
    if (!number) {
        auto& basis = out["basis"];
        if (q.dimension) {
            basis = Json::Value(Json::arrayValue);
            for (const auto& exponents : q.monomials) basis.append(centered_monomial(exponents, r));
        }
        if (r.operation == "local-jacobian-algebra") {
            auto& generators = out["generators"] = Json::Value(Json::arrayValue);
            for (const auto& row : q.derivatives) generators.append(format_polynomial(row, r.names));
        }
    }
    return out;
}

Json::Value calculate(const Request& r) {
    Stages stages;
    Basis input;
    for (const auto& source : r.generators) {
        auto f = parse_expression(r.names, source);
        if (!f.empty() || global_ideal_query(r.operation) || r.operation == "ae-codimension") input.push_back(std::move(f));
    }
    Json::Value doc, ring;
    ring["variables"] = array(r.names);
    ring["field"] = "QQ";
    ring["order"] = r.order;
    auto& result = doc["result"];
    if (r.operation == "ae-codimension") {
        result = ae_codimension(input, r.names);
        result["scope"] = "origin";
        result["type"] = "integer";
        stages("ae_codimension");
    } else if (local_quotient(r.operation)) {
        if (r.query) input.push_back(parse_expression(r.names, *r.query));
        result = local_quotient_result(r, input, stages);
    } else if (dimension_query(r.operation)) {
        const bool smoothness = r.operation == "local-smoothness";
        stages("parse_local_input");
        auto d = dimension_at_point(input, r.names, r.point, r.order == "lex", smoothness);
        stages(smoothness ? "local_smoothness" : "local_dimension");
        result["local_dimension"] = d.dimension;
        result["scope"] = r.point_given ? "point" : "origin";
        if (smoothness) {
            result["type"] = "boolean";
            result["value"] = d.smooth;
            result["jacobian_rank"] = d.jacobian_rank;
            result["embedding_dimension"] = d.embedding_dimension;
        } else {
            result["type"] = "integer";
            result["value"] = d.dimension;
        }
    } else if (r.operation == "quotient-ring") {
        Basis operands;
        if (r.query) operands.push_back(parse_expression(r.names, *r.query));
        if (r.other_query) operands.push_back(parse_expression(r.names, *r.other_query));
        auto exponent = atlas::quotient_ring::parse_exponent(r.exponent.value_or("0"));
        stages("parse_input");
        result = atlas::quotient_ring::compute(input, r.names, r.order == "lex", r.action, operands, exponent);
        stages("quotient_ring");
    } else if (power_query(r.operation)) {
        const auto exponent = atlas::ideal_powers::parse_exponent(*r.exponent);
        const bool symbolic = r.operation == "symbolic-power";
        stages("parse_input");
        result = atlas::ideal_powers::compute(input,r.names,r.order == "lex",exponent,symbolic);
        stages(symbolic ? "symbolic_power" : "ideal_power");
    } else if (r.operation == "localization") {
        Basis prime;
        for (const auto& source : r.primes) prime.push_back(parse_expression(r.names, source));
        std::optional<atlas::localization::Fraction> fraction;
        if (r.numerator || r.denominator)
            fraction = atlas::localization::Fraction{parse_expression(r.names, r.numerator.value_or("1")),
                                                     parse_expression(r.names, r.denominator.value_or("1"))};
        stages("parse_input");
        result = atlas::localization::compute(input, prime, r.names, r.order == "lex", fraction);
        stages("localization");
    } else if (r.operation == "normalization") {
        stages("parse_input");
        result = atlas::normalization::compute(input, r.names, r.order == "lex");
        stages("normalization");
        stages.values[stages.values.size()-1]["generators"] = Json::UInt64(r.generators.size());
    } else {
        result = r.operation == "primary-decomposition" ? decomposition_result(r, input, stages)
                                                        : local_result(r, input, stages);
        result["type"] = r.operation;
    }
    result["ring"] = ring;
    Strings point;
    if (!(dimension_query(r.operation) || local_quotient(r.operation)) || r.point_given)
        for (const auto& q : r.point) point.push_back(q.get_str());
    if (r.point_given || !point.empty()) result["point"] = array(point);
    doc["status"] = "ok";
    doc["operation"] = r.operation;
    doc["replay"] = replay(r,ring,point);
    auto& ex = doc["execution"];
    ex["backend"] = "cpu";
    ex["device"] = "CPU";
    ex["implementation"] = "native-cli";
    ex["requested"] = r.compute;
    ex["fallback_reasons"] = Json::Value(Json::arrayValue);
    ex["stages"] = std::move(stages.values);
    ex["elapsed_ms"] = std::chrono::duration<double,std::milli>(Clock::now()-stages.start).count();
    return doc;
}

// JsonCpp represents integers only up to uint64_t. Serialize a larger exact
// length at its own field while leaving every other value to the JSON writer.
void write_record(std::ostream& stream, const Json::Value& record, const std::string& operation) {
    const auto& result = record["result"];
    const char* key = result["type"] == "integer" ? "value" : "dimension";
    if ((!local_quotient(operation) && operation != "ae-codimension") || !result[key].isString()) {
        stream << json_text(record) << '\n';
        return;
    }
    Json::Value envelope = record;
    Json::Value fields, length;
    envelope.removeMember("result", &fields);
    fields.removeMember(key, &length);
    // Both objects retain required fields (e.g. ring and execution).
    stream << "{\"result\":{\"" << key << "\":" << length.asString() << ','
           << json_text(fields).substr(1) << ',' << json_text(envelope).substr(1) << '\n';
}

// The Python entry point has already validated the ring, operation and ordered
// components. Run under its ordinary worker supervision and reuse the complete
// CLI record, including the arbitrary-precision integer writer and replay.
int ae_worker() {
    try {
        Json::Value data;
        if (!(std::cin >> data)) return 75;
        Request request;
        request.operation = "ae-codimension";
        request.compute = data["compute"].asString();
        if (request.compute != "cpu" &&
                (request.compute != "auto" || needs_device_dispatch() || std::getenv("ATLAS_GROEBNER_GPU")))
            return 75;
        const auto& ring = data["ring"];
        request.order = ring["order"].asString();
        for (const auto& name : ring["variables"]) request.names.push_back(name.asString());
        for (const auto& component : data["operands"]["components"])
            request.generators.push_back(component.asString());
        const auto result = calculate(request);
        std::cout << "[200,";
        write_record(std::cout, result, request.operation);
        std::cout << "]\n";
        return 0;
    } catch (const std::exception&) {
        // The shared worker distinguishes input limits, grammar errors and
        // computation failures. A partially completed request never succeeds.
        return 75;
    }
}

void show(const Request& r, const Json::Value& result) {
    Strings point;
    for (const auto& q : r.point) point.push_back(q.get_str());
    std::cout << (r.names.empty() ? "QQ" : "QQ[" + join(r.names,", ") + "]") << "   " << r.order << '\n';
    if ((dimension_query(r.operation) || local_quotient(r.operation) || r.operation == "ae-codimension") && !r.point_given) std::cout << "Scope: origin\n";
    else if (r.point_given || !point.empty()) std::cout << "Scope: point\nPoint: (" << join(point,", ") << ")\n";
    auto generators = [](const Json::Value& gens, const char* zero) {
        Strings polys;
        for (const auto& g : gens) polys.push_back(g.asString());
        return polys.empty() ? std::string(zero) : join(polys, ", ");
    };
    if (r.operation == "ae-codimension") {
        std::cout << "Ae-codimension: ";
        if (result["finite"].isNull()) std::cout << "inconclusive\n" << result["reason"].asString();
        else if (result["finite"].asBool()) std::cout << result["value"].asString();
        else std::cout << "infinite";
        std::cout << '\n';
    } else if (local_quotient(r.operation)) {
        const bool number = result["type"] == "integer";
        const auto& length = result[number ? "value" : "dimension"];
        if (result.isMember("generators"))
            std::cout << "Jacobian generators: " << generators(result["generators"], "(none)") << '\n';
        const char* label = number ? (r.operation == "milnor-number" ? "Milnor number" : "Tjurina number")
                                   : "Vector space dimension over QQ";
        std::cout << label << ": " << (length.isNull() ? "infinite" : length.asString()) << '\n';
        if (!number && !length.isNull()) std::cout << "Basis: " << generators(result["basis"], "(empty)") << '\n';
    } else if (r.operation == "quotient-ring") {
        std::cout << "Relations: (" << generators(result["relations"], "0") << ")\n";
        if (result.isMember("representative")) std::cout << "Representative: " << result["representative"].asString() << '\n';
        if (result.isMember("value")) std::cout << "Result: " << (result["value"].asBool() ? "true" : "false") << '\n';
    } else if (power_query(r.operation)) {
        std::cout << "Generators: " << generators(result["generators"], "(zero ideal)") << '\n';
    } else if (r.operation == "localization") {
        std::cout << "Prime ideal: (" << generators(result["prime"], "0") << ")\n"
                  << "Contracted ideal: (" << generators(result["generators"], "0") << ")\n";
        if (result.isMember("element")) {
            const auto& element = result["element"];
            std::cout << "Fraction: (" << element["numerator"].asString() << ")/(" << element["denominator"].asString() << ")\n"
                      << "In the localized ideal: " << (element["member"].asBool() ? "yes" : "no") << '\n'
                      << "Unit in the local ring: " << (element["unit"].asBool() ? "yes" : "no") << '\n';
        }
    } else if (r.operation == "normalization") {
        if (!result["input_reduced"].asBool())
            std::cout << "The input has nilpotents. Showing the normalization of its reduction.\n";
        const auto& components = result["components"];
        if (components.empty()) std::cout << "The zero algebra has no components.\n";
        for (Json::ArrayIndex i = 0; i < components.size(); ++i) {
            const auto& c = components[i];
            const auto& names = c["ring"]["variables"];
            std::cout << "Component " << i+1 << ", prime ideal (" << generators(c["prime"], "0") << ")\n"
                      << "Normalized algebra: QQ[" << generators(names, "") << "]/(" << generators(c["relations"], "0") << ")\n";
            for (Json::ArrayIndex j = 0; j < c["images"].size(); ++j)
                std::cout << "  " << r.names[j] << " -> " << c["images"][j].asString() << '\n';
            for (Json::ArrayIndex j = 0; j < names.size(); ++j)
                std::cout << "  " << names[j].asString() << " = (" << c["fractions"][j]["numerator"].asString()
                          << ")/(" << c["fractions"][j]["denominator"].asString() << ")\n";
        }
    } else if (r.operation == "primary-decomposition") {
        const auto& components = result["components"];
        if (components.empty()) std::cout << "The unit ideal has no primary components.\n";
        for (Json::ArrayIndex i = 0; i < components.size(); ++i) {
            std::cout << "Primary component " << i+1 << ": (" << generators(components[i]["primary"], "0") << ")\n";
            std::cout << "Associated prime: (" << generators(components[i]["prime"], "0") << ")\n";
        }
    } else if (dimension_query(r.operation)) {
        std::cout << "Local dimension: " << result["local_dimension"].asInt() << '\n';
        if (r.operation == "local-smoothness")
            std::cout << "Embedding dimension: " << result["embedding_dimension"].asInt()
                      << "\nJacobian rank: " << result["jacobian_rank"].asInt()
                      << "\nSmooth at point: " << (result["value"].asBool() ? "yes" : "no") << '\n';
    } else if (r.query)
        std::cout << "Reduction: " << result["reduction"].asString() << "\nRemainder: "
                  << result["remainder"].asString() << "\nUnit: " << result["unit"].asString() << '\n';
    else std::cout << "Generators: " << generators(result["generators"], "(zero ideal)") << '\n';
    std::cout << "Device: CPU (requested " << r.compute << ")\n";
}

int shared_frontend(char** argv) {
    char python[] = "python3";
    argv[0] = python;
    execvp(python,argv);
    std::perror("python3");
    return 127;
}

volatile std::sig_atomic_t interrupted = 0;
volatile std::sig_atomic_t active_child = 0;
void interrupt(int) {
    const int saved_errno = errno;
    interrupted = 1;
    if (active_child) kill(active_child, SIGKILL);
    errno = saved_errno;
}

int cli_error(const Request& r, const char* code, const char* message) {
    if (r.json) {
        Json::Value failure;
        failure["status"] = "error";
        failure["error"]["code"] = code;
        failure["error"]["message"] = message;
        std::cout << json_text(failure) << '\n' << std::flush;
    } else std::cerr << message << '\n';
    return 2;
}

int run_isolated(const Request& r, char** argv) {
    // The public CLI has no deadline. Keep the kernel in a separate process
    // so Ctrl-C and a kernel terminated by OS resource limits have the
    // same exit status and diagnostics as the shared worker.
    struct sigaction action{}, previous{};
    action.sa_handler = interrupt;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &previous)) return shared_frontend(argv);
    const auto child = fork();
    if (child <= 0) {
        sigaction(SIGINT, &previous, nullptr);
        if (child < 0) return shared_frontend(argv);
        try {
            const auto record = calculate(r);
            if (r.json) std::cout << json_text(record) << '\n';
            else show(r,record["result"]);
            std::cout.flush();
            _exit(0);
        } catch (const atlas::quotient_ring::InvalidInput& error) {
            _exit(cli_error(r, "invalid_request", error.what()));
        } catch (const atlas::ideal_powers::InvalidInput& error) {
            _exit(cli_error(r, "invalid_request", error.what()));
        } catch (const atlas::localization::InvalidInput& error) {
            _exit(cli_error(r, "invalid_request", error.what()));
        } catch (const std::exception&) {
            // Let the original command parser and worker classify a failed
            // parse or computation. Replacement happens in the parent.
            _exit(1);
        }
    }
    active_child = child;
    if (interrupted) kill(child, SIGKILL);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child,&status,0); } while (waited < 0 && errno == EINTR);
    active_child = 0;
    sigaction(SIGINT, &previous, nullptr);
    if (waited < 0) return shared_frontend(argv);
    if (interrupted || (WIFSIGNALED(status) && WTERMSIG(status) == SIGINT)) {
        std::cerr << "Algebra calculation interrupted.\n";
        return 130;
    }
    if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
        return code == 0 || code == 2 ? code : shared_frontend(argv);
    }
    return cli_error(r, "worker_unavailable", "Algebra worker stopped before returning an exact result.");
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--ae-worker") return ae_worker();
    if (argc < 4 || std::string(argv[2]) != "algebra" ||
        (std::string(argv[3]) != "local-normal-form" && std::string(argv[3]) != "local-standard-basis" &&
         std::string(argv[3]) != "ae-codimension" && !global_ideal_query(argv[3]) && !dimension_query(argv[3]) && !local_quotient(argv[3])))
        return 2;
    try {
        auto request = arguments(argc,argv);
        if (request.compute == "auto" && needs_device_dispatch())
            throw std::invalid_argument("Use the configured device policy.");
        if (request.compute == "auto" && dimension_query(request.operation) &&
            std::min(request.names.size(), request.generators.size()) >= 256)
            throw std::invalid_argument("Use the shared Jacobian device selector.");
        if (global_ideal_query(request.operation)) return run_isolated(request,argv);
        Json::Value result;
        try {
            result = calculate(request);
        } catch (const HypersurfaceOutside& error) {
            return cli_error(request, "invalid_request", error.what());
        } catch (const PointOutsideScheme& error) {
            if (request.json) {
                Json::Value failure;
                failure["status"] = "error";
                failure["error"]["code"] = "invalid_request";
                failure["error"]["message"] = error.what();
                std::cout << json_text(failure) << '\n';
            } else std::cerr << error.what() << '\n';
            return 2;
        }
        if (request.json) write_record(std::cout, result, request.operation);
        else show(request,result["result"]);
        return 0;
    } catch (const std::exception&) {
        // Preserve argparse diagnostics, device execution, wide arithmetic and
        // the public worker's error records when this frontend cannot handle a request.
        return shared_frontend(argv);
    }
}
