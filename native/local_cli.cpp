// Fresh-process CPU frontend for the two point-local Atlas commands.
#include "local_io.hpp"
#include "local_division.hpp"
#include <json/json.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace {
using namespace atlas::local;
using Strings = std::vector<std::string>;
using Clock = std::chrono::steady_clock;

struct Request {
    std::string operation, order = "grevlex", compute = "auto";
    Strings names, generators;
    std::vector<mpq_class> point;
    std::optional<std::string> query;
    bool json = false;
};

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
        "--point", "--json", "--help", "-h", "--"
    };
    return flags.count(s.substr(0, s.find('=')));
}

Request arguments(int argc, char** argv) {
    Request r;
    r.operation = argv[3];
    std::optional<Strings> point;
    bool vars = false;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") { r.json = true; continue; }
        const auto eq = arg.find('=');
        auto flag = arg.substr(0, eq);
        if (flag != "--gen" && flag != "--poly" && flag != "--vars" && flag != "--point" &&
            flag != "--field" && flag != "--order" && flag != "--compute")
            throw std::invalid_argument("Use the general command parser.");
        if (eq == arg.npos && (++i == argc || is_option(argv[i])))
            throw std::invalid_argument("Missing argument.");
        auto value = eq == arg.npos ? std::string(argv[i]) : arg.substr(eq+1);
        // Unicode whitespace and argparse abbreviations are accepted by the general frontend.
        if (std::any_of(value.begin(), value.end(), [](unsigned char c){ return c >= 128; }))
            throw std::invalid_argument("Use the Unicode command parser.");
        if (flag == "--gen") r.generators.push_back(value);
        else if (flag == "--poly") r.query = value;
        else if (flag == "--vars") { r.names = comma_list(value); vars = true; }
        else if (flag == "--point") point = comma_list(value);
        else if (flag == "--order") r.order = value;
        else if (flag == "--compute") r.compute = value;
        else if (value != "QQ") throw std::invalid_argument("Unsupported field.");
    }
    if (!vars || r.names.empty() || (r.order != "lex" && r.order != "grevlex") ||
        (r.compute != "auto" && r.compute != "cpu") ||
        (r.operation == "local-normal-form") != r.query.has_value())
        throw std::invalid_argument("Use the general algebra handler.");
    std::set<std::string> distinct;
    for (const auto& name : r.names)
        if (name.empty() || name.size()>24 || !std::isalpha(name.front()) ||
            !std::all_of(name.begin(), name.end(), [](char c){ return std::isalnum(c) || c=='_'; }) ||
            !distinct.insert(name).second) throw std::invalid_argument("Invalid variables.");
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
    for (const auto& g : r.generators) operand("--gen",g);
    if (r.query) operand("--poly",*r.query);
    args.insert(args.end(), {"--point", join(point, ","), "--json"});
    for (auto& arg : args) arg = shell_quote(arg);
    Json::Value request;
    request["operation"] = r.operation;
    request["ring"] = ring;
    request["compute"] = r.compute;
    request["operands"]["ideal"] = array(r.generators);
    request["operands"]["point"] = array(point);
    if (r.query) request["operands"]["polynomial"] = *r.query;
    Json::Value out;
    out["command"] = join(args, " ");
    // algebra_request accepts serialized JSON. Quoting it as a string again
    // gives an Atlas string literal, even for names reserved by the session.
    out["script"] = "result = algebra_request("+json_text(json_text(request))+")\nprint(result.json)\n";
    return out;
}

Json::Value calculate(const Request& r) {
    auto start = Clock::now(), checkpoint = start;
    Json::Value stages(Json::arrayValue);
    auto stage = [&](const char* name, const char* label = nullptr) {
        const auto now = Clock::now();
        Json::Value s;
        s["operation"] = name;
        s["backend"] = "cpu";
        s["elapsed_ns"] = Json::Int64(std::chrono::duration_cast<std::chrono::nanoseconds>(now-checkpoint).count());
        if (label) s["label"] = label;
        stages.append(s);
        checkpoint = now;
    };
    Strings point;
    for (const auto& q : r.point) point.push_back(q.get_str());
    const bool lex = r.order == "lex";
    Basis input;
    for (const auto& source : r.generators) {
        auto f = parse_expression(r.names, source);
        if (!f.empty()) input.push_back(std::move(f));
    }
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
    } else if (shifted && has_point_unit(input, r.point)) {
        output = r.query ? Basis{one, {}} : Basis{one};
        stage("inspect_point_ideal");
    } else {
        Basis centered = translate(input, r.point);
        stage("center_at_point");
        auto finite = r.query ? finite_standard_basis(centered, r.names, lex) : std::nullopt;
        Basis basis = finite ? std::move(*finite) : standard_basis(centered, r.names, lex);
        stage(finite ? "finite_jet_standard_basis" : "local_standard_basis", "local_standard_basis");
        if (r.query) {
            if (basis == Basis{one}) output = {one, {}};
            else {
                auto jet = finite_query_jet(basis, query, r.point, lex);
                Polynomial f = jet ? std::move(*jet) : translate({query}, r.point).front();
                stage("center_query_at_point");
                auto nf = normal_form(basis, f, r.names.size(), lex);
                weak = nf.weak;
                output = {std::move(nf.unit), std::move(nf.remainder)};
                stage("local_normal_form");
            }
        } else output = std::move(basis);
        auto back = r.point;
        for (auto& q : back) q = -q;
        output = translate(output, back);
        stage("restore_point_coordinates");
    }
    Strings polynomials;
    for (const auto& f : output) polynomials.push_back(format_polynomial(f, r.names));
    Json::Value doc, ring;
    ring["variables"] = array(r.names);
    ring["field"] = "QQ";
    ring["order"] = r.order;
    auto& result = doc["result"];
    result["ring"] = ring;
    result["point"] = array(point);
    result["scope"] = "point";
    result["type"] = r.operation;
    if (r.query) {
        result["unit"] = polynomials.at(0);
        result["remainder"] = polynomials.at(1);
        result["reduction"] = weak ? "weak" : "standard";
    } else result["generators"] = array(polynomials);
    doc["status"] = "ok";
    doc["operation"] = r.operation;
    doc["replay"] = replay(r,ring,point);
    auto& ex = doc["execution"];
    ex["backend"] = "cpu";
    ex["device"] = "CPU";
    ex["implementation"] = "native-cli";
    ex["requested"] = r.compute;
    ex["fallback_reasons"] = Json::Value(Json::arrayValue);
    ex["stages"] = std::move(stages);
    ex["elapsed_ms"] = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    return doc;
}

void show(const Request& r, const Json::Value& result) {
    Strings point;
    for (const auto& q : r.point) point.push_back(q.get_str());
    std::cout << "QQ[" << join(r.names,", ") << "]   " << r.order << "\nScope: point\nPoint: ("
              << join(point,", ") << ")\n";
    if (r.query)
        std::cout << "Reduction: " << result["reduction"].asString() << "\nRemainder: "
                  << result["remainder"].asString() << "\nUnit: " << result["unit"].asString() << '\n';
    else {
        Strings polys;
        for (const auto& s : result["generators"]) polys.push_back(s.asString());
        std::cout << "Generators: " << (polys.empty() ? "(zero ideal)" : join(polys,", ")) << '\n';
    }
    std::cout << "Device: CPU (requested " << r.compute << ")\n";
}
}

int main(int argc, char** argv) {
    if (argc < 4 || std::string(argv[2]) != "algebra" ||
        (std::string(argv[3]) != "local-normal-form" && std::string(argv[3]) != "local-standard-basis"))
        return 2;
    try {
        auto request = arguments(argc,argv);
        if (request.compute == "auto" && needs_device_dispatch())
            throw std::invalid_argument("Use the configured device policy.");
        auto result = calculate(request);
        if (request.json) std::cout << json_text(result) << '\n';
        else show(request,result["result"]);
        return 0;
    } catch (const std::exception&) {
        // Preserve argparse diagnostics, device execution, wide arithmetic and
        // the public worker's error records when this frontend cannot handle a request.
        char python[] = "python3";
        argv[0] = python;
        execvp(python,argv);
        std::perror("python3");
        return 127;
    }
}
