#pragma once
#include <gmpxx.h>
#include <json/json.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace atlas::sparse_json {
using Z = mpz_class;
using Q = mpq_class;
using Powers = std::vector<Z>;
using Row = std::vector<Q>;
using Terms = std::map<Powers, Q>;
using Matrix = std::vector<Row>;

inline Powers read_exponents(const Json::Value& array, size_t width) {
    if (!array.isArray() || array.size() != width) throw std::invalid_argument("Invalid exponent vector.");
    Powers e;
    for (const auto& item : array) {
        e.emplace_back(item.asString(), 10);
        if (e.back() < 0) throw std::invalid_argument("Negative polynomial exponent.");
    }
    return e;
}
inline Row read_row(const Json::Value& array) {
    Row row;
    for (const auto& item : array) {
        row.emplace_back(item.asString(), 10);
        if (row.back().get_den() == 0) throw std::invalid_argument("Zero rational denominator.");
        row.back().canonicalize();
    }
    return row;
}
inline Terms read_terms(const Json::Value& data, size_t width) {
    Terms terms;
    for (const auto& entry : data) {
        Q c(entry[0].asString(), 10);
        if (c.get_den() == 0) throw std::invalid_argument("Zero rational denominator.");
        c.canonicalize();
        if (c != 0) terms[read_exponents(entry[1], width)] += c;
    }
    return terms;
}
template<class Vector>
Json::Value strings(const Vector& values) {
    Json::Value out(Json::arrayValue);
    for (const auto& n : values) out.append(n.get_str());
    return out;
}
inline Json::Value term_list(const Terms& terms) {
    Json::Value out(Json::arrayValue);
    for (const auto& [e, c] : terms) {
        if (c == 0) continue;
        Json::Value entry(Json::arrayValue);
        entry.append(c.get_str());
        entry.append(strings(e));
        out.append(entry);
    }
    return out;
}

template<class Operation>
const char* invoke(const char* input, Operation compute) noexcept {
    static thread_local std::string text;
    try {
        Json::Value request, response;
        Json::CharReaderBuilder parser;
        std::istringstream stream(input);
        std::string error;
        if (!Json::parseFromStream(parser, stream, &request, &error)) throw std::invalid_argument(error);
        response["result"] = compute(request);
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        text = Json::writeString(writer, response);
    } catch (const std::exception& ex) {
        Json::Value error;
        error["error"] = ex.what();
        text = Json::writeString(Json::StreamWriterBuilder(), error);
    }
    return text.c_str();
}
}  // namespace atlas::sparse_json
