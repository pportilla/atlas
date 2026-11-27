// Optional differential replay of the decisions made inside one centre search.
#pragma once

#include "chart_expr.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

namespace atlas::centre_record {
using namespace chart;
using Clock = std::chrono::steady_clock;

inline std::string compact(const V& v) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, v);
}

// A worker may fork with the library already loaded. Resolve the filename at
// each write so two workers cannot share a buffered stream or interleave rows.
inline void append(const char* dir, const char* prefix, const V& record) {
    auto path = std::filesystem::path(dir);
    std::filesystem::create_directories(path);
    std::ostringstream name;
    name << prefix << '-' << getpid() << '-' << std::this_thread::get_id() << ".jsonl";
    std::ofstream out(path / name.str(), std::ios::app);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << compact(record) << '\n';
}

inline V inputs(const char* kind, const Variables& vars, const Ideal& equations, const Ideal& extra = {}) {
    V r;
    r["operation"] = "replay_inner";
    r["kind"] = kind;
    r["names"] = V(Json::arrayValue);
    r["identities"] = V(Json::arrayValue);
    for (const auto& v : vars) {
        r["names"].append(v.name);
        r["identities"].append(Json::Int64(v.identity));
    }
    r["equations"] = json_rows(vars, equations);
    r["extra"] = json_rows(vars, extra);
    return r;
}

inline void inner(const char* dir, V request, bool packed, const V& value, Clock::duration elapsed) {
    request["packed_present"] = packed;
    V record, response;
    response["result"] = value;
    record["transport"] = "json";
    record["symbol"] = "atlas_centre_maximum";
    record["request"] = compact(request);
    record["response"] = compact(response);
    record["elapsed_us"] = Json::Int64(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    append(dir, "requests-centre", record);
}
}
