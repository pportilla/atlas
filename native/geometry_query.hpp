#pragma once
#include <json/json.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace atlas {
struct GeometryQuery {
    using Callback=const char*(*)(const char*);
    Callback dispatch;
    Json::Value call(const std::string& op, Json::Value args=Json::Value()) const {
        args["query"]=op;
        Json::StreamWriterBuilder w;w["indentation"]="";
        const auto input=Json::writeString(w,args);
        const char* value=dispatch(input.c_str());
        if (!value) throw std::runtime_error("A geometry query was interrupted.");
        Json::Value result;std::string error;Json::CharReaderBuilder parser;std::istringstream in(value);
        if (!Json::parseFromStream(parser,in,&result,&error)) throw std::runtime_error(error);
        return result;
    }
};
}
