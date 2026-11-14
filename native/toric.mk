CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all
all: build/toricfan

build/toricfan: toricfan.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
