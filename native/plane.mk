CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
build/plane: plane.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
