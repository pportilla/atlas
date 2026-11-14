CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
NORMALIZ_LIBS ?= -lnormaliz $(shell pkg-config --exists e-antic && pkg-config --libs e-antic || (test -e /usr/include/e-antic/renf_class.hpp && echo -leanticxx)) -lgmpxx -lgmp

.PHONY: all
all: build/newton_fan

build/newton_fan: newton_fan.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $< -o $@ $(LDFLAGS) $(NORMALIZ_LIBS) $(shell pkg-config --libs jsoncpp flint)
