CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
NORMALIZ_LIBS ?= -lnormaliz $(shell pkg-config --exists e-antic && pkg-config --libs e-antic || (test -e /usr/include/e-antic/renf_class.hpp && echo -leanticxx)) -lgmpxx -lgmp

.PHONY: all
ifeq ($(shell uname -s),Darwin)
SMALL_FAN := build/libsmall_newton_fan.dylib
SMALL_FAN_SHARED := -dynamiclib
else
SMALL_FAN := build/libsmall_newton_fan.so
SMALL_FAN_SHARED := -shared
endif
all: build/newton_fan $(SMALL_FAN)

$(SMALL_FAN): newton_fan.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -DATLAS_SMALL_NEWTON_FAN -fPIC $(SMALL_FAN_SHARED) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint) -lgmpxx -lgmp

build/newton_fan: newton_fan.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $< -o $@ $(LDFLAGS) $(NORMALIZ_LIBS) $(shell pkg-config --libs jsoncpp flint)
