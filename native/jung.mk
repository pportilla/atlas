CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
JUNG ?= build/jung
ifeq ($(shell uname -s),Darwin)
HELPER_SHARED := -dynamiclib
HELPER_SUFFIX := dylib
else
HELPER_SHARED := -shared
HELPER_SUFFIX := so
endif
# The same entry as the program, for solvers that run it in a forked copy of
# themselves instead of starting the program (see engine/native_process.py).
HELPER_FLAGS := -DATLAS_HELPER_LIBRARY -fPIC -fvisibility=hidden $(HELPER_SHARED)
JUNG_LIB := build/libjung.$(HELPER_SUFFIX)
JUNG_DEPS := jung.cpp jung_local.hpp jung_series.hpp plane_resolver.hpp common_field.cpp common_field.hpp algebraic_number.hpp
all: $(JUNG) $(JUNG_LIB)
$(JUNG_LIB): $(JUNG_DEPS)
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $(HELPER_FLAGS) jung.cpp common_field.cpp -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
$(JUNG): $(JUNG_DEPS)
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) jung.cpp common_field.cpp -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
