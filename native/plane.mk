CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
PLANE ?= build/plane
ifeq ($(shell uname -s),Darwin)
HELPER_SHARED := -dynamiclib
HELPER_SUFFIX := dylib
else
HELPER_SHARED := -shared
HELPER_SUFFIX := so
endif
# The same entry as the program, for solvers that run it in a forked copy of
# themselves instead of starting the program.
HELPER_FLAGS := -DATLAS_HELPER_LIBRARY -fPIC -fvisibility=hidden $(HELPER_SHARED)
PLANE_LIB := build/libplane.$(HELPER_SUFFIX)
all: $(PLANE) $(PLANE_LIB)
$(PLANE_LIB): plane.cpp common_field.cpp common_field.hpp algebraic_number.hpp binomial_root_family.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $(HELPER_FLAGS) plane.cpp common_field.cpp -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
$(PLANE): plane.cpp common_field.cpp common_field.hpp algebraic_number.hpp binomial_root_family.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) plane.cpp common_field.cpp -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
