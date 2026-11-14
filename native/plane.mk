CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
PLANE ?= build/plane
$(PLANE): plane.cpp common_field.cpp common_field.hpp algebraic_number.hpp binomial_root_family.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) plane.cpp common_field.cpp -o $@ $(LDFLAGS) $(shell pkg-config --libs jsoncpp flint)
