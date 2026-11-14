CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
build/face_gb: face_gb.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs flint)
