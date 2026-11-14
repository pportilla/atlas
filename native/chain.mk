CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ifeq ($(shell uname -s),Darwin)
  CHAIN_LIB := build/libchain.dylib
  CHAIN_SHARED := -dynamiclib
else
  CHAIN_LIB := build/libchain.so
  CHAIN_SHARED := -shared
endif

.PHONY: all
all: $(CHAIN_LIB)

$(CHAIN_LIB): chain.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(CHAIN_SHARED) $< -o $@ $(LDFLAGS) -lgmpxx -lgmp
