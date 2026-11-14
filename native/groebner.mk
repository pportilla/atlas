CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ifeq ($(shell uname -s),Darwin)
  GROEBNER_LIB := build/libgroebner.dylib
  GROEBNER_SHARED := -dynamiclib
else
  GROEBNER_LIB := build/libgroebner.so
  GROEBNER_SHARED := -shared
endif

.PHONY: all
all: $(GROEBNER_LIB)

$(GROEBNER_LIB): trace_levels.cpp groebner.cpp ideal.cpp modular.cpp hilbert.cpp euler.cpp affine.cpp zeta.cpp field_wide.cpp field_coefficient.cpp lattice_volume.cpp fan_cover.cpp centre_weights.cpp binomial_fan.cpp groebner.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) -fPIC $(GROEBNER_SHARED) $(filter %.cpp,$^) -o $@ $(LDFLAGS) $(shell pkg-config --libs flint) -pthread

$(GROEBNER_LIB): trace_screen.inc trace_rows.inc groebner_verify.hpp trace_levels.hpp

.PHONY: test-verify
test-verify: build/test_groebner_verify
	./build/test_groebner_verify

build/test_groebner_verify: test_groebner_verify.cpp groebner_verify.hpp groebner.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) $< $(abspath $(GROEBNER_LIB)) -o $@ $(LDFLAGS) $(shell pkg-config --libs flint) -pthread

# The optional OpenCL plugin screens trace rows and replays prime batches.
ifeq ($(shell uname -s),Darwin)
  GROEBNER_GPU_LIB := build/libgroebner_opencl.dylib
  OPENCL_LIBS ?= -framework OpenCL
else
  GROEBNER_GPU_LIB := build/libgroebner_opencl.so
  OPENCL_LIBS ?= -lOpenCL
  ifeq ($(wildcard /usr/include/CL/cl.h),)
    OPENCL_INCLUDE := $(firstword $(foreach dir,/opt/rocm/include /opt/cuda/include,$(if $(wildcard $(dir)/CL/cl.h),$(dir))))
  endif
endif
OPENCL_CFLAGS ?= $(if $(OPENCL_INCLUDE),-I$(OPENCL_INCLUDE))
.PHONY: gpu
gpu: all $(GROEBNER_GPU_LIB)

$(GROEBNER_GPU_LIB): opencl_groebner.cpp trace_levels.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCL_CFLAGS) -fPIC $(GROEBNER_SHARED) $< -o $@ $(LDFLAGS) $(OPENCL_LIBS)
