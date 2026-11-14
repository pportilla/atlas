CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ifeq ($(shell uname -s),Darwin)
  GROEBNER_LIB := build/libgroebner.dylib
  GROEBNER_SHARED := -dynamiclib
  GROEBNER_ID := -Wl,-install_name,@rpath/$(notdir $(GROEBNER_LIB))
  LOCAL_RPATH := -Wl,-rpath,@loader_path
else
  GROEBNER_LIB := build/libgroebner.so
  GROEBNER_SHARED := -shared
  LOCAL_RPATH := -Wl,-rpath,'$$ORIGIN'
endif

.PHONY: all
all: $(GROEBNER_LIB) build/local_algebra

LOCAL_SOURCES := local_cli.cpp
build/local_algebra: $(LOCAL_SOURCES) local_io.hpp local_basis.hpp local_division.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $(LOCAL_SOURCES) -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

KERNEL_SOURCES := trace_levels.cpp groebner.cpp ideal.cpp modular.cpp hilbert.cpp euler.cpp affine.cpp zeta.cpp \
  field_wide.cpp field_coefficient.cpp lattice_volume.cpp fan_cover.cpp centre_weights.cpp binomial_fan.cpp \
  monomial_quotient.cpp point_geometry.cpp polynomial_factors.cpp local_operations.cpp local_basis.cpp local_division.cpp local_io.cpp
KERNEL_OBJECTS := $(patsubst %.cpp,build/kernel/%.o,$(KERNEL_SOURCES))

build/kernel/%.o: %.cpp
	@mkdir -p build/kernel
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint jsoncpp) -fPIC -MMD -MP -c $< -o $@

-include $(KERNEL_OBJECTS:.o=.d)

$(GROEBNER_LIB): $(KERNEL_OBJECTS)
	$(CXX) $(GROEBNER_SHARED) $(GROEBNER_ID) $^ -o $@.tmp $(LDFLAGS) $(shell pkg-config --libs flint jsoncpp) -lgmpxx -lgmp -pthread
	mv $@.tmp $@

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
