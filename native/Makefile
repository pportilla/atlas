CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ifeq ($(shell uname -s),Darwin)
  LIBRARY := build/libresolution.dylib
  SHARED := -dynamiclib
else
  LIBRARY := build/libresolution.so
  SHARED := -shared
endif

.PHONY: all clean
ZOO_LIBRARY := $(subst libresolution,libzoo_search,$(LIBRARY))
all: $(LIBRARY) $(ZOO_LIBRARY)
ifeq ($(shell uname -s),Linux)
all: build/resolve_supervisor
endif

build/resolve_supervisor: resolve_supervisor.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(LIBRARY): ordinary.cpp chart_batch.cpp chart_equation.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(SHARED) $(filter %.cpp,$^) -o $@ $(LDFLAGS) -lgmpxx -lgmp

$(ZOO_LIBRARY): zoo_search.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(SHARED) $< -o $@ $(LDFLAGS)

clean:
	$(RM) -r build

# On Linux the OpenCL headers and loader can come from different SDKs.
ifeq ($(shell uname -s),Darwin)
  GPU_LIBRARY := build/libresolution_opencl.dylib
  OPENCL_LIBS ?= -framework OpenCL
else
  GPU_LIBRARY := build/libresolution_opencl.so
  OPENCL_LIBS ?= -lOpenCL
  ifeq ($(wildcard /usr/include/CL/cl.h),)
    OPENCL_INCLUDE := $(firstword $(foreach dir,/opt/rocm/include /opt/cuda/include,$(if $(wildcard $(dir)/CL/cl.h),$(dir))))
  endif
endif
OPENCL_CFLAGS ?= $(if $(OPENCL_INCLUDE),-I$(OPENCL_INCLUDE))
.PHONY: gpu
gpu: all $(GPU_LIBRARY)

$(GPU_LIBRARY): opencl_rank.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENCL_CFLAGS) -fPIC $(SHARED) $< -o $@ $(LDFLAGS) $(OPENCL_LIBS)
