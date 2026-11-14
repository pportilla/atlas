CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ifeq ($(shell uname -s),Darwin)
  GROEBNER_LIB := build/libgroebner.dylib
  GROEBNER_SHARED := -dynamiclib
  GROEBNER_ID := -Wl,-install_name,@rpath/$(notdir $(GROEBNER_LIB))
  LOCAL_RPATH := -Wl,-rpath,@loader_path
else
  GROEBNER_LIB := build/libgroebner.so
  GROEBNER_SHARED := -shared -Wl,--no-undefined
  LOCAL_RPATH := -Wl,-rpath,'$$ORIGIN'
endif

.PHONY: all
all: $(GROEBNER_LIB) build/graph_verify build/local_algebra build/primary_decomposition build/normalization build/localization build/ideal_power build/symbolic_power build/quotient_ring

build/graph_verify: graph_verify.cpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner

LOCAL_SOURCES := local_cli.cpp
build/bench_local_dimension: ../benchmarks/local_dimension_kernel.cpp local_dimension.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/bench_ae_codimension: ../benchmarks/ae_codimension_kernel.cpp ae_codimension.hpp ae_graded.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/bench_ideal_powers: ../benchmarks/ideal_powers_kernel.cpp ideal_powers.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/local_algebra: $(LOCAL_SOURCES) ae_codimension.hpp local_io.hpp local_basis.hpp local_division.hpp primary_decomposition.hpp normalization.hpp localization.hpp ideal_powers.hpp quotient_ring.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $(LOCAL_SOURCES) -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

# Install a command marker only once the corresponding native frontend exists.
build/primary_decomposition build/normalization build/localization build/ideal_power build/symbolic_power build/quotient_ring: build/local_algebra
	ln -sf local_algebra $@

KERNEL_SOURCES := trace_levels.cpp groebner.cpp ideal.cpp modular.cpp modular_replay.cpp hilbert.cpp euler.cpp euler_lattice.cpp affine.cpp zeta.cpp \
  field_wide.cpp field_coefficient.cpp common_field.cpp field_factorization.cpp marked_scalar.cpp lattice_volume.cpp toric_graph.cpp fan_cover.cpp centre_weights.cpp binomial_fan.cpp \
  atlas_incidence.cpp torus_sections.cpp monomial_resolution.cpp component_geometry.cpp finite_sheet_incidence.cpp identity_divisor.cpp monomial_quotient.cpp rational_arithmetic.cpp \
  triton_kernels.cpp triton_rank.cpp triton_trace.cpp monodromy.cpp point_geometry.cpp \
  quotient_ring.cpp quotient_packed.cpp quotient_inverse.cpp quotient_finite_inverse.cpp ideal_powers.cpp ideal_power_principal.cpp ideal_powers_packed.cpp ideal_powers_dense.cpp ideal_powers_echelon.cpp ideal_powers_content.cpp ideal_powers_monomial.cpp localization.cpp localization_membership.cpp localization_special.cpp normalization.cpp normalization_content.cpp normalization_special.cpp primary_decomposition.cpp primary_special.cpp polynomial_factors.cpp resolution_tangents.cpp \
  ae_codimension.cpp ae_special.cpp ae_graded.cpp ae_graded_surface.cpp ae_surface_coordinates.cpp \
  resolution_candidate.cpp local_operations.cpp local_basis.cpp local_division.cpp local_io.cpp local_power_coordinates.cpp local_dimension_graph.cpp \
  local_point_basis.cpp local_point_normal_form.cpp local_parameter_basis.cpp local_quotient_initial.cpp local_quotient_jets.cpp local_point_quotient.cpp
KERNEL_OBJECTS := $(patsubst %.cpp,build/kernel/%.o,$(KERNEL_SOURCES))

build/kernel/%.o: %.cpp
	@mkdir -p build/kernel
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint jsoncpp) -fPIC -MMD -MP -c $< -o $@

-include $(KERNEL_OBJECTS:.o=.d)

$(GROEBNER_LIB): $(KERNEL_OBJECTS)
	@set -e; output=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$output"' EXIT; \
	  $(CXX) $(GROEBNER_SHARED) $(GROEBNER_ID) $^ -o "$$output" $(LDFLAGS) $(shell pkg-config --libs flint jsoncpp) -lgmpxx -lgmp -pthread; \
	  chmod 755 "$$output"; mv -f "$$output" "$@"


.PHONY: test-verify
test-verify: build/test_groebner_verify build/test_local_dimension_graph
	./build/test_groebner_verify
	./build/test_local_dimension_graph

build/test_local_dimension_graph: test_local_dimension_graph.cpp local_dimension_graph.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp

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

build/bench_quotient_ring: ../benchmarks/quotient_ring_kernel.cpp quotient_ring.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)
