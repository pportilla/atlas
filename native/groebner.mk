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
all: $(GROEBNER_LIB) build/graph_verify build/local_algebra build/primary_decomposition build/normalization build/localization build/ideal_sum build/ideal_product build/ideal_power build/symbolic_power build/quotient_ring

# A headerless installation uses the same kernel through ctypes. When
# available, build one stable-ABI binding usable by CPython 3.10 and newer.
PYTHON ?= python3
PYTHON_INCLUDE := $(shell $(PYTHON) -c 'import os,sys,sysconfig; p=sysconfig.get_path("include"); print(p if sys.implementation.name=="cpython" and sys.version_info>=(3,10) and not sysconfig.get_config_var("Py_GIL_DISABLED") and os.path.isfile(os.path.join(p,"Python.h")) else "")' 2>/dev/null)
ifneq ($(PYTHON_INCLUDE),)
all: build/_atlas_ideal.abi3.so
ifeq ($(shell uname -s),Darwin)
PYTHON_MODULE_LINK := -bundle -undefined dynamic_lookup
else
PYTHON_MODULE_LINK := -shared
endif
build/_atlas_ideal.abi3.so: python_ideal.cpp python_session.cpp prepared_ideal.hpp $(GROEBNER_LIB)
	@set -e; output=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$output"' EXIT; \
	  $(CXX) $(CPPFLAGS) $(CXXFLAGS) -I"$(PYTHON_INCLUDE)" -fPIC $(PYTHON_MODULE_LINK) $(filter %.cpp,$^) -o "$$output" $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner; \
	  chmod 755 "$$output"; mv -f "$$output" "$@"
endif

build/graph_verify: graph_verify.cpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner

LOCAL_SOURCES := local_cli.cpp
build/bench_local_dimension: ../benchmarks/local_dimension_kernel.cpp local_dimension.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/bench_groebner_basis: ../benchmarks/bench_groebner_basis.cpp groebner.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner $(shell pkg-config --libs flint)

# This independent comparison executable uses the headers and libraries of the installed Singular.
SINGULAR_CFLAGS ?= $(shell pkg-config --cflags Singular)
SINGULAR_LIBS ?= $(shell pkg-config --libs Singular)
build/bench_singular_basis: ../benchmarks/bench_singular_basis.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SINGULAR_CFLAGS) $< -o $@ $(SINGULAR_LIBS) -lgmpxx -lgmp

build/bench_ae_codimension: ../benchmarks/ae_codimension_kernel.cpp ae_codimension.hpp ae_graded.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/bench_ideal_powers: ../benchmarks/ideal_powers_kernel.cpp ideal_powers.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/bench_ideal_operators: ../benchmarks/ideal_operators_kernel.cpp prepared_ideal.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) -I. $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

build/local_algebra: $(LOCAL_SOURCES) ae_codimension.hpp local_io.hpp local_basis.hpp local_division.hpp primary_decomposition.hpp normalization.hpp localization.hpp ideal_arithmetic.hpp ideal_powers.hpp quotient_ring.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags jsoncpp flint) $(LOCAL_SOURCES) -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp flint)

# Install a command marker only once the corresponding native frontend exists.
build/primary_decomposition build/normalization build/localization build/ideal_sum build/ideal_product build/ideal_power build/symbolic_power build/quotient_ring: build/local_algebra
	ln -sf local_algebra $@

KERNEL_SOURCES := trace_levels.cpp groebner.cpp ideal.cpp ideal_arithmetic.cpp prepared_ideal.cpp modular.cpp modular_replay.cpp hilbert.cpp euler.cpp euler_lattice.cpp affine.cpp zeta.cpp \
  graph_replay.cpp graph_wide.cpp field_wide.cpp field_coefficient.cpp common_field.cpp field_factorization.cpp marked_scalar.cpp lattice_volume.cpp toric_graph.cpp fan_cover.cpp centre_weights.cpp centre_search.cpp centre_affine_maximum.cpp surface_centre_choice.cpp binomial_fan.cpp newton_repair.cpp repair_geometry.cpp \
  atlas_incidence.cpp graph_centre_incidence.cpp graph_regular.cpp torus_sections.cpp fan_surface.cpp owned_chart_incidence.cpp chart_gluing.cpp marked_chart_geometry.cpp ordered_ideals.cpp monomial_resolution.cpp component_projection.cpp component_geometry.cpp finite_sheet_incidence.cpp identity_divisor.cpp monomial_quotient.cpp rational_arithmetic.cpp \
  triton_kernels.cpp triton_rank.cpp triton_trace.cpp monodromy.cpp point_geometry.cpp \
  quotient_ring.cpp quotient_ideal.cpp quotient_packed.cpp quotient_inverse.cpp quotient_finite_inverse.cpp ideal_powers.cpp ideal_power_principal.cpp ideal_powers_packed.cpp ideal_powers_dense.cpp ideal_powers_echelon.cpp ideal_powers_content.cpp ideal_powers_monomial.cpp localization.cpp localization_membership.cpp localization_special.cpp normalization.cpp normalization_content.cpp normalization_special.cpp primary_decomposition.cpp primary_special.cpp polynomial_factors.cpp resolution_tangents.cpp \
  ae_codimension.cpp ae_special.cpp ae_graded.cpp ae_graded_surface.cpp ae_weighted_ranks.cpp ae_weighted_quotient.cpp ae_surface_coordinates.cpp \
  resolution_candidate.cpp smooth_point_order.cpp graph_verify_entry.cpp fan_verification.cpp local_operations.cpp local_basis.cpp local_division.cpp local_io.cpp local_power_coordinates.cpp local_dimension_graph.cpp \
  local_point_basis.cpp local_point_normal_form.cpp local_parameter_basis.cpp local_quotient_initial.cpp local_quotient_jets.cpp local_point_quotient.cpp
# The canonical blowup stage composes the operations above in one call.
KERNEL_SOURCES += canonical_blowup.cpp centre_maximum.cpp
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
test-verify: build/test_groebner_verify build/test_local_dimension_graph build/test_small_prime_rank build/test_ae_weighted_quotient build/test_chart_expr
	./build/test_groebner_verify
	./build/test_local_dimension_graph
	./build/test_small_prime_rank
	./build/test_ae_weighted_quotient
	./build/test_chart_expr

build/test_chart_expr: test_chart_expr.cpp chart_expr.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint jsoncpp) $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lflint -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp)

build/test_ae_weighted_quotient: test_ae_weighted_quotient.cpp ae_weighted_ranks.hpp ae_graded.hpp local_io.hpp $(GROEBNER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint jsoncpp) $< -o $@ $(LDFLAGS) -Lbuild $(LOCAL_RPATH) -lgroebner -lgmpxx -lgmp $(shell pkg-config --libs jsoncpp)

build/test_small_prime_rank: test_small_prime_rank.cpp small_prime_rank.hpp
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs flint)

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
