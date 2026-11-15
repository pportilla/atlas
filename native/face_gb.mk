CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
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
all: build/face_gb build/libface_gb.$(HELPER_SUFFIX)
build/libface_gb.$(HELPER_SUFFIX): face_gb.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) $(HELPER_FLAGS) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs flint)
build/face_gb: face_gb.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(shell pkg-config --cflags flint) $< -o $@ $(LDFLAGS) $(shell pkg-config --libs flint)
