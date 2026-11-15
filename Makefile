.PHONY: all ordinary plane chain toric newton groebner gpu test check-native-boundary test-gpu test-triton bench docs clean
all: ordinary plane chain toric newton groebner
ordinary:
	$(MAKE) -C native
plane:
	$(MAKE) -C native -f plane.mk
chain:
	$(MAKE) -C native -f chain.mk
toric:
	$(MAKE) -C native -f toric.mk
newton:
	$(MAKE) -C native -f face_gb.mk
	$(MAKE) -C native -f newton_fan.mk
groebner:
	$(MAKE) -C native -f groebner.mk
gpu: all
	$(MAKE) -C native gpu
	$(MAKE) -C native -f groebner.mk gpu
test-gpu: gpu
	python3 -m unittest tests.test_native_opencl tests.test_native_chart_batch \
		tests.test_gpu_trace_rows tests.test_gpu_normal_form tests.test_replay_scope tests.test_gpu_resolution
test-triton: all
	python3 -c 'import sys; from engine.gpu_replay import triton_available; sys.exit(0 if triton_available() else "No usable physical Triton device at the selected index")'
	python3 -m unittest tests.test_native_chart_batch tests.test_gpu_trace_rows tests.test_gpu_normal_form \
		tests.test_gpu_rank tests.test_gpu_replay tests.test_gpu_resolution
check-native-boundary:
	python3 -m unittest tests.test_native_math_boundary.NativeMathBoundaryTests
test: all
	$(MAKE) -C native -f groebner.mk test-verify
	python3 -m unittest discover -s tests -v
bench: all
	python3 benchmarks/resolve.py
docs:
	python3 docs/site/build.py --out build/docs
clean:
	$(MAKE) -C native clean
