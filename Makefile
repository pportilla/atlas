.PHONY: all ordinary plane chain toric newton groebner gpu test check-native-boundary test-gpu test-triton bench clean
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
	python3 native/test_opencl.py
	python3 native/test_chart_batch.py
	python3 -m unittest discover -s tests -p 'test_gpu_trace_rows.py'
	python3 -m unittest discover -s tests -p 'test_gpu_normal_form.py'
	python3 -m unittest discover -s tests -p 'test_replay_scope.py'
	python3 -m unittest discover -s tests -p 'test_gpu_resolution.py'
test-triton: all
	python3 -c 'import sys; from engine.gpu_replay import triton_available; sys.exit(0 if triton_available() else "No usable physical Triton device at the selected index")'
	python3 native/test_chart_batch.py
	python3 -m unittest discover -s tests -p 'test_gpu_trace_rows.py'
	python3 -m unittest discover -s tests -p 'test_gpu_normal_form.py'
	python3 -m unittest discover -s tests -p 'test_gpu_rank.py'
	python3 -m unittest discover -s tests -p 'test_gpu_replay.py'
	python3 -m unittest discover -s tests -p 'test_gpu_resolution.py'
check-native-boundary:
	python3 -m unittest tests.test_native_math_boundary.NativeMathBoundaryTests
test: all
	$(MAKE) -C native -f groebner.mk test-verify
	python3 -m unittest discover -s tests -v
	python3 native/test_ordinary.py
	python3 native/test_chain.py
bench: all
	python3 benchmarks/resolve.py
clean:
	$(MAKE) -C native clean
