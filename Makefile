.PHONY: all ordinary plane chain toric newton groebner gpu test test-gpu bench clean
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
test: all
	python3 -m unittest discover -s tests -v
	python3 native/test_ordinary.py
	python3 native/test_chain.py
bench: all
	python3 benchmarks/resolve.py
clean:
	$(MAKE) -C native clean
