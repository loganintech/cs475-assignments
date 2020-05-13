CC = clang -fopenmp -lm main.c -o simdspeed.o

e:
	@$(CC)

clean:
	@rm simdspeed.o

bench:
	@hyperfine  --show-output ./simdspeed.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1

per_size:
	@echo "[" > size_test.json
	@for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152; do \
		$(CC) -DARRAY_SIZE=$$i ;\
		./simdspeed.o >> size_test.json;\
	done
	@echo "]" >> size_test.json

per_thread:
	@echo "[" > thread_test.json
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
		$(CC) -DARRAY_SIZE=262144 -DNUMT=$$i ;\
		./simdspeed.o >> thread_test.json;\
	done
	@echo "]" >> thread_test.json

viz:
	python3 visualize.py tests.json