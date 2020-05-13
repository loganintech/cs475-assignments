CC = g++ -fopenmp -lm main.c -o simdspeed.o

e:
	@$(CC)

clean:
	@rm simdspeed.o

bench:
	@hyperfine  --show-output ./simdspeed.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1

per_size:
	for i in 1024 2048 51200 5242880 8388608; do \
		$(CC) -DARRAY_SIZE=$$i ;\
		./simdspeed.o;\
	done
