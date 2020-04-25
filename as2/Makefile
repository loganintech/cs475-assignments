CC = clang -O3 -fopenmp -lm main.c -o monte.o

clean:
	@rm monte.o

bench:
	@hyperfine  --show-output ./monte.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1

per_thread:
	for i in 12 11 10 9 8 7 6 5 4 3 2 1; do \
		$(CC) -DNUMT=$$i ;\
		./monte.o | rg "\{.*\}," -o >> thread_records.json ; \
	done

per_trial:
	for i in 100 1000 10000 100000 1000000 10000000; do \
		$(CC) -DNUMTRIALS=$$i -DNUMT=12;\
		./monte.o | rg "\{.*\}," -o >> trial_records.json ; \
	done
