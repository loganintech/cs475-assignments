CC = clang -O3 -fopenmp -lm main.c -o superquad.o

e:
	@$(CC)

clean:
	@rm superquad.o

bench:
	@hyperfine  --show-output ./superquad.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1

per_thread:
	for i in 12 11 10 9 8 7 6 5 4 3 2 1; do \
		$(CC) -DNUMT=$$i ;\
		./superquad.o; \
	done

per_nodes:
	for i in 500 1000 1500 2000 2500 3000 3500 4000; do \
		$(CC) -DNUMNODES=$$i -DNUMT=12;\
		./superquad.o; \
	done
