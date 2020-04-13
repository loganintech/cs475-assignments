CC = clang -O3 -fopenmp main.c -o opmbench.o

1:
	$(CC) -DNUMT=1

4:
	$(CC) -DNUMT=4

6:
	$(CC) -DNUMT=6

12:
	$(CC) -DNUMT=12

clean:
	@rm opmbench.o

bench:
	@hyperfine  --show-output ./opmbench.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1
