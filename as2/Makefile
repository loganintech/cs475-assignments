CC = clang -O3 -fopenmp -lm main.c -o monte.o -DNUMTRIES=1000

1:
	$(CC) -DNUMT=1

4:
	$(CC) -DNUMT=4

6:
	$(CC) -DNUMT=6

12:
	$(CC) -DNUMT=12

clean:
	@rm monte.o

bench:
	@hyperfine  --show-output ./monte.o 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1
