CC = clang -O3 -fopenmp main.c -o opmbench

1:
	$(CC) -DNUMT=1

4:
	$(CC) -DNUMT=4

6:
	$(CC) -DNUMT=6

12:
	$(CC) -DNUMT=12

clean:
	@rm opmbench

bench:
	@hyperfine  --show-output ./opmbench 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1
