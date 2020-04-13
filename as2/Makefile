CC = clang -O3 -fopenmp main.c -o opmmonte

1:
	$(CC) -DNUMT=1

4:
	$(CC) -DNUMT=4

6:
	$(CC) -DNUMT=6

12:
	$(CC) -DNUMT=12

clean:
	@rm opmmonte

bench:
	@hyperfine  --show-output ./opmmonte 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1
