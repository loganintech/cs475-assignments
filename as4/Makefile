EXE_NAME = simulate.o
CC = clang -fopenmp -lm main.c -o $(EXE_NAME)

e:
	@$(CC)

clean:
	@rm $(EXE_NAME)
	@rm visualized.png

sim: e
	@echo "[" > simulated.json
	@./$(EXE_NAME) >> simulated.json
	@echo "]" >> simulated.json

visualize: sim
	@python3 visualize.py simulated.json



# bench:
# 	@hyperfine  --show-output ./$(EXE_NAME) 2>/dev/null | rg "([0-9]+\.[0-9]{2})" -o | sort -rn | head -n 1

# per_thread:
# 	for i in 12 11 10 9 8 7 6 5 4 3 2 1; do \
# 		$(CC) -DNUMT=$$i ;\
# 		./$(EXE_NAME) | rg "\{.*\}," -o >> thread_records.json; \
# 	done

# per_nodes:
# 	for i in 500 1000 1500 2000 2500 3000 3500 4000; do \
# 		$(CC) -DNUMNODES=$$i -DNUMT=12;\
# 		./$(EXE_NAME) | rg "\{.*\}," -o >> node_records.json;\
# 	done
