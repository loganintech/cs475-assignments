CC = g++ -O3 src/main.cpp src/cl.h /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -lm -fopenmp src/cl_platform.h -o signal.o

clean:
	-@rm *.o
	-@rm data.txt
	-@rm performance.txt
	-@rm *.png

main:
	@$(CC)

data: 
	@$(CC) -D_DATA
	@./signal.o > data.txt

perf:
	@$(CC) -D_PERF
	@./signal.o > performance.txt

display_data:
	@python3 src/visualize_data.py

display_perf:
	@python3 src/visualize_perf.py

do_everything: perf data display_perf display_data