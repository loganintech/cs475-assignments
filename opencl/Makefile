CC = g++ src/cl.h src/cl_platform.h /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp

all: first second third

first:
	$(CC) src/first.cpp -o first.o

second:
	$(CC) src/second.cpp -o second.o

third:
	$(CC) src/third.cpp -o third.o

info:
	$(CC) src/printinfo.cpp -o printinfo

clean:
	@rm *.o
	@rm *results.txt

benchall: benchfirst benchsecond benchthird

benchfirst:
	@echo -e "NUM_ELEMENTS,NMB,LOCAL_SIZE,WORK_GROUPS,GIGAMULTS" > mul_results.txt
	@for i in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608; do \
		for x in 8 16 32 64 128 256 512; do \
			$(CC) -DNUM_ELEMENTS=$$i -DLOCAL_SIZE=$$x src/first.cpp -o first.o;\
			./first.o 2>> mul_results.txt; \
		done; \
	done


benchsecond:
	@echo -e "NUM_ELEMENTS,NMB,LOCAL_SIZE,WORK_GROUPS,GIGAMULTS" > mulsum_results.txt
	@for i in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608; do \
		for x in 8 16 32 64 128 256 512; do \
			$(CC) -DNUM_ELEMENTS=$$i -DLOCAL_SIZE=$$x src/second.cpp -o second.o;\
			./second.o 2>> mulsum_results.txt; \
		done; \
	done


benchthird:
	@echo -e "NUM_ELEMENTS,NMB,LOCAL_SIZE,WORK_GROUPS,GIGAMULTS" > mulreduce_results.txt
	@for i in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608; do \
		for x in 8 16 32 64 128 256 512; do \
			$(CC) -DNUM_ELEMENTS=$$i -DLOCAL_SIZE=$$x src/third.cpp -o third.o;\
			./third.o 2>> mulreduce_results.txt; \
		done; \
	done
