# COMPILATION EXAMPLES
# --------------------
# cd src; make myersCPU; make install; cd ..;
# cd src; make myersGPU arch=kepler line=ca th_block=128 th_sm=128; make install; cd ..;
#
# DEBUG:
# ---------------------
# source /opt/intel/composer_xe_2013/bin/compilervars.sh intel64;
# cd src; make myersCPU; make install; cd ..;
#
# EXECUTION:
# ---------------------
# bin/greference input/human_g1k_v37.cleaned.fasta
# bin/gcandidates input/regions-score.1M.prof
# bin/myers 0.2 human_g1k_v37.cleaned.fasta.3101804739.bin.ref  regions-score.1M.prof.1000000.20.100.bin.qry
# bin/myersGPU-kepler 0.2 input/human_g1k_v37.cleaned.fasta.3101804739.bin.ref input/regions-score.1M.prof.1000000.20.100.bin.qry
# bin/checker 0.2 input/regions-score.1M.prof.1000000.20.100.bin.qry input/regions-score.1M.prof.1000000.20.100.bin.qry.res.cpu
# bin/swmatrix ANNEALING ANNUAL
# bin/reference human_g1k_v37.MAY.fasta
# bin/greference-4bits input/human_g1k_v37.cleaned.fasta

# cd src; make myers4bitsGPU-col arch=titan sizeQuery=1000 funnel=off shuffle=off ballot=off word=32 line=ca th_block=128 th_sm=128; make install; cd ..;

# Shell interpreter #
#####################
SHELL := /bin/bash

# Compilers  #
##############
NVCC=nvcc
CC=gcc
ICC=icc

# CPU flags  #
##############
CFLAGS=-O3 -m64
LFLAGS=

# CUDA flags #
##############
CUDA_OPT_FLAGS=-O3 -m64
CUDA_LIBRARY_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart 

# DEBUG FLAGS #
###############
GCC_GDB_FLAGS=-g -O0 -m64
NVCC_GDB_FLAGS=-g -G -O0 -m64

ifeq ($(arch),tesla)
	NVCFLAGS=-gencode arch=compute_10,code=sm_10 --ptxas-options=-v
	CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
	CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),fermi)
	NVCFLAGS=-gencode arch=compute_20,code=sm_20 --ptxas-options=-v
	CUDA_DEVICE=-DDEVICE=1
	CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
	CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),kepler)
	NVCFLAGS=-gencode arch=compute_30,code=sm_30 --ptxas-options=-v
	CUDA_DEVICE=-DDEVICE=0
	CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
	CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),titan)
	NVCFLAGS=-gencode arch=compute_35,code=sm_35 --ptxas-options=-v
	CUDA_DEVICE=-DDEVICE=0
	CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
	CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

# Conf save results #
#####################
MEMORY_WRITES=-DMEMORY_WRITES=1
ifeq ($(gstores), off)
	MEMORY_WRITES=-DMEMORY_WRITES=0
endif

ifeq ($(funnel), on)
	OPT_FUNNEL=-DFUNNEL
endif

ifeq ($(shuffle), on)
	OPT_SHUFFLE=-DSHUFFLE
endif

ifeq ($(ballot), on)
	OPT_BALLOT=-DBALLOT
endif

SIZEQUERY=-DSIZE_QUERY=$(sizeQuery)

NUMBER=$(shell echo "($(sizeQuery) - 1) % $(word)" | bc)
RESULT=$(shell [[ ($(NUMBER) -ge   "0") && ($(NUMBER) -le  "31") ]] && echo "0"; \
			   [[ ($(NUMBER) -ge  "32") && ($(NUMBER) -le  "63") ]] && echo "1"; \
			   [[ ($(NUMBER) -ge  "64") && ($(NUMBER) -le  "95") ]] && echo "2"; \
			   [[ ($(NUMBER) -ge  "96") && ($(NUMBER) -le "127") ]] && echo "3"; \
			   [[ ($(NUMBER) -ge "128") && ($(NUMBER) -le "159") ]] && echo "4"; \
			   [[ ($(NUMBER) -ge "160") && ($(NUMBER) -le "191") ]] && echo "5"; \
			   [[ ($(NUMBER) -ge "192") && ($(NUMBER) -le "223") ]] && echo "6"; \
			   [[ ($(NUMBER) -ge "224") && ($(NUMBER) -le "255") ]] && echo "7";)
UNROLL=-DREG=$(RESULT)

######################
#  BUILD PARSERS
######################

greference:
	$(CC) $(CFLAGS) genReference.c -o greference
greference-dbg:
	$(CC) $(GCC_GDB_FLAGS) genReference.c -o greference-dbg

gcandidates:
	$(CC) $(CFLAGS) genCandidates.c -o gcandidates
gcandidates-dbg:
	$(CC) $(GCC_GDB_FLAGS) genCandidates.c -o gcandidates-dbg

greference-4bits:
	$(CC) $(CFLAGS) genReference-4bits.c -o greference-4bits

######################
#  BUILD PARSERS
######################

checker:
	$(CC) $(CFLAGS) checksum.c -o checker
checker-dbg:
	$(CC) $(GCC_GDB_FLAGS) checksum.c -o checker


checker-2:
	$(CC) $(CFLAGS) checksum-2.c -o checker-2


swmatrix:
	$(CC) $(CFLAGS) -o swmatrix sw.c $(LFLAGS)
swmatrix-dbg:
	$(CC) $(GCC_GDB_FLAGS) -o swmatrix-dbg sw.c $(LFLAGS)

##################################################
#  SMITH & WATERMAN CPU VERSIONS				 # 
##################################################

swCPU:
	$(CC) $(CFLAGS) sw.c -o swCPU -lrt $(LFLAGS)
swCPU-dbg:
	$(CC) $(GCC_GDB_FLAGS) sw.c -o swCPU -lrt $(LFLAGS)

##################################################
#  MYERS CPU VERSIONS		  					 # 
##################################################

myersCPU:
	$(CC) $(CFLAGS) -o myersCPU myers.c $(LFLAGS) -lrt -fopenmp

myers4bitsCPU:
	$(CC) $(CFLAGS) -o myers4bitsCPU myers-4bits.c $(LFLAGS) -lrt -fopenmp

myers4bitsCPU-64:
	$(CC) $(CFLAGS) -o myers4bitsCPU-64 myers-4bits-64bits.c $(LFLAGS) -lrt -fopenmp

myers4bitsCPU-padding:
	$(CC) $(CFLAGS) -o myers4bitsCPU-padding myers-4bits-padding.c $(LFLAGS) -lrt -fopenmp


##################################################
#  MYERS GPU VERSIONS		 					 # 
##################################################

myersGPU:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myersGPU-$(arch) myers.c myersGPU.o -lrt
	rm myersGPU.o

myers4bitsGPU:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-$(arch) myers-4bits.c myersGPU-4bits.o -lrt
	rm myersGPU-4bits.o

myersGPU-shared:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-shared.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myersGPU-shared-$(arch) myers.c myersGPU-shared.o -lrt
	rm myersGPU-shared.o

myers4bitsGPU-shared:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-shared.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-shared-$(arch) myers-4bits.c myersGPU-4bits-shared.o -lrt
	rm myersGPU-4bits-shared.o

myers4bitsGPU-all-shared:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-all-shared.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-all-shared-$(arch) myers-4bits.c myersGPU-4bits-all-shared.o -lrt
	rm myersGPU-4bits-all-shared.o

myers4bitsGPU-col:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "$(SIZEQUERY) $(OPT_FUNNEL) $(OPT_SHUFFLE) $(OPT_BALLOT) $(UNROLL) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-col.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-col-$(arch) myers-4bits.c myersGPU-4bits-col.o -lrt
	rm myersGPU-4bits-col.o

myers4bitsGPU-col-64:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -lineinfo -Xcompiler "$(SIZEQUERY) $(OPT_FUNNEL) $(OPT_SHUFFLE) $(OPT_BALLOT) $(UNROLL) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-col-64.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-col-64-$(arch) myers-4bits.c myersGPU-4bits-col-64.o -lrt
	rm myersGPU-4bits-col-64.o

myers4bitsGPU-col-128:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -lineinfo -Xcompiler "$(SIZEQUERY) $(OPT_FUNNEL) $(OPT_SHUFFLE) $(OPT_BALLOT) $(UNROLL) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-col-128.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-col-128-$(arch) myers-4bits.c myersGPU-4bits-col-128.o -lrt
	rm myersGPU-4bits-col-128.o

myers4bitsGPU-col-256:
	$(NVCC) $(CUDA_OPT_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -lineinfo -Xcompiler "$(SIZEQUERY) $(OPT_FUNNEL) $(OPT_SHUFFLE) $(OPT_BALLOT) $(UNROLL) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-4bits-col-256.cu
	$(CC) $(CUDA_OPT_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myers4bitsGPU-col-256-$(arch) myers-4bits.c myersGPU-4bits-col-256.o -lrt
	rm myersGPU-4bits-col-256.o

##################################################
#  MYERS DEBUG GPU VERSIONS						 # 
##################################################

myersGPU-dbg:
	$(NVCC) $(NVCC_GDB_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU.cu
	$(CC) $(GCC_GDB_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myersGPU-$(arch)-dbg myers.c myersGPU.o -lrt
	rm myersGPU.o

myersGPU-shared-dbg:
	$(NVCC) $(NVCC_GDB_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-shared.cu
	$(CC) $(GCC_GDB_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myersGPU-shared-$(arch) myers.c myersGPU-shared.o -lrt
	rm myersGPU-shared.o

myersGPU-col-dbg:
	$(NVCC) $(NVCC_GDB_FLAGS) $(NVCFLAGS) -Xptxas -dlcm=$(line) -Xcompiler "-DN_ENTRIES=$(neq) -DCUDA $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c myersGPU-col.cu
	$(CC) $(GCC_GDB_FLAGS) -msse4.2 -DCUDA $(CUDA_LIBRARY_FLAGS) -o myersGPU-col-dbg-$(arch) myers.c myersGPU-col.o -lrt
	rm myersGPU-col.o

######################
#  INSTALL
######################

install:
	find ./ -perm /a+x -type f -exec mv {} ../bin \;

