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
CUDA_LIBRARY_FLAGS=-L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcuda -lcudart

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

ifeq ($(funnel), on)
	OPT_FUNNEL=-DFUNNEL
endif

ifeq ($(shuffle), on)
	OPT_SHUFFLE=-DSHUFFLE
endif

ifeq ($(ballot), on)
	OPT_BALLOT=-DBALLOT
endif

######################
#  BUILD PARSERS
######################

gcandidates-GEM-1st:
	$(CC) $(CFLAGS) genCandidate-reorder.c -o gcandidates-GEM-1st

##################################################
#  MYERS CPU VERSIONS		  					 #
##################################################

myersCPU-padding-32bits:
	$(CC) $(CFLAGS) -o myersCPU-padding-32bits myers-4bits-padding-32bits.c $(LFLAGS) -lrt -fopenmp

myersCPU-GEM-1st:
	$(CC) $(CFLAGS) -o myersCPU-GEM-1st myers-reorderBuffer.c $(LFLAGS) -lrt -fopenmp

myersCPU-GEM-1st-Padding:
	$(CC) $(CFLAGS) -o myersCPU-GEM-1st-Padding myers-reorderBuffer-padding.c $(LFLAGS) -lrt -fopenmp

##################################################
#  1st STEP INTEGRATION		(reorder buffer)	 #
##################################################

sample-0:
	nvcc -O3 -m64 -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -c coreMyersGPU-CC20.cu
	nvcc -O3 -m64 -gencode arch=compute_30,code=sm_30 --ptxas-options=-v -c coreMyersGPU-CC30.cu
	nvcc -O3 -m64 -gencode arch=compute_35,code=sm_35 --ptxas-options=-v -c coreMyersGPU-CC35.cu
	nvcc -O3 -m64 -gencode arch=compute_50,code=sm_50 --ptxas-options=-v -c coreMyersGPU-CC50.cu
	gcc -O3 -m64 $(CUDA_LIBRARY_FLAGS) -c myersGPU.c
	gcc -O3 -m64 $(CUDA_LIBRARY_FLAGS) test.c coreMyersGPU-CC20.o coreMyersGPU-CC30.o coreMyersGPU-CC35.o coreMyersGPU-CC50.o myersGPU.o -o TEST -lrt -fopenmp
	rm coreMyersGPU-CC20.o coreMyersGPU-CC30.o coreMyersGPU-CC35.o coreMyersGPU-CC50.o myersGPU.o

sample-0-dbg:
	nvcc -G -O0 -m64 -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -c coreMyersGPU-CC20.cu
	nvcc -G -O0 -m64 -gencode arch=compute_30,code=sm_30 --ptxas-options=-v -c coreMyersGPU-CC30.cu
	nvcc -G -O0 -m64 -gencode arch=compute_35,code=sm_35 --ptxas-options=-v -c coreMyersGPU-CC35.cu
	nvcc -G -O0 -m64 -gencode arch=compute_50,code=sm_50 --ptxas-options=-v -c coreMyersGPU-CC50.cu
	gcc  -g -O0 -m64 $(CUDA_LIBRARY_FLAGS) -c myersGPU.c
	gcc  -g -O0 -m64 $(CUDA_LIBRARY_FLAGS) test.c coreMyersGPU-CC20.o coreMyersGPU-CC30.o coreMyersGPU-CC35.o coreMyersGPU-CC50.o myersGPU.o -o TEST-dbg -lrt -fopenmp
	rm coreMyersGPU-CC20.o coreMyersGPU-CC30.o coreMyersGPU-CC35.o coreMyersGPU-CC50.o myersGPU.o

######################
#  INSTALL
######################

install:
	find ./ -perm /a+x -type f -exec mv {} ../bin \;
