/*
 * myersGPU-col.cu
 *
 *  Created on: 27/10/2013
 *      Author: achacon
 */


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define		NUM_BITS		4
#define     NUM_BASES       5
#define     SIZE_HW_WORD    32
#define     MAX_VALUE       0xFFFFFFFF
#define     HIGH_MASK_32    0x80000000
#define     LOW_MASK_32     0x00000001

#define     SIZE_WARP		32
 #define 	BASES_X_ENTRY	8



#define HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif

// add & output temporal carry in internal register
#define UADD__CARRY_OUT(c, a, b) \
     asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b));
// add with temporal carry of internal register
#define UADD__IN_CARRY(c, a, b) \
     asm volatile("addc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b));


typedef struct {
	uint32_t bitmap[NUM_BASES];
} qryEntry_t;

typedef struct {
	uint32_t column;
	uint32_t score;
} resEntry_t;

typedef struct {
	uint32_t query;
	uint32_t position;
} candInfo_t;


typedef struct {
	uint32_t size;
	uint32_t numEntries;
	uint32_t *h_reference;
	uint32_t *d_reference;
} ref_t;

typedef struct {
	uint32_t numResults;
	resEntry_t* h_results;
	resEntry_t* d_results;
} res_t;

typedef struct {
	uint32_t totalSizeQueries;
	uint32_t totalQueriesEntries;
	uint32_t sizeQueries;
	uint32_t numQueries;
	uint32_t numCandidates;
	float 	 distance;

	qryEntry_t *h_queries;
	qryEntry_t *d_queries;
	candInfo_t *h_candidates;
	candInfo_t *d_candidates;
	uint32_t *d_Pv;
	uint32_t *d_Mv;
} qry_t;

extern "C"
static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
      		printf( "%s in %s at line %d\n", cudaGetErrorString(err),  file, line );
       		exit( EXIT_FAILURE );
   	}
}

inline __device__ uint32_t collaborative_sum(uint32_t a, uint32_t b, uint32_t localThreadIdx)
{

	uint32_t carry, c;

	UADD__CARRY_OUT(c, a, b)
	UADD__IN_CARRY(carry, 0, 0)

	while(__any(carry)){
		carry = __shfl_up((int) (carry), 1);
		//TODO: condición de frontera entre queries
		carry = (localThreadIdx == 0) ? 0 : carry;
		UADD__CARRY_OUT(c, c, carry)
		UADD__IN_CARRY(carry, 0, 0) // save carry-out
	}

	return c;
}

inline __device__ uint32_t collaborative_shift(uint32_t value, uint32_t localThreadIdx)
{
	uint32_t carry;

	carry = __shfl_up((int) (value >> 31), 1);
	//TODO: condición de frontera entre queries
	carry = (localThreadIdx == 0) ? 0 : carry;
	carry = (value << 1) | carry;
	return (carry);
}

inline __device__ uint32_t selectEq(uint32_t indexBase, uint32_t Eq0, uint32_t Eq1, uint32_t Eq2, uint32_t Eq3, uint32_t Eq4)
{
	uint32_t Eq = Eq0;

	Eq = (indexBase == 1) ? Eq1 : Eq;
	Eq = (indexBase == 2) ? Eq2 : Eq;
	Eq = (indexBase == 3) ? Eq3 : Eq;
	Eq = (indexBase == 4) ? Eq4 : Eq;

	return Eq;
}

__global__ void myersKernel(qryEntry_t *d_queries, uint32_t *d_reference, candInfo_t *d_candidates, resEntry_t *d_results,
							uint32_t sizeCandidate, uint32_t sizeQueries, uint32_t sizeRef, uint32_t numEntriesPerQuery,
							uint32_t numCandidates)
{
	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;
	uint32_t Eq0, Eq1, Eq2, Eq3, Eq4;
	uint32_t candidate;
	uint32_t sum, entry, idColumn, indexBase, aline;

	uint32_t globalThreadIdx = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;
	uint32_t localThreadIdx = threadIdx.x % SIZE_WARP;
	uint32_t idCandidate = globalThreadIdx / SIZE_WARP;

	if ((threadIdx.x < MAX_THREADS_PER_SM) && (idCandidate < numCandidates)){

		uint32_t positionRef = d_candidates[idCandidate].position;
		uint32_t entryRef = positionRef / BASES_X_ENTRY;
		int32_t score = sizeQueries, minScore = sizeQueries;
		uint32_t minColumn = 0;
		uint32_t mask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
		uint32_t word = 0;

		if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

			//Init variables
			Pv = MAX_VALUE;
			Mv = 0;
			entry = (d_candidates[idCandidate].query * numEntriesPerQuery) + localThreadIdx;
			Eq0 = d_queries[entry].bitmap[0];
			Eq1 = d_queries[entry].bitmap[1];
			Eq2 = d_queries[entry].bitmap[2];
			Eq3 = d_queries[entry].bitmap[3];
			Eq4 = d_queries[entry].bitmap[4];

			for(idColumn = 0; idColumn < sizeCandidate; idColumn++){

				//Read the next candidate letter (column)
				aline = (positionRef % BASES_X_ENTRY);
				if((aline == 0) || (idColumn == 0)) {
						candidate = d_reference[entryRef + word] >>  (aline * NUM_BITS); 
						word++;
				}

				indexBase = candidate & 0x07;

				////////   complete column - MYERS    ///////
				Eq = selectEq(indexBase, Eq0, Eq1, Eq2, Eq3, Eq4);
				Xv = Eq | Mv;
				sum = collaborative_sum(Eq & Pv, Pv, localThreadIdx);
				Xh = (sum ^ Pv) | Eq;
				Ph = Mv | ~(Xh | Pv);
				Mh = Pv & Xh;

				score += ((Ph & mask) != 0) - ((Mh & mask) != 0);

				Ph = collaborative_shift(Ph, localThreadIdx);
				Mh = collaborative_shift(Mh, localThreadIdx);
				Pv = Mh | ~(Xv | Ph);
				Mv = Ph & Xv;
				////////   end - MYERS    ///////

				candidate >>= 4;
				positionRef++;

				minColumn = (score < minScore) ? idColumn : minColumn;
				minScore  = (score < minScore) ? score : minScore;
			}

			if(localThreadIdx == 31){
	    		d_results[idCandidate].column = minColumn;
	    		d_results[idCandidate].score = minScore;
			}
		}
	}
}

extern "C"
void computeAllQueriesGPU(void *reference, void *queries, void *results)
{

	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);

	uint32_t blocks = ((qry->numCandidates * numEntriesPerQuery) / MAX_THREADS_PER_SM) + (((qry->numCandidates * numEntriesPerQuery) % MAX_THREADS_PER_SM) ? 1 : 0);
	uint32_t threads = CUDA_NUM_THREADS;

	printf("-- Bloques: %d - Th_block %d - Th_sm %d\n", blocks, threads, MAX_THREADS_PER_SM);

	myersKernel<<<blocks,threads>>>(qry->d_queries, ref->d_reference, qry->d_candidates, res->d_results,
		 			  sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, qry->numCandidates);

	cudaThreadSynchronize();
}

extern "C"
int transferCPUtoGPU(void *reference, void *queries, void *results)
{
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

	//uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);

    HANDLE_ERROR(cudaSetDevice(DEVICE));

	//allocate & transfer Binary Reference to GPU
	HANDLE_ERROR(cudaMalloc((void**) &ref->d_reference, ((uint64_t) ref->numEntries) * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(ref->d_reference, ref->h_reference, ((uint64_t) ref->numEntries) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//allocate & transfer Binary Queries to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_queries, qry->totalQueriesEntries * sizeof(qryEntry_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_queries, qry->h_queries, qry->totalQueriesEntries * sizeof(qryEntry_t), cudaMemcpyHostToDevice));

	//allocate & transfer Candidates to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_candidates, qry->numCandidates * sizeof(candInfo_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_candidates, qry->h_candidates, qry->numCandidates * sizeof(candInfo_t), cudaMemcpyHostToDevice));

	//allocate Results
	HANDLE_ERROR(cudaMalloc((void**) &res->d_results, res->numResults * sizeof(resEntry_t)));
 	HANDLE_ERROR(cudaMemset(res->d_results, 0, res->numResults * sizeof(resEntry_t)));

	return (0);
}

extern "C"
int transferGPUtoCPU(void *results)
{
	res_t *res = (res_t *) results;

	HANDLE_ERROR(cudaMemcpy(res->h_results, res->d_results, res->numResults * sizeof(resEntry_t), cudaMemcpyDeviceToHost));

	return (0);
}

extern "C"
int freeReferenceGPU(void *reference)
{
	ref_t *ref = (ref_t *) reference;

	if(ref->d_reference != NULL){
		cudaFree(ref->d_reference);
		ref->d_reference=NULL;
	}

	return(0);
}

extern "C"
int freeQueriesGPU(void *queries)
{
	qry_t *qry = (qry_t *) queries;

	if(qry->d_queries != NULL){
		cudaFree(qry->d_queries);
		qry->d_queries=NULL;
	}

	if(qry->d_candidates != NULL){
        cudaFree(qry->d_candidates);
        qry->d_candidates = NULL;
    }

	return(0);
}

extern "C"
int freeResultsGPU(void *results)
{
	res_t *res = (res_t *) results;

	if(res->d_results != NULL){
		cudaFree(res->d_results);
		res->d_results=NULL;
	}

	return(0);
}
