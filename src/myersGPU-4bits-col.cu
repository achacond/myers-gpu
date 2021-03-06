/*
 * myersGPU-col.cu
 *
 *  Created on: 25/11/2013
 *      Author: achacon
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define	    NUM_BITS				4
#define     NUM_BASES       		5
#define     SIZE_HW_WORD    		32
#define     MAX_VALUE       		0xFFFFFFFF
#define     HIGH_MASK_32    		0x80000000
#define     LOW_MASK_32     		0x00000001

#define		SIZE_WARP				32
#define 	BASES_PER_ENTRY			8
#define		BASES_PER_THREAD		32
#define		ENTRIES_PER_THREAD		(BASES_PER_THREAD / SIZE_HW_WORD)

#define		SPACE_PER_QUERY		((((SIZE_QUERY-1)/BASES_PER_THREAD)+1) * BASES_PER_THREAD)
#define		BASES_PER_WARP		(SIZE_WARP * BASES_PER_THREAD)
#define 	QUERIES_PER_WARP	(BASES_PER_WARP / SPACE_PER_QUERY)
#define 	THREADS_PER_QUERY	(SPACE_PER_QUERY / BASES_PER_THREAD)
#define		WARP_THREADS_IDLE	(SIZE_WARP - (THREADS_PER_QUERY * QUERIES_PER_WARP))
#define		WARP_THREADS_ACTIVE	(THREADS_PER_QUERY * QUERIES_PER_WARP)

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

#if defined(SHUFFLE) || defined(BALLOT)
#else
inline __device__ uint32_t shared_collaborative_sum(const uint32_t a, const uint32_t b, const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff)
{

	uint32_t carry, c;

	UADD__CARRY_OUT(c, a, b)
	UADD__IN_CARRY(carry, 0, 0)

	interBuff[intraWarpIdx + 1] = carry;
	carry = interBuff[intraWarpIdx];
	carry = (localThreadIdx == 0) ? 0 : carry;

	UADD__CARRY_OUT(c, c, carry)
	UADD__IN_CARRY(carry, 0, 0)

	while(__any(carry)){
		interBuff[intraWarpIdx + 1] = carry;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx == 0) ? 0 : carry;
		UADD__CARRY_OUT(c, c, carry)
		UADD__IN_CARRY(carry, 0, 0)
	}

	return c;
}

inline __device__ uint32_t shared_collaborative_shift(const uint32_t value, const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff)
{
	uint32_t carry;
	#ifdef FUNNEL
		interBuff[intraWarpIdx + 1] = value;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx == 0) ? 0 : carry;
		carry = __funnelshift_lc(carry, value, 1);
	#else
		interBuff[intraWarpIdx + 1] = value >> 31;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx == 0) ? 0 : carry;
		carry = (value << 1) | carry;
	#endif
	return (carry);

}
#endif

#ifdef SHUFFLE
inline __device__ uint32_t shuffle_collaborative_shift(const uint32_t value, const uint32_t localThreadIdx)
{
	uint32_t carry;
	#ifdef FUNNEL
		carry = __shfl_up((int) value, 1);
		carry = (localThreadIdx == 0) ? 0 : carry;
		carry = __funnelshift_lc(carry, value, 1);
	#else
		carry = __shfl_up((int) (value >> 31), 1);
		carry = (localThreadIdx == 0) ? 0 : carry;
		carry = (value << 1) | carry;
	#endif
	return (carry);
}

inline __device__ uint32_t shuffle_collaborative_sum(const uint32_t a, const uint32_t b, const uint32_t localThreadIdx)
{

	uint32_t carry, c;

	UADD__CARRY_OUT(c, a, b)
	UADD__IN_CARRY(carry, 0, 0)

	carry = __shfl_up((int) (carry), 1);
	carry = (localThreadIdx == 0) ? 0 : carry;

	UADD__CARRY_OUT(c, c, carry)
	UADD__IN_CARRY(carry, 0, 0)

	while(__any(carry)){
		carry = __shfl_up((int) (carry), 1);
		carry = (localThreadIdx == 0) ? 0 : carry;
		UADD__CARRY_OUT(c, c, carry)
		UADD__IN_CARRY(carry, 0, 0)
	}

	return c;
}
#endif

#ifdef BALLOT
inline __device__ uint32_t ballot_collaborative_shift(const uint32_t value, const uint32_t localThreadIdx, const uint32_t intraWarpIdx)
{
	uint32_t carry;
		carry = ((__ballot(value >> 31) << 1) & (1 << intraWarpIdx)) != 0;
		carry = (localThreadIdx == 0) ? 0 : carry;
		carry = (value << 1) | carry;
	return (carry);

}

inline __device__ uint32_t ballot_collaborative_sum(const uint32_t a, const uint32_t b, const uint32_t localThreadIdx, const uint32_t intraWarpIdx)
{

	uint32_t carry, c;

	UADD__CARRY_OUT(c, a, b)
	UADD__IN_CARRY(carry, 0, 0)

	carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
	carry = (localThreadIdx == 0) ? 0 : carry;

	UADD__CARRY_OUT(c, c, carry)
	UADD__IN_CARRY(carry, 0, 0)

	while(__any(carry)){
		carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
		carry = (localThreadIdx == 0) ? 0 : carry;
		UADD__CARRY_OUT(c, c, carry)
		UADD__IN_CARRY(carry, 0, 0)
	}

	return c;
}
#endif

inline __device__ uint32_t selectEq(const uint32_t indexBase, 
				    const uint32_t Eq0, const uint32_t Eq1, 
				    const uint32_t Eq2, const uint32_t Eq3, 
				    const uint32_t Eq4)
{
	uint32_t Eq = Eq0;

	Eq = (indexBase == 1) ? Eq1 : Eq;
	Eq = (indexBase == 2) ? Eq2 : Eq;
	Eq = (indexBase == 3) ? Eq3 : Eq;
	Eq = (indexBase == 4) ? Eq4 : Eq;

	return Eq;
}

__global__ void myersKernel(const qryEntry_t *d_queries, const uint32_t __restrict *d_reference, const candInfo_t *d_candidates, resEntry_t *d_results,
			    const uint32_t sizeCandidate, const uint32_t sizeQueries, const uint32_t sizeRef, const uint32_t numEntriesPerQuery,
			    const uint32_t numEntriesPerCandidate, const uint32_t numCandidates, const uint32_t numThreads)
{

	const uint32_t __restrict * localCandidate;

	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;

	uint32_t Eq0, Eq1, Eq2, Eq3, Eq4;

	uint32_t sum;

	uint32_t candidate;
	uint32_t entry, idColumn = 0, indexBase;

	uint32_t globalThreadIdx = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;
	uint32_t intraQueryThreadIdx = (threadIdx.x % SIZE_WARP) % THREADS_PER_QUERY;
	uint32_t idCandidate = ((globalThreadIdx / SIZE_WARP) * QUERIES_PER_WARP) + ((threadIdx.x % SIZE_WARP) / THREADS_PER_QUERY);

	#if defined(SHUFFLE) || defined(BALLOT)
	#else
		__shared__
		uint32_t globalInterBuff[(SIZE_WARP + 1) * (CUDA_NUM_THREADS/SIZE_WARP)];
		uint32_t *localInterBuff = globalInterBuff + ((threadIdx.x/SIZE_WARP) * (SIZE_WARP + 1));
	#endif

	#ifndef SHUFFLE
		uint32_t intraWarpIdx = threadIdx.x % SIZE_WARP;
	#endif

	if ((threadIdx.x < MAX_THREADS_PER_SM) && (idCandidate < numCandidates)){

		uint32_t positionRef = d_candidates[idCandidate].position;
		uint32_t entryRef = positionRef / BASES_PER_ENTRY;
		int32_t score = sizeQueries, minScore = sizeQueries;
		uint32_t minColumn = 0;
		uint32_t mask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
		uint32_t intraBase, idEntry;

		if((positionRef < sizeRef) && ((sizeRef - positionRef) > sizeCandidate)){

			localCandidate = d_reference + entryRef;

			Pv = MAX_VALUE;
			Mv = 0;
			entry = (d_candidates[idCandidate].query * numEntriesPerQuery) + intraQueryThreadIdx;

			Eq0 = d_queries[entry].bitmap[0];
			Eq1 = d_queries[entry].bitmap[1];
			Eq2 = d_queries[entry].bitmap[2];
			Eq3 = d_queries[entry].bitmap[3];
			Eq4 = d_queries[entry].bitmap[4];

			for(idEntry = 0; idEntry < numEntriesPerCandidate; idEntry++){

				candidate = localCandidate[idEntry]; 

				for(intraBase = 0; intraBase < BASES_PER_ENTRY; intraBase++){	
					
					indexBase = candidate & 0x07;
					Eq = selectEq(indexBase, Eq0, Eq1, Eq2, Eq3, Eq4);
					Xv = Eq | Mv;

					#ifdef SHUFFLE
						sum = shuffle_collaborative_sum(Eq & Pv, Pv, intraQueryThreadIdx);
					#else
						#ifdef BALLOT
							sum = ballot_collaborative_sum(Eq & Pv, Pv, intraQueryThreadIdx, intraWarpIdx);
						#else
							sum = shared_collaborative_sum(Eq & Pv, Pv, intraQueryThreadIdx, intraWarpIdx, localInterBuff);
						#endif
					#endif

					Xh = (sum ^ Pv) | Eq;
					Ph = Mv | ~(Xh | Pv);
					Mh = Pv & Xh;

					score += ((Ph & mask) != 0) - ((Mh & mask) != 0);

					#ifdef SHUFFLE
							Ph = shuffle_collaborative_shift(Ph, intraQueryThreadIdx);
							Mh = shuffle_collaborative_shift(Mh, intraQueryThreadIdx);
					#else
						#ifdef BALLOT
							Ph = ballot_collaborative_shift(Ph, intraQueryThreadIdx, intraWarpIdx);
							Mh = ballot_collaborative_shift(Mh, intraQueryThreadIdx, intraWarpIdx);
						#else
							Ph = shared_collaborative_shift(Ph, intraQueryThreadIdx, intraWarpIdx, localInterBuff);
							Mh = shared_collaborative_shift(Mh, intraQueryThreadIdx, intraWarpIdx, localInterBuff);
						#endif
					#endif

					Pv = Mh | ~(Xv | Ph);
					Mv = Ph & Xv;

					candidate >>= 4;
					minColumn = (score < minScore) ? idColumn : minColumn;
					minScore  = (score < minScore) ? score : minScore;
					idColumn++;
				}
			}

			if(intraQueryThreadIdx  == (THREADS_PER_QUERY - 1)){
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

	uint32_t blocksPerGrid, threadsPerBlock = MAX_THREADS_PER_SM;
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);
	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);
	uint32_t numEntriesPerCandidate = (sizeCandidate / BASES_PER_ENTRY) + ((sizeCandidate % BASES_PER_ENTRY) ? 2 : 1);

	uint32_t maxCandidates, numCandidates, lastCandidates, processedCandidates;
	uint32_t numLaunches, kernelIdx, maxThreads;
	uint32_t activeThreads, idleThreads, numThreads;

	printf("-- Word size: %d - Query Size: %d - Query Space: %d - Threads per Query: %d - Queries per Warp: %d - Threads Idle: %d\n", 
			BASES_PER_THREAD, SIZE_QUERY, SPACE_PER_QUERY, THREADS_PER_QUERY, QUERIES_PER_WARP, WARP_THREADS_IDLE);

	#ifdef FUNNEL
		printf("-- OPT: funnelShift [ON] -- ");
	#else
		printf("-- OPT: funnelShift [OFF] -- ");
	#endif

	#ifdef SHUFFLE
		printf("shuffle [ON] -- ");
	#else
		printf("shuffle [OFF] -- ");
	#endif

	#ifdef BALLOT
		printf("ballot [ON]\n");
	#else
		printf("ballot [OFF]\n");
	#endif
	printf("\n");

	/////////LAUNCH GPU KERNELS:
	//LAUNCH KERNELS FOR KEPLERs GPUs
	if(DEVICE == 0){
		activeThreads = (qry->numCandidates * THREADS_PER_QUERY);
		idleThreads = ((activeThreads / WARP_THREADS_ACTIVE) * WARP_THREADS_IDLE);
		numThreads = activeThreads + idleThreads; 
		blocksPerGrid = (numThreads / MAX_THREADS_PER_SM) + ((numThreads % MAX_THREADS_PER_SM) ? 1 : 0);
		printf("KEPLER: LAUNCH KERNEL 0 -- Bloques: %d - Th_block %d - Th_sm %d\n", 
			blocksPerGrid, threadsPerBlock, MAX_THREADS_PER_SM);
		myersKernel<<<blocksPerGrid, threadsPerBlock>>>(qry->d_queries, ref->d_reference, qry->d_candidates, res->d_results,
		 			  	sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, 
						numEntriesPerCandidate, qry->numCandidates, numThreads);
		cudaThreadSynchronize();
	}

	//LAUNCH KERNELS FOR FERMIs GPUs
	if(DEVICE == 1){
		maxThreads = threadsPerBlock * 65535;
		maxCandidates = (maxThreads / SIZE_WARP) * QUERIES_PER_WARP;
		numLaunches = (qry->numCandidates / maxCandidates) + ((qry->numCandidates / maxCandidates) ? 1 : 0);
		lastCandidates = qry->numCandidates;
		processedCandidates = 0;

		for(kernelIdx=0; kernelIdx<numLaunches; kernelIdx++){
			numCandidates = MIN(lastCandidates, maxCandidates);
 			activeThreads = (numCandidates * THREADS_PER_QUERY);
			idleThreads = ((activeThreads / WARP_THREADS_ACTIVE) * WARP_THREADS_IDLE);
			numThreads = activeThreads + idleThreads; 
			blocksPerGrid = (numThreads / MAX_THREADS_PER_SM) + ((numThreads % MAX_THREADS_PER_SM) ? 1 : 0);
			printf("FERMI: LAUNCH KERNEL %d -- Bloques: %d - Th_block %d - Th_sm %d\n", 
				 kernelIdx, blocksPerGrid, threadsPerBlock, MAX_THREADS_PER_SM);
			myersKernel<<<blocksPerGrid, threadsPerBlock>>>(qry->d_queries, ref->d_reference, qry->d_candidates + processedCandidates, 
							res->d_results + processedCandidates, sizeCandidate, qry->sizeQueries, 
							ref->size, numEntriesPerQuery, numEntriesPerCandidate, numCandidates, numThreads);
			lastCandidates -= numCandidates;
			processedCandidates += numCandidates;
		}
		cudaThreadSynchronize();
	}
}

extern "C"
int transferCPUtoGPU(void *reference, void *queries, void *results)
{
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

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
	HANDLE_ERROR(cudaMemcpy(res->d_results, res->h_results, res->numResults * sizeof(resEntry_t), cudaMemcpyHostToDevice));

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
