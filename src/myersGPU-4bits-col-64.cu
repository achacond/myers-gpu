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
#define		BASES_PER_THREAD		64
#define		ENTRIES_PER_THREAD		(BASES_PER_THREAD / SIZE_HW_WORD)

#define		SPACE_PER_QUERY			((((SIZE_QUERY-1)/BASES_PER_THREAD)+1) * BASES_PER_THREAD)
#define		BASES_PER_WARP			(SIZE_WARP * BASES_PER_THREAD)
#define 	QUERIES_PER_WARP		(BASES_PER_WARP / SPACE_PER_QUERY)
#define 	THREADS_PER_QUERY		(SPACE_PER_QUERY / BASES_PER_THREAD)
#define		WARP_THREADS_IDLE		(SIZE_WARP - (THREADS_PER_QUERY * QUERIES_PER_WARP))
#define		WARP_THREADS_ACTIVE		(THREADS_PER_QUERY * QUERIES_PER_WARP)

#define HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif

// output temporal carry in internal register
#define UADD__CARRY_OUT(c, a, b) \
     asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b));

// add & output with temporal carry of internal register
#define UADD__IN_CARRY_OUT(c, a, b) \
     asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b));

// add with temporal carry of internal register
#define UADD__IN_CARRY(c, a, b) \
     asm volatile("addc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b));

#if (REG == 0)
	#define REG_PH	Ph_A
	#define REG_MH	Mh_A
#endif

#if (REG == 1)
	#define REG_PH	Ph_B
	#define REG_MH	Mh_B
#endif

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
inline __device__ void shared_collaborative_shift(uint32_t value_A, uint32_t value_B, 
					       						   const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff,
					       						   uint32_t* res_A, uint32_t* res_B)
{
	uint32_t carry;
	#ifdef FUNNEL
		interBuff[intraWarpIdx + 1] = value_B;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		value_B = __funnelshift_lc(value_A, value_B, 1);
		value_A = __funnelshift_lc(carry, value_A, 1);
	#else
		interBuff[intraWarpIdx + 1] = value_B;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		value_B = (value_A >> 31) | (value_B << 1);
		value_A = (carry   >> 31) | (value_A << 1);
	#endif
	(* res_A) = value_A;
	(* res_B) = value_B;
}

inline __device__ void shared_collaborative_sum(const uint32_t a_A, const uint32_t a_B, 
					 const uint32_t b_A, const uint32_t b_B, 
					 const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff,
					 uint32_t* sum_A, uint32_t* sum_B)
{

	uint32_t carry, c_A, c_B;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY    (carry, 0, 0)

	interBuff[intraWarpIdx + 1] = carry;
	carry = interBuff[intraWarpIdx];
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY    (carry, 0, 0)

	while(__any(carry)){
		interBuff[intraWarpIdx + 1] = carry;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
}
#endif

#ifdef SHUFFLE
inline __device__ void shuffle_collaborative_shift(uint32_t value_A, uint32_t value_B, 
					       						   const uint32_t localThreadIdx, 
					       						   uint32_t* res_A, uint32_t* res_B)
{
	uint32_t carry;
	#ifdef FUNNEL
		carry = __shfl_up((int) value_B, 1);
		carry = (localThreadIdx) ? carry : 0;
		value_B = __funnelshift_lc(value_A, value_B, 1);
		value_A = __funnelshift_lc(carry, 	value_A, 1);
	#else
		carry = __shfl_up((int) value_B, 1);
		carry = (localThreadIdx) ? carry : 0;
		value_B = (value_A >> 31) | (value_B << 1);
		value_A = (carry   >> 31) | (value_A << 1);
	#endif
	(* res_A) = value_A;
	(* res_B) = value_B;
}

inline __device__ void shuffle_collaborative_sum(const uint32_t a_A, const uint32_t a_B, 
					 							 const uint32_t b_A, const uint32_t b_B, 
					 							 const uint32_t localThreadIdx,
					 							 uint32_t* sum_A, uint32_t* sum_B)
{

	uint32_t carry, c_A, c_B;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY    (carry, 0, 0)

	carry = __shfl_up((int) (carry), 1);
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY    (carry, 0, 0)

	while(__any(carry)){
		carry = __shfl_up((int) (carry), 1);
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
}
#endif

#ifdef BALLOT
inline __device__ void ballot_collaborative_shift(uint32_t value_A, uint32_t value_B, 
					       						   const uint32_t localThreadIdx, const uint32_t intraWarpIdx,
					       						   uint32_t* res_A, uint32_t* res_B)
{
	uint32_t carry;
	carry = ((__ballot(value_B >> 31) << 1) & (1 << intraWarpIdx)) != 0;
	carry = (localThreadIdx) ? carry : 0;
	value_B = (value_A >> 31) | (value_B << 1);
	value_A =  carry          | (value_A << 1);

	(* res_A) = value_A;
	(* res_B) = value_B;
}

inline __device__ void ballot_collaborative_sum(const uint32_t a_A, const uint32_t a_B, 
					 							 const uint32_t b_A, const uint32_t b_B, 
					 							 const uint32_t localThreadIdx, const uint32_t intraWarpIdx,
					 							 uint32_t* sum_A, uint32_t* sum_B)
{
	uint32_t carry, c_A, c_B;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY    (carry, 0, 0)

	carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY    (carry, 0, 0)

	while(__any(carry)){
		carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
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

	uint32_t Ph_A, Mh_A, Pv_A, Mv_A, Xv_A, Xh_A, Eq_A, tEq_A;
	uint32_t Ph_B, Mh_B, Pv_B, Mv_B, Xv_B, Xh_B, Eq_B, tEq_B;

	uint32_t Eq0_A, Eq1_A, Eq2_A, Eq3_A, Eq4_A;
	uint32_t Eq0_B, Eq1_B, Eq2_B, Eq3_B, Eq4_B;

	uint32_t sum_A, sum_B;

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

			Pv_A = MAX_VALUE;
			Mv_A = 0;

			Pv_B = MAX_VALUE;
			Mv_B = 0;

			entry = (d_candidates[idCandidate].query * numEntriesPerQuery) + (ENTRIES_PER_THREAD * intraQueryThreadIdx);

			Eq0_A = d_queries[entry].bitmap[0];
			Eq1_A = d_queries[entry].bitmap[1];
			Eq2_A = d_queries[entry].bitmap[2];
			Eq3_A = d_queries[entry].bitmap[3];
			Eq4_A = d_queries[entry].bitmap[4];

			Eq0_B = d_queries[entry + 1].bitmap[0];
			Eq1_B = d_queries[entry + 1].bitmap[1];
			Eq2_B = d_queries[entry + 1].bitmap[2];
			Eq3_B = d_queries[entry + 1].bitmap[3];
			Eq4_B = d_queries[entry + 1].bitmap[4];

			for(idEntry = 0; idEntry < numEntriesPerCandidate; idEntry++){

				candidate = localCandidate[idEntry]; 

				for(intraBase = 0; intraBase < BASES_PER_ENTRY; intraBase++){	
					
					indexBase = candidate & 0x07;
					Eq_A = selectEq(indexBase, Eq0_A, Eq1_A, Eq2_A, Eq3_A, Eq4_A);
					Eq_B = selectEq(indexBase, Eq0_B, Eq1_B, Eq2_B, Eq3_B, Eq4_B);

					Xv_A = Eq_A | Mv_A;
					Xv_B = Eq_B | Mv_B;

					tEq_A = Eq_A & Pv_A;
					tEq_B = Eq_B & Pv_B;

					#ifdef SHUFFLE
						shuffle_collaborative_sum(tEq_A, tEq_B, Pv_A, Pv_B, intraQueryThreadIdx, &sum_A, &sum_B);
					#else
						#ifdef BALLOT
							ballot_collaborative_sum(tEq_A, tEq_B, Pv_A, Pv_B, intraQueryThreadIdx, intraWarpIdx, &sum_A, &sum_B);
						#else
							shared_collaborative_sum(tEq_A, tEq_B, Pv_A, Pv_B, intraQueryThreadIdx, intraWarpIdx, localInterBuff, &sum_A, &sum_B);
						#endif
					#endif

					Xh_A = (sum_A ^ Pv_A) | Eq_A;
					Xh_B = (sum_B ^ Pv_B) | Eq_B;

					Ph_A = Mv_A | ~(Xh_A | Pv_A);
					Ph_B = Mv_B | ~(Xh_B | Pv_B);

					Mh_A = Pv_A & Xh_A;
					Mh_B = Pv_B & Xh_B;

					score += ((REG_PH & mask) != 0) - ((REG_MH & mask) != 0);

					#ifdef SHUFFLE
						shuffle_collaborative_shift(Ph_A, Ph_B, intraQueryThreadIdx, &Ph_A, &Ph_B);
						shuffle_collaborative_shift(Mh_A, Mh_B, intraQueryThreadIdx, &Mh_A, &Mh_B);
					#else
						#ifdef BALLOT
							ballot_collaborative_shift(Ph_A, Ph_B, intraQueryThreadIdx, intraWarpIdx, &Ph_A, &Ph_B);
							ballot_collaborative_shift(Mh_A, Mh_B, intraQueryThreadIdx, intraWarpIdx, &Mh_A, &Mh_B);
						#else
							shared_collaborative_shift(Ph_A, Ph_B, intraQueryThreadIdx, intraWarpIdx, localInterBuff, &Ph_A, &Ph_B);
							shared_collaborative_shift(Mh_A, Mh_B, intraQueryThreadIdx, intraWarpIdx, localInterBuff, &Mh_A, &Mh_B);
						#endif
					#endif

					Pv_A = Mh_A | ~(Xv_A | Ph_A);
					Pv_B = Mh_B | ~(Xv_B | Ph_B);

					Mv_A = Ph_A & Xv_A;
					Mv_B = Ph_B & Xv_B;

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

	printf("-- Word size: %d - Query Size: %d - Query Space: %d - Last Register: %d\n-- Threads per Query: %d - Queries per Warp: %d - Threads Idle: %d\n", 
			BASES_PER_THREAD, SIZE_QUERY, SPACE_PER_QUERY, REG, THREADS_PER_QUERY, QUERIES_PER_WARP, WARP_THREADS_IDLE);

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
