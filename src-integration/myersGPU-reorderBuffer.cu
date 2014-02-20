/*
 * Myers for GPU
 *
 *  Created on: 12/2/2014
 *      Author: Alejandro Chacón
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
#define		BASES_PER_THREAD		128
#define		ENTRIES_PER_THREAD		(BASES_PER_THREAD / SIZE_HW_WORD)
#define		NUMBUCKETS				(SIZE_WARP + 1)

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

typedef struct {
	uint4 bitmap[NUM_BASES];
} qryEntry_t;

typedef struct {
	uint32_t column;
	uint32_t score;
} resEntry_t;

typedef struct {
	uint32_t id;
	uint32_t local;
} testEntry_t;

typedef struct {
	uint32_t query;
	uint32_t position;
} candInfo_t;

typedef struct {
	uint posEntry;
	uint size;
} qryInfo_t;

typedef struct {
	uint32_t size;
	uint32_t numEntries;
	uint32_t *h_reference;
	uint32_t *d_reference;
} ref_t;

typedef struct {
	uint32_t	numBuckets;
	uint32_t	candidatesPerBufffer;
	uint32_t	numWarps;
	uint32_t   	*h_reorderBuffer;
	uint32_t   	*d_reorderBuffer;
	uint32_t 	*h_initPosPerBucket;
	uint32_t 	*h_initWarpPerBucket;
	uint32_t 	*d_initPosPerBucket;
	uint32_t 	*d_initWarpPerBucket;
} buff_t;

typedef struct {
	uint32_t numResults;
 	uint32_t numReorderedResults;
	resEntry_t* h_results;
	resEntry_t* d_results;
	resEntry_t* h_reorderResults;
	resEntry_t* d_reorderResults;
} res_t;

typedef struct {
	uint32_t totalSizeQueries;
	uint32_t totalQueriesEntries;
	uint32_t sizeQueries;
	uint32_t numQueries;
	uint32_t numCandidates;
	float distance;
	qryEntry_t *h_queries;
	qryEntry_t *d_queries;
	candInfo_t *h_candidates;
	candInfo_t *d_candidates;
	qryInfo_t *h_qinfo;
	qryInfo_t *d_qinfo;
} qry_t;

extern "C"
static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
      		printf( "%s in %s at line %d\n", cudaGetErrorString(err),  file, line );
       		exit( EXIT_FAILURE );
   	}
}

inline __device__ void shuffle_collaborative_shift(uint32_t value_A, uint32_t value_B, uint32_t value_C, uint32_t value_D,  
					       						   const uint32_t localThreadIdx, 
					       						   uint32_t* res_A, uint32_t* res_B, uint32_t* res_C, uint32_t* res_D)
{
	uint32_t carry;

	carry = __shfl_up((int) value_D, 1);
	carry = (localThreadIdx) ? carry : 0;
	value_D = __funnelshift_lc(value_C, value_D, 1);
	value_C = __funnelshift_lc(value_B, value_C, 1);
	value_B = __funnelshift_lc(value_A, value_B, 1);
	value_A = __funnelshift_lc(carry,   value_A, 1);

	(* res_A) = value_A;
	(* res_B) = value_B;
	(* res_C) = value_C;
	(* res_D) = value_D;
}

inline __device__ void shuffle_collaborative_sum(const uint32_t a_A, const uint32_t a_B, const uint32_t a_C, const uint32_t a_D, 
					 							 const uint32_t b_A, const uint32_t b_B, const uint32_t b_C, const uint32_t b_D, 
					 							 const uint32_t localThreadIdx,
					 							 uint32_t* sum_A, uint32_t* sum_B, uint32_t* sum_C, uint32_t* sum_D)
{

	uint32_t carry, c_A, c_B, c_C, c_D;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY_OUT(c_C, a_C, b_C)
	UADD__IN_CARRY_OUT(c_D, a_D, b_D)
	UADD__IN_CARRY    (carry, 0,   0)

	while(__any(carry)){
		carry = __shfl_up((int) (carry), 1);
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY_OUT(c_C, c_C, 0)
		UADD__IN_CARRY_OUT(c_D, c_D, 0)
		UADD__IN_CARRY    (carry, 0, 0) 
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
	(* sum_C) = c_C;
	(* sum_D) = c_D;
}

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

inline __device__ uint32_t select(const uint32_t indexWord, 
				    const uint32_t A, const uint32_t B, 
				    const uint32_t C, const uint32_t D)
{
	uint32_t value = A;

	value = (indexWord == 1) ? B : value;
	value = (indexWord == 2) ? C : value;
	value = (indexWord == 3) ? D : value;

	return value;
}

__device__ void myerslocalKernel(const qryEntry_t *d_queries, const uint32_t * __restrict d_reference, const candInfo_t *d_candidates, 
								 const uint32_t *d_reorderBuffer, resEntry_t *d_reorderResults, const qryInfo_t *d_qinfo, 
								 const uint32_t idCandidate, const uint32_t sizeRef, const uint32_t numReorderedResults, 
								 const float distance, const uint32_t intraQueryThreadIdx, const uint32_t threadsPerQuery)
{
	if (idCandidate < numReorderedResults){

		const uint32_t * __restrict localCandidate;

		uint32_t Ph_A, Mh_A, Pv_A, Mv_A, Xv_A, Xh_A, Eq_A, tEq_A;
		uint32_t Ph_B, Mh_B, Pv_B, Mv_B, Xv_B, Xh_B, Eq_B, tEq_B;
		uint32_t Ph_C, Mh_C, Pv_C, Mv_C, Xv_C, Xh_C, Eq_C, tEq_C;
		uint32_t Ph_D, Mh_D, Pv_D, Mv_D, Xv_D, Xh_D, Eq_D, tEq_D;
		uint4 	 Eq0, Eq1, Eq2, Eq3, Eq4;
		uint32_t PH, MH, indexWord;
		uint32_t sum_A, sum_B, sum_C, sum_D;

		const uint32_t originalCandidate = d_reorderBuffer[idCandidate];
		const uint32_t positionRef = d_candidates[originalCandidate].position;
		const uint32_t sizeQuery = d_qinfo[d_candidates[originalCandidate].query].size;
		const uint32_t entry = d_qinfo[d_candidates[originalCandidate].query].posEntry + intraQueryThreadIdx;
		const uint32_t sizeCandidate = sizeQuery * (1 + 2 * distance);
		const uint32_t numEntriesPerCandidate = (sizeCandidate / BASES_PER_ENTRY) + ((sizeCandidate % BASES_PER_ENTRY) ? 2 : 1);
		uint32_t candidate;

		const uint32_t mask = ((sizeQuery % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQuery % SIZE_HW_WORD) - 1);
		int32_t  score = sizeQuery, minScore = sizeQuery;
		uint32_t idColumn = 0, minColumn = 0, indexBase;
		uint32_t intraBase, idEntry;
		
		indexWord = ((sizeQuery - 1) & (BASES_PER_THREAD - 1)) / SIZE_HW_WORD;

		if((positionRef < sizeRef) && ((sizeRef - positionRef) > sizeCandidate)){

			localCandidate = d_reference + (positionRef / BASES_PER_ENTRY);

			Pv_A = MAX_VALUE;
			Mv_A = 0;

			Pv_B = MAX_VALUE;
			Mv_B = 0;

			Pv_C = MAX_VALUE;
			Mv_C = 0;

			Pv_D = MAX_VALUE;
			Mv_D = 0;

			Eq0 = d_queries[entry].bitmap[0];
			Eq1 = d_queries[entry].bitmap[1];
			Eq2 = d_queries[entry].bitmap[2];
			Eq3 = d_queries[entry].bitmap[3];
			Eq4 = d_queries[entry].bitmap[4];

			for(idEntry = 0; idEntry < numEntriesPerCandidate; idEntry++){

				candidate = localCandidate[idEntry];

				for(intraBase = 0; intraBase < BASES_PER_ENTRY; intraBase++){	
					
					indexBase = candidate & 0x07;
					Eq_A = selectEq(indexBase, Eq0.x, Eq1.x, Eq2.x, Eq3.x, Eq4.x);
					Eq_B = selectEq(indexBase, Eq0.y, Eq1.y, Eq2.y, Eq3.y, Eq4.y);
					Eq_C = selectEq(indexBase, Eq0.z, Eq1.z, Eq2.z, Eq3.z, Eq4.z);
					Eq_D = selectEq(indexBase, Eq0.w, Eq1.w, Eq2.w, Eq3.w, Eq4.w);

					Xv_A = Eq_A | Mv_A;
					Xv_B = Eq_B | Mv_B;
					Xv_C = Eq_C | Mv_C;
					Xv_D = Eq_D | Mv_D;

					tEq_A = Eq_A & Pv_A;
					tEq_B = Eq_B & Pv_B;
					tEq_C = Eq_C & Pv_C;
					tEq_D = Eq_D & Pv_D;

					shuffle_collaborative_sum(tEq_A, tEq_B, tEq_C, tEq_D, Pv_A, Pv_B, Pv_C, Pv_D, 
											  intraQueryThreadIdx, 
											  &sum_A, &sum_B, &sum_C, &sum_D);

					Xh_A = (sum_A ^ Pv_A) | Eq_A;
					Xh_B = (sum_B ^ Pv_B) | Eq_B;
					Xh_C = (sum_C ^ Pv_C) | Eq_C;
					Xh_D = (sum_D ^ Pv_D) | Eq_D;

					Ph_A = Mv_A | ~(Xh_A | Pv_A);
					Ph_B = Mv_B | ~(Xh_B | Pv_B);
					Ph_C = Mv_C | ~(Xh_C | Pv_C);
					Ph_D = Mv_D | ~(Xh_D | Pv_D);

					Mh_A = Pv_A & Xh_A;
					Mh_B = Pv_B & Xh_B;
					Mh_C = Pv_C & Xh_C;
					Mh_D = Pv_D & Xh_D;

					PH = select(indexWord, Ph_A, Ph_B, Ph_C, Ph_D);
					MH = select(indexWord, Mh_A, Mh_B, Mh_C, Mh_D);
					score += (((PH & mask) != 0) - ((MH & mask) != 0));

					shuffle_collaborative_shift(Ph_A, Ph_B, Ph_C, Ph_D, 
												intraQueryThreadIdx,
												&Ph_A, &Ph_B, &Ph_C, &Ph_D);
					shuffle_collaborative_shift(Mh_A, Mh_B, Mh_C, Mh_D, 
												intraQueryThreadIdx,
												&Mh_A, &Mh_B, &Mh_C, &Mh_D);

					Pv_A = Mh_A | ~(Xv_A | Ph_A);
					Pv_B = Mh_B | ~(Xv_B | Ph_B);
					Pv_C = Mh_C | ~(Xv_C | Ph_C);
					Pv_D = Mh_D | ~(Xv_D | Ph_D);

					Mv_A = Ph_A & Xv_A;
					Mv_B = Ph_B & Xv_B;
					Mv_C = Ph_C & Xv_C;
					Mv_D = Ph_D & Xv_D;

					candidate >>= 4;
					minColumn = (score < minScore) ? idColumn : minColumn;
					minScore  = (score < minScore) ? score    : minScore;
					if(intraQueryThreadIdx  == (threadsPerQuery - 1))
					idColumn++;
				}
			}

			if(intraQueryThreadIdx  == (threadsPerQuery - 1)){
	    		d_reorderResults[idCandidate].column = minColumn/* - (positionRef % BASES_PER_ENTRY)*/;
	    		d_reorderResults[idCandidate].score = minScore;
			}
		}
	}
}

__global__ void myersKernel(const qryEntry_t *d_queries, const uint32_t * d_reference, const candInfo_t *d_candidates, const uint32_t *d_reorderBuffer, 
						    resEntry_t *d_reorderResults, const qryInfo_t *d_qinfo, const uint32_t sizeRef,  const uint32_t numReorderedResults,
						    const float distance, uint32_t *d_initPosPerBucket, uint32_t *d_initWarpPerBucket, uint32_t numWarps)
{
		uint32_t bucketIdx = 0;
		uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t globalWarpIdx = globalThreadIdx / SIZE_WARP;
		uint32_t localThreadInTheBucket, idCandidate, intraQueryThreadIdx, threadsPerQuery, queriesPerWarp, localIdCandidateInTheBucket;

		while((bucketIdx != (SIZE_WARP + 1)) && (d_initWarpPerBucket[bucketIdx] <= globalWarpIdx)){
			bucketIdx++;
		}
		bucketIdx--;

		localThreadInTheBucket = globalThreadIdx - (d_initWarpPerBucket[bucketIdx] * SIZE_WARP);
		threadsPerQuery = bucketIdx + 1;
		queriesPerWarp = SIZE_WARP / threadsPerQuery;
		localIdCandidateInTheBucket = ((localThreadInTheBucket / SIZE_WARP) * queriesPerWarp) + ((threadIdx.x % SIZE_WARP) / threadsPerQuery);
		idCandidate = d_initPosPerBucket[bucketIdx] + localIdCandidateInTheBucket;
		intraQueryThreadIdx = (threadIdx.x % SIZE_WARP) % threadsPerQuery;

		myerslocalKernel(d_queries, d_reference, d_candidates, d_reorderBuffer, d_reorderResults, d_qinfo,
		 				 idCandidate, sizeRef, numReorderedResults, distance, intraQueryThreadIdx, threadsPerQuery);
}

extern "C"
void computeAllQueriesGPU(void *reference, void *queries, void *results, void *buffer)
{

	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;
	buff_t *buff = (buff_t *) buffer;

	uint32_t threadsPerBlock = 128;
	uint32_t numThreads = buff->numWarps * SIZE_WARP;
	uint32_t blocksPerGrid = (numThreads / threadsPerBlock) + ((numThreads % threadsPerBlock) ? 1 : 0);

	if(DEVICE == 0){
		printf("KEPLER: LAUNCH KERNEL 0 -- Bloques: %d - Th_block %d\n", blocksPerGrid, threadsPerBlock);
		myersKernel<<<blocksPerGrid, threadsPerBlock>>>(qry->d_queries, ref->d_reference, qry->d_candidates, buff->d_reorderBuffer, 
														res->d_reorderResults, qry->d_qinfo, ref->size, res->numReorderedResults, 
														qry->distance, buff->d_initPosPerBucket, buff->d_initWarpPerBucket, buff->numWarps);
		cudaThreadSynchronize();
	}

}

extern "C"
int reorderingBuffer(void *queries, void *results, void *buffer)
{

	qry_t  *qry  = (qry_t  *) queries;
	res_t  *res  = (res_t  *) results;
	buff_t *buff = (buff_t *) buffer; 

	uint32_t idBucket, idCandidate, idBuff;
	uint32_t numThreadsPerQuery;
	uint32_t numQueriesPerWarp;
	uint32_t tmpBuckets[buff->numBuckets];
	uint32_t numCandidatesPerBucket[buff->numBuckets];
	uint32_t numWarpsPerBucket[buff->numBuckets];

	//Init buckets (32 buckets => max 4096 bases)
	for(idBucket = 0; idBucket < buff->numBuckets; idBucket++){
		numCandidatesPerBucket[idBucket] = 0;
		numWarpsPerBucket[idBucket] = 0;
		tmpBuckets[idBucket] = 0;
	}

	//Fill buckets with elements per bucket
	for(idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){
		idBucket = (qry->h_qinfo[qry->h_candidates[idCandidate].query].size - 1) / BASES_PER_THREAD;
		idBucket = (idBucket < (buff->numBuckets - 1)) ? idBucket : (buff->numBuckets - 1);
		numCandidatesPerBucket[idBucket]++;
	}
	
	//Number of warps per bucket
	buff->candidatesPerBufffer = 0;
	for(idBucket = 0; idBucket < buff->numBuckets - 1; idBucket++){
		numThreadsPerQuery = idBucket + 1;
		numQueriesPerWarp = SIZE_WARP / numThreadsPerQuery;
		numWarpsPerBucket[idBucket] = (numCandidatesPerBucket[idBucket] / numQueriesPerWarp) + 
											((numCandidatesPerBucket[idBucket] % numQueriesPerWarp) ? 1 : 0);
		buff->h_initPosPerBucket[idBucket] = buff->candidatesPerBufffer;
		buff->candidatesPerBufffer += numWarpsPerBucket[idBucket] * numQueriesPerWarp;
	}

	//Fill init position warps per bucket
	for(idBucket = 1; idBucket < buff->numBuckets; idBucket++)
		buff->h_initWarpPerBucket[idBucket] = buff->h_initWarpPerBucket[idBucket-1] + numWarpsPerBucket[idBucket-1];

	//Allocate buffer (candidates)
	buff->h_reorderBuffer = (uint32_t *) malloc(buff->candidatesPerBufffer * sizeof(uint32_t));
		if (buff->h_reorderBuffer == NULL) return (34);
	for(idBuff = 0; idBuff < buff->candidatesPerBufffer; idBuff++)
		buff->h_reorderBuffer[idBuff] = MAX_VALUE;
	//Allocate buffer (results)
	res->numReorderedResults = buff->candidatesPerBufffer;
	res->h_reorderResults = (resEntry_t *) malloc(res->numReorderedResults * sizeof(resEntry_t));
		if (res->h_reorderResults == NULL) return (34);
	for(idBuff = 0; idBuff < res->numReorderedResults; idBuff++){
		res->h_reorderResults[idBuff].column = 0;
    	res->h_reorderResults[idBuff].score = MAX_VALUE;
	}

	//Reorder by size the candidates
	for(idBucket = 0; idBucket < buff->numBuckets; idBucket++)
		tmpBuckets[idBucket] = buff->h_initPosPerBucket[idBucket];
	for(idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){
		//idBucket = qry->h_qinfo[qry->h_candidates[idCandidate].query].size / BASES_PER_THREAD;
		idBucket = (qry->h_qinfo[qry->h_candidates[idCandidate].query].size - 1) / BASES_PER_THREAD;
		if (idBucket < (buff->numBuckets - 1)){
			buff->h_reorderBuffer[tmpBuckets[idBucket]] = idCandidate;
			tmpBuckets[idBucket]++;
		}
	}

	/// rellenar los huecos con elementos duplicados (se duplica el último)
	for(idBuff = 0; idBuff < buff->candidatesPerBufffer; idBuff++)
		if(buff->h_reorderBuffer[idBuff] == MAX_VALUE) buff->h_reorderBuffer[idBuff] = buff->h_reorderBuffer[idBuff-1];

	// Calculate the number of warps necessaries in the GPU
	for(idBucket = 0; idBucket < (buff->numBuckets - 1); idBucket++)
		buff->numWarps += numWarpsPerBucket[idBucket];
		
	return (0);
}

extern "C"
int reorderingResults(void *results, void *buffer)
{
	buff_t *buff = (buff_t *) buffer;
	res_t *res = (res_t *) results;

	uint32_t idRes;

	for(idRes = 0; idRes < res->numReorderedResults; idRes++){
		res->h_results[buff->h_reorderBuffer[idRes]].column = res->h_reorderResults[idRes].column;
		res->h_results[buff->h_reorderBuffer[idRes]].score = res->h_reorderResults[idRes].score;
	}

	return (0);
}

extern "C"
int transferCPUtoGPU(void *reference, void *queries, void *results, void *buffer)
{
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;
	buff_t *buff = (buff_t *) buffer;

    HANDLE_ERROR(cudaSetDevice(DEVICE));

	//allocate & transfer Binary Reference to GPU
	HANDLE_ERROR(cudaMalloc((void**) &ref->d_reference, ((uint64_t) ref->numEntries) * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(ref->d_reference, ref->h_reference, ((uint64_t) ref->numEntries) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//allocate & transfer Binary Queries to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_queries, qry->totalQueriesEntries * sizeof(qryEntry_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_queries, qry->h_queries, qry->totalQueriesEntries * sizeof(qryEntry_t), cudaMemcpyHostToDevice));

	//allocate & transfer to GPU the information associated with Binary Queries
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_qinfo, qry->numQueries * sizeof(qryInfo_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_qinfo, qry->h_qinfo, qry->numQueries * sizeof(qryInfo_t), cudaMemcpyHostToDevice));

	//allocate & transfer Candidates to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_candidates, qry->numCandidates * sizeof(candInfo_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_candidates, qry->h_candidates, qry->numCandidates * sizeof(candInfo_t), cudaMemcpyHostToDevice));

	//allocate & transfer reordered buffer to GPU
	HANDLE_ERROR(cudaMalloc((void**) &buff->d_reorderBuffer, buff->candidatesPerBufffer * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(buff->d_reorderBuffer, buff->h_reorderBuffer, buff->candidatesPerBufffer * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//allocate & transfer bucket information to GPU
	HANDLE_ERROR(cudaMalloc((void**) &buff->d_initPosPerBucket, buff->numBuckets * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(buff->d_initPosPerBucket, buff->h_initPosPerBucket, buff->numBuckets * sizeof(uint32_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**) &buff->d_initWarpPerBucket, buff->numBuckets * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(buff->d_initWarpPerBucket, buff->h_initWarpPerBucket, buff->numBuckets * sizeof(uint32_t), cudaMemcpyHostToDevice));

	//allocate Results
	HANDLE_ERROR(cudaMalloc((void**) &res->d_reorderResults, res->numReorderedResults * sizeof(resEntry_t)));
	HANDLE_ERROR(cudaMemcpy(res->d_reorderResults, res->h_reorderResults, res->numReorderedResults * sizeof(resEntry_t), cudaMemcpyHostToDevice));

	return (0);
}

extern "C"
int transferGPUtoCPU(void *results)
{
	res_t *res = (res_t *) results;

	HANDLE_ERROR(cudaMemcpy(res->h_reorderResults, res->d_reorderResults, res->numReorderedResults * sizeof(resEntry_t), cudaMemcpyDeviceToHost));

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

	if(res->d_reorderResults != NULL){
		cudaFree(res->d_results);
		res->d_reorderResults = NULL;
	}

	return(0);
}

extern "C"
int freeBufferGPU(void *buffer)
{
	buff_t *buff = (buff_t *) buffer;

	if(buff->d_reorderBuffer != NULL){
		cudaFree(buff->d_reorderBuffer);
		buff->d_reorderBuffer = NULL;
	}

	if(buff->d_initPosPerBucket != NULL){
		cudaFree(buff->d_initPosPerBucket);
		buff->d_initPosPerBucket = NULL;
	}

	if(buff->d_initWarpPerBucket != NULL){
		cudaFree(buff->d_initWarpPerBucket);
		buff->d_initWarpPerBucket = NULL;
	}

	return(0);
}
