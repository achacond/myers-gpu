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

#define 	BASES_PER_ENTRY	8


#define HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
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

__global__ void myersKernel(const qryEntry_t *d_queries, const uint32_t __restrict *d_reference, const candInfo_t *d_candidates, resEntry_t *d_results, 
							const uint32_t sizeCandidate, const uint32_t sizeQueries, const uint32_t sizeRef, const uint32_t numEntriesPerQuery,
							const uint32_t numCandidates)
{ 
	__shared__ 
    uint32_t s_Pv[CUDA_NUM_THREADS * N_ENTRIES], s_Mv[CUDA_NUM_THREADS * N_ENTRIES];
    __shared__ 
    qryEntry_t s_Query[CUDA_NUM_THREADS * N_ENTRIES];

	uint32_t *tmpPv, *tmpMv;
	qryEntry_t *tmpQuery;
	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;
	uint32_t initEntry, idEntry, idColumn, indexBase, aline, mask;
	int8_t carry, nextCarry;
	uint32_t candidate;

	uint32_t idCandidate = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;

	if ((threadIdx.x < MAX_THREADS_PER_SM) && (idCandidate < numCandidates)){

		uint32_t positionRef = d_candidates[idCandidate].position;
		uint32_t entryRef = positionRef / BASES_PER_ENTRY;
		int32_t score = sizeQueries, minScore = sizeQueries;
		uint32_t minColumn = 0;
		uint32_t finalMask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
		uint32_t word = 0;

		if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

			tmpPv = s_Pv + (threadIdx.x * numEntriesPerQuery);
			tmpMv = s_Mv + (threadIdx.x * numEntriesPerQuery);
			tmpQuery = s_Query + (threadIdx.x * numEntriesPerQuery);

			initEntry = d_candidates[idCandidate].query * numEntriesPerQuery;
			for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
				tmpPv[idEntry] = MAX_VALUE; 
				tmpMv[idEntry] = 0;
				tmpQuery[idEntry].bitmap[0] = d_queries[initEntry + idEntry].bitmap[0];
				tmpQuery[idEntry].bitmap[1] = d_queries[initEntry + idEntry].bitmap[1];
				tmpQuery[idEntry].bitmap[2] = d_queries[initEntry + idEntry].bitmap[2];
				tmpQuery[idEntry].bitmap[3] = d_queries[initEntry + idEntry].bitmap[3];
				tmpQuery[idEntry].bitmap[4] = d_queries[initEntry + idEntry].bitmap[4];
			}

			for(idColumn = 0; idColumn < sizeCandidate; idColumn++){

				carry = 0;
				aline = (positionRef % BASES_PER_ENTRY);
				if((aline == 0) || (idColumn == 0)) {
						candidate = d_reference[entryRef + word] >>  (aline * NUM_BITS); 
						word++;
				}

				indexBase = candidate & 0x07;

				for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
					Pv = tmpPv[idEntry];
					Mv = tmpMv[idEntry];
					Eq = tmpQuery[idEntry].bitmap[indexBase];
					mask = (idEntry + 1 == numEntriesPerQuery) ? finalMask : HIGH_MASK_32;

					Xv = Eq | Mv;
					Eq |= (carry >> 1) & 1;
					Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

					Ph = Mv | ~(Xh | Pv);
					Mh = Pv & Xh;

					nextCarry = ((Ph & mask) != 0) - ((Mh & mask) != 0);

					Ph <<= 1;
					Mh <<= 1;

					Mh |= (carry >> 1) & 1;
					Ph |= (carry + 1) >> 1;

					carry = nextCarry;
					tmpPv[idEntry] = Mh | ~(Xv | Ph);
					tmpMv[idEntry] = Ph & Xv;
				}

				candidate >>= 4;
				positionRef++;
				
				score += carry;

				if(score < minScore){
					minScore = score;
					minColumn = idColumn;
				}		
			}

	    	d_results[idCandidate].column = minColumn;
	    	d_results[idCandidate].score = minScore;
		}
	}
}

extern "C" 
void computeAllQueriesGPU(void *reference, void *queries, void *results)
{
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

	uint32_t blocks, threads = MAX_THREADS_PER_SM;
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);
	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);

	uint32_t maxCandidates, numCandidates, lastCandidates, processedCandidates;
	uint32_t numLaunches, kernelIdx, maxThreads;

	/////////LAUNCH GPU KERNELS:
	//LAUNCH KERNELS FOR KEPLERs GPUs
	if(DEVICE == 0){
		blocks = (qry->numCandidates / MAX_THREADS_PER_SM) + ((qry->numCandidates % MAX_THREADS_PER_SM) ? 1 : 0);
		printf("KEPLER: LAUNCH KERNEL 0 -- Bloques: %d - Th_block %d - Th_sm %d\n", blocks, threads, MAX_THREADS_PER_SM);
		myersKernel<<<blocks,threads>>>(qry->d_queries, ref->d_reference, qry->d_candidates, res->d_results,
		 			  					sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, qry->numCandidates);
		cudaThreadSynchronize();
	}

	//LAUNCH KERNELS FOR FERMIs GPUs
	if(DEVICE == 1){
		maxThreads = threads * 65535;
		numLaunches = (qry->numCandidates / maxThreads) + ((qry->numCandidates % maxThreads) ? 1 : 0);
		lastCandidates = qry->numCandidates;
		processedCandidates = 0;

		for(kernelIdx=0; kernelIdx<numLaunches; kernelIdx++){
			maxCandidates = maxThreads;
			numCandidates = MIN(lastCandidates, maxCandidates);
			blocks = (numCandidates / MAX_THREADS_PER_SM) + ((numCandidates % MAX_THREADS_PER_SM) ? 1 : 0);
			printf("FERMI: LAUNCH KERNEL %d -- Bloques: %d - Th_block %d - Th_sm %d\n", kernelIdx, blocks, threads, MAX_THREADS_PER_SM);
			myersKernel<<<blocks,threads>>>(qry->d_queries, ref->d_reference, qry->d_candidates + processedCandidates, res->d_results + processedCandidates,
				 			  sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, numCandidates);
			cudaThreadSynchronize();
			lastCandidates -= numCandidates;
			processedCandidates += numCandidates;
		}
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
