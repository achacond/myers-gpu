#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define     NUM_BITS        3
#define     NUM_BASES       5
#define     SIZE_HW_WORD    32
#define     MAX_VALUE       0xFFFFFFFF
#define     HIGH_MASK_32    0x80000000
#define     LOW_MASK_32     0x00000001


#define HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN */

typedef struct {
	uint32_t bitmap[NUM_BITS];
} refEntry_t;

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
	refEntry_t *h_reference;
	refEntry_t *d_reference;
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

const __constant__	uint32_t P_hin[3] = {0, 0, 1};
const __constant__	uint32_t N_hin[3] = {1, 0, 0};
const __constant__	int8_t T_hout[4] = {0, -1, 1, 1};

__global__ void myersKernel(qryEntry_t *d_queries, refEntry_t *d_reference, candInfo_t *d_candidates, resEntry_t *d_results, 
							uint32_t sizeCandidate, uint32_t sizeQueries, uint32_t sizeRef, uint32_t numEntriesPerQuery,
							uint32_t numCandidates, uint32_t concurrentThreads, /*uint32_t *d_Pv, uint32_t *d_Mv*/
							int32_t flag)
{ 
/*  
	uint32_t P_hin[3] = {0, 0, 1};
	uint32_t N_hin[3] = {1, 0, 0};
	int8_t T_hout[4] = {0, -1, 1, 1};
*/
	//uint32_t *tmpPv, *tmpMv;
	uint32_t tmpPv[N_ENTRIES], tmpMv[N_ENTRIES];
	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;
	uint32_t candidateX, candidateY, candidateZ;
	uint32_t initEntry, idEntry, idColumn, indexBase, aline, mask;
	int8_t carry, nextCarry;

	//uint32_t idCandidate = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;
	uint32_t idCandidate = 0;

	if ((threadIdx.x < MAX_THREADS_PER_SM) && (idCandidate < numCandidates)){

		uint32_t positionRef = d_candidates[idCandidate].position;
		uint32_t entryRef = positionRef / SIZE_HW_WORD;
		int32_t score = sizeQueries, minScore = sizeQueries;
		uint32_t minColumn = 0;
		uint32_t finalMask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
		uint32_t word = 0;

		if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

			//tmpPv = d_Pv + ((idCandidate % concurrentThreads) * numEntriesPerQuery);
			//tmpMv = d_Mv + ((idCandidate % concurrentThreads) * numEntriesPerQuery);

			//Init 
			initEntry = d_candidates[idCandidate].query * numEntriesPerQuery;
			for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
				tmpPv[idEntry] = MAX_VALUE; 
				tmpMv[idEntry] = 0;
			}

			for(idColumn = 0; idColumn < sizeCandidate; idColumn++){

				//Read the next candidate letter (column)
				carry = 0;
				aline = (positionRef % SIZE_HW_WORD);
				if((aline == 0) || (idColumn == 0)) {
						candidateX = d_reference[entryRef + word].bitmap[0] << aline; 
						candidateY = d_reference[entryRef + word].bitmap[1] << aline;
						candidateZ = d_reference[entryRef + word].bitmap[2] << aline;
						word++;
				}

				indexBase = ((candidateX >> 31) & 0x1) | ((candidateY >> 30) & 0x2) | ((candidateZ >> 29) & 0x4);

				for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
	
					if (1 == flag){
						Pv = tmpPv[idEntry];
						Mv = tmpMv[idEntry];
					}
					carry++;
					Eq = d_queries[initEntry + idEntry].bitmap[indexBase];
					mask = (idEntry + 1 == numEntriesPerQuery) ? finalMask : HIGH_MASK_32;

					Xv = Eq | Mv;
					Eq |= N_hin[carry];
					Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

					Ph = Mv | ~(Xh | Pv);
					Mh = Pv & Xh;

					nextCarry = T_hout[(((Ph & mask) != 0) * 2) + ((Mh & mask) != 0)];

					Ph <<= 1;
					Mh <<= 1;

					Mh |= N_hin[carry];
					Ph |= P_hin[carry];

					carry = nextCarry;

					Mh = Mh | ~(Xv | Ph);
					Ph = Ph & Xv;					

					if (1 == (Mh * Ph * flag)){
						tmpPv[idEntry] = Mh;
						tmpMv[idEntry] = Ph;
					}
				}

				candidateX <<= 1;
				candidateY <<= 1;
				candidateZ <<= 1;
				positionRef++;
				
				score += carry;

				if(score < minScore){
					minScore = score;
					minColumn = idColumn;
				}		
			}

		//MODIFICAR PARA NO ESCRIBIR SALIDA
			if (1 == flag){
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

	uint32_t concurrentThreads = 2048 * 8;

	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);

	uint32_t blocks = (qry->numCandidates / MAX_THREADS_PER_SM) + ((qry->numCandidates % MAX_THREADS_PER_SM) ? 1 : 0);
	uint32_t threads = CUDA_NUM_THREADS;

	printf("NumEntries: %d -- Bloques: %d - Th_block %d - Th_sm %d\n", N_ENTRIES, blocks, threads, MAX_THREADS_PER_SM);

	myersKernel<<<blocks,threads>>>(qry->d_queries, ref->d_reference, qry->d_candidates, res->d_results, 
		 			  sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, qry->numCandidates,
					  concurrentThreads, 0/*, qry->d_Pv, qry->d_Mv*/);

	cudaThreadSynchronize();
}

extern "C" 
int transferCPUtoGPU(void *reference, void *queries, void *results)
{
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

	//hardcoded: TODO query to GPU
	//The goal is safe memory to GPU
	uint32_t concurrentThreads = 2048 * 8;
 
	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);

    HANDLE_ERROR(cudaSetDevice(DEVICE));

	//allocate & transfer Binary Reference to GPU
	HANDLE_ERROR(cudaMalloc((void**) &ref->d_reference, ((uint64_t) ref->numEntries) * sizeof(refEntry_t)));
	HANDLE_ERROR(cudaMemcpy(ref->d_reference, ref->h_reference, ((uint64_t) ref->numEntries) * sizeof(refEntry_t), cudaMemcpyHostToDevice));

	//allocate & transfer Binary Queries to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_queries, qry->totalQueriesEntries * sizeof(qryEntry_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_queries, qry->h_queries, qry->totalQueriesEntries * sizeof(qryEntry_t), cudaMemcpyHostToDevice));

	//allocate & transfer Candidates to GPU
	HANDLE_ERROR(cudaMalloc((void**) &qry->d_candidates, qry->numCandidates * sizeof(candInfo_t)));
	HANDLE_ERROR(cudaMemcpy(qry->d_candidates, qry->h_candidates, qry->numCandidates * sizeof(candInfo_t), cudaMemcpyHostToDevice));

	//TODO: Allocate temporal PV MV arrays
	//HANDLE_ERROR(cudaMalloc((void**) &qry->d_Pv, numEntriesPerQuery * concurrentThreads * sizeof(uint32_t)));
 	//HANDLE_ERROR(cudaMemset(qry->d_Pv, 0, numEntriesPerQuery * qry->numCandidates * sizeof(uint32_t)));

	//TODO: Allocate temporal PV MV arrays
	//HANDLE_ERROR(cudaMalloc((void**) &qry->d_Mv, numEntriesPerQuery * concurrentThreads * sizeof(uint32_t)));
 	//HANDLE_ERROR(cudaMemset(qry->d_Mv, 0xFF, numEntriesPerQuery * qry->numCandidates * sizeof(uint32_t)));

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
	//recordar que hemos de liberar el fmi que esta en GPU
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
