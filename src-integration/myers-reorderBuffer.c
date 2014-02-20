#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "omp.h"

#define		NUM_BITS		4
#define		NUM_BASES		5
#define		SIZE_HW_WORD	32
#define		MAX_VALUE		0xFFFFFFFF
#define		HIGH_MASK_32	0x80000000
#define		LOW_MASK_32		0x00000001

#define 	BASES_PER_ENTRY	8

#define		NUM_BASES_ENTRY	128
#define		NUM_SUB_ENTRIES (NUM_BASES_ENTRY / SIZE_HW_WORD)
#define		SIZE_WARP		32
#define		NUMBUCKETS		(SIZE_WARP + 1)


#define CATCH_ERROR(error) {{if (error) { fprintf(stderr, "%s\n", processError(error)); exit(EXIT_FAILURE); }}}
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif

double sampleTime()
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return((tv.tv_sec+tv.tv_nsec/1000000000.0));
}

typedef struct {
	uint32_t bitmap[NUM_BASES][NUM_SUB_ENTRIES];
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
	uint32_t numResults;
 	uint32_t numReorderedResults;
	resEntry_t* h_results;
	resEntry_t* d_results;
	resEntry_t* h_reorderResults;
	resEntry_t* d_reorderResults;
} res_t;

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

int computeMyers(qryEntry_t *h_queries, uint32_t *h_reference, resEntry_t *h_results, 
 				  uint32_t idCandidate, uint32_t sizeCandidate, uint32_t sizeQueries, uint32_t sizeRef, 
 				  uint32_t numEntriesPerQuery, uint32_t positionRef, uint32_t initEntry)
{
  
	uint32_t tmpPv[numEntriesPerQuery][4], tmpMv[numEntriesPerQuery][4];
	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;
	uint32_t candidate;
	uint32_t idEntry, idColumn, indexBase, aline, mask;
	int8_t carry, nextCarry;

	uint32_t entryRef = positionRef / BASES_PER_ENTRY;
	int32_t  score = sizeQueries,  minScore = sizeQueries;
	uint32_t minColumn = 0;
	uint32_t finalMask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
	uint32_t word = 0;
	uint32_t id, i;

	if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

		for(id = 0; id < numEntriesPerQuery/4; id++){
			for (i = 0; i < 4; ++i)
			{
				tmpPv[id][i] = MAX_VALUE;
				tmpMv[id][i] = 0;
			}
		}

		for(idColumn = 0; idColumn < sizeCandidate; idColumn++){

			carry = 0;
			aline = (positionRef % BASES_PER_ENTRY);
			if((aline == 0) || (idColumn == 0)) {
					candidate = h_reference[entryRef + word] >>  (aline * NUM_BITS); 
					word++;
			}

			indexBase = candidate & 0x07;

			for(id = 0; id < numEntriesPerQuery; id++){
				idEntry = id / 4;
				i = i % 4;
				Pv = tmpPv[idEntry][i];
				Mv = tmpMv[idEntry][i];
				Eq = h_queries[initEntry + idEntry].bitmap[indexBase][i];
				mask = (id + 1 == numEntriesPerQuery) ? finalMask : HIGH_MASK_32;

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
				tmpPv[idEntry][i] = Mh | ~(Xv | Ph);
				tmpMv[idEntry][i] = Ph & Xv;
				i++;
			}

			candidate >>= 4;
			positionRef++;
			
			score += carry;

			if(score < minScore){
				minScore = score;
				minColumn = idColumn;
			}
		
		}		
    	h_results[idCandidate].column = minColumn;
    	h_results[idCandidate].score = minScore;
	}

    return(0);
}

int computeAllQueriesCPU(void *reference, void *queries, void *results)
{
  
	ref_t *ref = (ref_t *) reference;
	qry_t *qry = (qry_t *) queries;
	res_t *res = (res_t *) results;

	uint32_t numEntriesPerQuery, numLargeEntriesPerQuery, sizeCandidate, sizeQuery, idCandidate, positionRef, initEntry;

	if(qry->sizeQueries != 0){
		//qry->numCandidates = 1000000;
		numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);
		numLargeEntriesPerQuery = (qry->sizeQueries / NUM_BASES_ENTRY) + ((qry->sizeQueries % NUM_BASES_ENTRY) ? 1 : 0);
		sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);
		#pragma omp for schedule (static)
		for (idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){
			positionRef = qry->h_candidates[idCandidate].position;
			initEntry = qry->h_candidates[idCandidate].query * numLargeEntriesPerQuery;
			computeMyers(qry->h_queries, ref->h_reference, res->h_results, 
			 			idCandidate, sizeCandidate, qry->sizeQueries, ref->size, 
						numEntriesPerQuery, positionRef, initEntry);
		}
	}else{
		//qry->numCandidates = 1000000;
		#pragma omp for schedule (static)
		for (idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){

			initEntry = qry->h_qinfo[qry->h_candidates[idCandidate].query].posEntry;
			sizeQuery = qry->h_qinfo[qry->h_candidates[idCandidate].query].size;
			positionRef = qry->h_candidates[idCandidate].position;
  			numEntriesPerQuery = (sizeQuery / SIZE_HW_WORD) + ((sizeQuery % SIZE_HW_WORD) ? 1 : 0);
			sizeCandidate = sizeQuery * (1 + 2 * qry->distance);

			computeMyers(qry->h_queries, ref->h_reference, res->h_results, 
			 			  idCandidate, sizeCandidate, sizeQuery, ref->size, 
						  numEntriesPerQuery, positionRef, initEntry);
		}
	}

	return(0);
}

int loadQueries(const char *fn, void **queries, float distance)
{
	FILE *fp = NULL;
	qry_t *qry = (qry_t*) malloc(sizeof(qry_t));
	size_t result;
	uint32_t idQuery;

	fp = fopen(fn, "rb");
	if (fp == NULL) return (2);

	fread(&qry->totalSizeQueries, sizeof(uint32_t), 1, fp);
	fread(&qry->totalQueriesEntries, sizeof(uint32_t), 1, fp);
	fread(&qry->sizeQueries, sizeof(uint32_t), 1, fp);
	fread(&qry->numQueries, sizeof(uint32_t), 1, fp);
	fread(&qry->numCandidates, sizeof(uint32_t), 1, fp);
	qry->distance = distance;

	qry->h_queries = NULL;
	qry->d_queries = NULL;
	qry->h_candidates = NULL;
	qry->d_candidates = NULL;

	qry->h_queries = (qryEntry_t *) malloc(qry->totalQueriesEntries * sizeof(qryEntry_t));
		if (qry->h_queries == NULL) return (33);
	result = fread(qry->h_queries, sizeof(qryEntry_t), qry->totalQueriesEntries, fp);
		if (result != qry->totalQueriesEntries) return (5);

	qry->h_candidates = (candInfo_t *) malloc(qry->numCandidates * sizeof(candInfo_t));
		if (qry->h_candidates == NULL) return (34);
	result = fread(qry->h_candidates , sizeof(candInfo_t), qry->numCandidates, fp);
		if (result != qry->numCandidates) return (5);

	qry->h_qinfo = (qryInfo_t *) malloc(qry->numQueries * sizeof(qryInfo_t));
		if (qry->h_qinfo == NULL) return (34);
	result = fread(qry->h_qinfo , sizeof(qryInfo_t), qry->numQueries, fp);
		if (result != qry->numQueries) return (5);

	fclose(fp);
	(* queries) = qry;
	return (0);
}

int loadReference(const char *fn, void **reference)
{
	FILE *fp = NULL;
	ref_t *ref = (ref_t *) malloc(sizeof(ref_t));
	size_t result;

	fp = fopen(fn, "rb");
	if (fp == NULL) return (1);

    fread(&ref->numEntries, sizeof(uint32_t), 1, fp);
    fread(&ref->size, sizeof(uint32_t), 1, fp);

	ref->h_reference = NULL;
	ref->d_reference = NULL;

	ref->h_reference = (uint32_t *) malloc(ref->numEntries * sizeof(uint32_t));
		if (ref->h_reference == NULL) return (31);
	result = fread(ref->h_reference, sizeof(uint32_t), ref->numEntries, fp);
		if (result != ref->numEntries) return (5);

	fclose(fp);
	(* reference) = ref;
	return (0);
}

int initResults(void **results, void *queries)
{
	res_t *res = (res_t *) malloc(sizeof(res_t));
	qry_t *qry = (qry_t *) queries;
	uint32_t idCandidate;

	res->h_results = NULL;
	res->d_results = NULL;

	res->numResults = qry->numCandidates;
	res->h_results = (resEntry_t *) malloc(res->numResults * sizeof(resEntry_t));
		if (res->h_results == NULL) return (45);

	for (idCandidate = 0; idCandidate < res->numResults; idCandidate++)
	{
		res->h_results[idCandidate].column = 0;
    	res->h_results[idCandidate].score = MAX_VALUE;
	}

	(* results) = res;
	return (0);
}

int initBuffer(void **buffer)
{
	buff_t *buff = (buff_t *) malloc(sizeof(buff_t));

	uint32_t idBucket;

	buff->numBuckets = NUMBUCKETS;
	buff->numWarps = 0;
	buff->h_reorderBuffer 		= NULL;
	buff->d_reorderBuffer		= NULL;
	buff->h_initPosPerBucket	= NULL;
	buff->h_initWarpPerBucket	= NULL;
	buff->d_initPosPerBucket	= NULL;
	buff->d_initWarpPerBucket	= NULL;

	buff->h_initPosPerBucket = (uint32_t *) malloc(buff->numBuckets * sizeof(uint32_t));
		if (buff->h_initPosPerBucket == NULL) return (45);

	buff->h_initWarpPerBucket = (uint32_t *) malloc(buff->numBuckets * sizeof(uint32_t));
		if (buff->h_initWarpPerBucket == NULL) return (45);

	for (idBucket = 0; idBucket < buff->numBuckets; idBucket++)
	{
		buff->h_initPosPerBucket[idBucket] = 0;
    	buff->h_initWarpPerBucket[idBucket] = 0;
	}

	(* buffer) = buff;
	return (0);
}

int freeQueries(void *queries)
{   
    qry_t *qry = (qry_t *) queries;  
	
    if(qry->h_queries != NULL){
        free(qry->h_queries);
        qry->h_queries = NULL;
    }

    if(qry->h_candidates != NULL){
        free(qry->h_candidates);
        qry->h_candidates = NULL;
    }  

    return(0);
}

int freeReference(void *reference)
{   
    ref_t *ref = (ref_t *) reference;  

    if(ref->h_reference != NULL){
        free(ref->h_reference);
        ref->h_reference=NULL;
    }
	
    return(0);
}

int freeResults(void *results)
{   
    res_t *res = (res_t *) results;  

    if(res->h_results != NULL){
        free(res->h_results);
        res->h_results = NULL;
    }

    if(res->h_reorderResults != NULL){
        free(res->h_reorderResults);
        res->h_reorderResults = NULL;
    }

    return(0);
}

int freeBuffer(void *buffer)
{
	buff_t *buff = (buff_t *) buffer;

	if(buff->h_reorderBuffer != NULL){
		free(buff->h_reorderBuffer);
		buff->h_reorderBuffer = NULL;
	}

	if(buff->h_initPosPerBucket != NULL){
		free(buff->h_initPosPerBucket);
		buff->h_initPosPerBucket = NULL;
	}

	if(buff->h_initWarpPerBucket != NULL){
		free(buff->h_initWarpPerBucket);
		buff->h_initWarpPerBucket = NULL;
	}

	return(0);
}

int saveResults(const char *fn, void *results)
{
	res_t *res = (res_t *) results;

	FILE *fp = NULL;
	char resFileOut[512];
	char cadena[256];
	uint32_t error, idCandidate;

	#ifdef CUDA
		sprintf(resFileOut, "%s.gemres.gpu", fn);
	#else
		sprintf(resFileOut, "%s.gemres.cpu", fn);
	#endif

	fp = fopen(resFileOut, "wb");
	if (fp == NULL) return (47);

	sprintf(cadena, "%u\n", res->numResults);
	fputs(cadena, fp);
	for(idCandidate = 0; idCandidate < res->numResults; idCandidate++){
		sprintf(cadena, "%u %u %u\n", idCandidate, res->h_results[idCandidate].score, res->h_results[idCandidate].column);
		fputs(cadena, fp);
	}

	fclose(fp);
	return (0);
}

char *processError(int e){ 

    switch(e) {
        case 0:  return "No error"; break; 
        case 1:  return "Error opening reference file"; break; 
        case 2:  return "Error opening queries file"; break; 
        case 30: return "Cannot open reference file"; break;
        case 31: return "Cannot allocate reference"; break;
        case 33: return "Cannot allocate queries"; break;
        case 34: return "Cannot allocate candidates"; break;
        case 32: return "Reference file isn't multifasta format"; break;
        case 37: return "Cannot open reference file on write mode"; break;
        case 42: return "Cannot open queries file"; break;
        case 43: return "Cannot allocate queries"; break;
        case 45: return "Cannot allocate results"; break;
        case 47: return "Cannot open results file for save intervals"; break;
        case 48: return "Cannot open results file for load intervals"; break;
        case 99: return "Not implemented"; break;
        default: return "Unknown error";
    }   
}

int main(int argc, char *argv[])
{
    void 		  *reference, *queries, *results, *buffer;
    int32_t 	   error;
    float 		   distance = atof(argv[1]);
    unsigned char *refFile  = argv[2];
    unsigned char *qryFile  = argv[3];

    double ts,ts1;

	error = loadReference(refFile, &reference);
    CATCH_ERROR(error);

    error = loadQueries(qryFile, &queries, distance);
    CATCH_ERROR(error);

    error = initBuffer(&buffer);
	CATCH_ERROR(error);

    error = initResults(&results, queries);    
	CATCH_ERROR(error);
	
	#ifdef CUDA
		ts=sampleTime();
			error = reorderingBuffer(queries, results, buffer);
			CATCH_ERROR(error);
		ts1=sampleTime();
		printf("TIME REORDERING BUFFER: \t %f \n", (ts1 - ts));

		ts=sampleTime();
			error = transferCPUtoGPU(reference, queries, results, buffer);
			CATCH_ERROR(error);
		ts1=sampleTime();
		printf("TIME TRANSFER CPU -> GPU: \t %f \n", (ts1 - ts));
	#endif

	ts=sampleTime();
		#ifdef CUDA
				computeAllQueriesGPU(reference, queries, results, buffer);
		#else
			#pragma omp parallel
			{
					computeAllQueriesCPU(reference, queries, results);
			}	
		#endif
	ts1=sampleTime();
	printf("TIME COMPUTE MYERS: \t %f \n", (ts1 - ts));

	#ifdef CUDA
		ts=sampleTime();
			error = transferGPUtoCPU(results);
			CATCH_ERROR(error);
		ts1=sampleTime();
		printf("TIME TRANSFER GPU -> CPU: \t %f \n", (ts1 - ts));

		ts=sampleTime();
			error = reorderingResults(results, buffer);
			CATCH_ERROR(error);
		ts1=sampleTime();
		printf("TIME REORDERING OUTPUT: \t %f \n", (ts1 - ts));
	#endif

    error = saveResults(qryFile, results);
    CATCH_ERROR(error);
    
	#ifdef CUDA
		error = freeReferenceGPU(reference);
		CATCH_ERROR(error);
		error = freeQueriesGPU(queries);
		CATCH_ERROR(error);
		error = freeBufferGPU(buffer);
		CATCH_ERROR(error);
	    error = freeResultsGPU(results);
		CATCH_ERROR(error);
	#endif

    error = freeReference(reference);
    CATCH_ERROR(error);

    error = freeQueries(queries);
    CATCH_ERROR(error);

    error = freeBuffer(buffer);
    CATCH_ERROR(error);

    error = freeResults(results);
    CATCH_ERROR(error);

    return (0);
}
