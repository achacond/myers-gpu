#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "omp.h"

#define		NUM_BITS		3
#define		NUM_BASES		5
#define		SIZE_HW_WORD	32
#define		MAX_VALUE		0xFFFFFFFF
#define		HIGH_MASK_32	0x80000000
#define		LOW_MASK_32		0x00000001



#define CATCH_ERROR(error) {{if (error) { fprintf(stderr, "%s\n", processError(error)); exit(EXIT_FAILURE); }}}
#ifndef MIN
	#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN */

double sampleTime()
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return((tv.tv_sec+tv.tv_nsec/1000000000.0));
}

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
	float distance;

	qryEntry_t *h_queries;
	qryEntry_t *d_queries;
	candInfo_t *h_candidates;
	candInfo_t *d_candidates;
	uint32_t *d_Pv;
	uint32_t *d_Mv;
} qry_t;

 int computeMyers(qryEntry_t *h_queries, refEntry_t *h_reference, candInfo_t *h_candidates, resEntry_t *h_results, 
 				  uint32_t idCandidate, uint32_t sizeCandidate, uint32_t sizeQueries, uint32_t sizeRef, uint32_t numEntriesPerQuery, 
 				  uint32_t *N_hin, uint32_t *P_hin, int8_t *T_hout)
{
  
	uint32_t tmpPv[numEntriesPerQuery], tmpMv[numEntriesPerQuery];
	uint32_t Ph, Mh, Pv, Mv, Xv, Xh, Eq;
	uint32_t candidateX, candidateY, candidateZ;
	uint32_t initEntry, idEntry, idColumn, indexBase, aline, mask;
	int8_t carry, nextCarry;

	uint32_t positionRef = h_candidates[idCandidate].position;
	uint32_t entryRef = positionRef / SIZE_HW_WORD;
	int32_t  score = sizeQueries,  minScore = sizeQueries;
	uint32_t minColumn = 0;
	uint32_t finalMask = ((sizeQueries % SIZE_HW_WORD) == 0) ? HIGH_MASK_32 : 1 << ((sizeQueries % SIZE_HW_WORD) - 1);
	uint32_t word = 0;

	if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

		//Init 
		initEntry = h_candidates[idCandidate].query * numEntriesPerQuery;
		for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
			tmpPv[idEntry] = MAX_VALUE;
			tmpMv[idEntry] = 0;
		}

		for(idColumn = 0; idColumn < sizeCandidate; idColumn++){

			//Read the next candidate letter (column)
			carry = 0;
			aline = (positionRef % SIZE_HW_WORD);
			if((aline == 0) || (idColumn == 0)) {
					candidateX = h_reference[entryRef + word].bitmap[0] << aline; 
					candidateY = h_reference[entryRef + word].bitmap[1] << aline;
					candidateZ = h_reference[entryRef + word].bitmap[2] << aline;
					word++;
			}

			indexBase = ((candidateX >> 31) & 0x1) | ((candidateY >> 30) & 0x2) | ((candidateZ >> 29) & 0x4);

			for(idEntry = 0; idEntry < numEntriesPerQuery; idEntry++){
				Pv = tmpPv[idEntry];
				Mv = tmpMv[idEntry];
				carry++;
				Eq = h_queries[initEntry + idEntry].bitmap[indexBase];
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
				tmpPv[idEntry] = Mh | ~(Xv | Ph);
				tmpMv[idEntry] = Ph & Xv;
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

	uint32_t numEntriesPerQuery = (qry->sizeQueries / SIZE_HW_WORD) + ((qry->sizeQueries % SIZE_HW_WORD) ? 1 : 0);
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);
	uint32_t idCandidate;

	// Myers algorithm for ASM
	uint32_t P_hin[3] = {0, 0, 1};
	uint32_t N_hin[3] = {1, 0, 0};
	int8_t T_hout[4] = {0, -1, 1, 1};

	//CPU program
	//Parallelize this bucle with openMP
	#pragma omp for schedule (static)
	for (idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){
		 computeMyers(qry->h_queries, ref->h_reference, qry->h_candidates, res->h_results, 
		 			  idCandidate, sizeCandidate, qry->sizeQueries, ref->size, numEntriesPerQuery, 
		 			  N_hin, P_hin, T_hout);
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

	// Elimina el 5to bitmap (G) nada hace matching
	//for(idQuery = 0; idQuery < qry->totalQueriesEntries; idQuery++){
	//	qry->h_queries[idQuery].bitmap[4] = 0;
	//}

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

	ref->h_reference = (refEntry_t *) malloc(ref->numEntries * sizeof(refEntry_t));
		if (ref->h_reference == NULL) return (31);
	result = fread(ref->h_reference, sizeof(refEntry_t), ref->numEntries, fp);
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
        res->h_results=NULL;
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
		sprintf(resFileOut, "%s.res.gpu", fn);
	#else
		sprintf(resFileOut, "%s.res.cpu", fn);
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
    void *reference, *queries, *results;
    int32_t error;
    float distance		   = atof(argv[1]);
    unsigned char *refFile = argv[2];
    unsigned char *qryFile = argv[3];

    double ts,ts1,total;
    uint iter = 1, n;

	error = loadReference(refFile, &reference);
    CATCH_ERROR(error);

    error = loadQueries(qryFile, &queries, distance);
    CATCH_ERROR(error);

    error = initResults(&results, queries);    
	CATCH_ERROR(error);
	
	#ifdef CUDA
		error = transferCPUtoGPU(reference, queries, results);
		CATCH_ERROR(error);
	#endif

	ts=sampleTime();

		#ifdef CUDA
			for(n = 0; n < iter; n++)
				computeAllQueriesGPU(reference, queries, results);
		#else
			#pragma omp parallel private(n)
			{

				for(n=0; n < iter; n++)
					computeAllQueriesCPU(reference, queries, results);
			}	
		#endif

	ts1=sampleTime();

	#ifdef CUDA
		error = transferGPUtoCPU(results);
		CATCH_ERROR(error);
	#endif

	total = (ts1 - ts) / iter;
	printf("TIME: \t %f \n", total);

    error = saveResults(qryFile, results);
    CATCH_ERROR(error);
    
	#ifdef CUDA
		error = freeReferenceGPU(reference);
		CATCH_ERROR(error);
		error = freeQueriesGPU(queries);
		CATCH_ERROR(error);
	    error = freeResultsGPU(results);
		CATCH_ERROR(error);
	#endif

    error = freeReference(reference);
    CATCH_ERROR(error);

    error = freeQueries(queries);
    CATCH_ERROR(error);

    error = freeResults(results);
    CATCH_ERROR(error);

    return (0);
}
