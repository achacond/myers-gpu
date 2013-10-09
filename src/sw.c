#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "omp.h"


#define		MAX_VALUE		0xFFFFFFFF
#define		NUM_BITS		3
#define		NUM_BASES		5


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
    uint posEntry;
    uint size;
} qryInfo_t;

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
	char *char_reference;
} ref_t;

typedef struct {
	uint32_t numResults; 
	resEntry_t* h_results;
} res_t;

typedef struct {
	uint32_t totalSizeQueries;
	uint32_t totalQueriesEntries;
	uint32_t sizeQueries;
	uint32_t numQueries;
	uint32_t numCandidates;
	float distance;

	char *char_queries;
	candInfo_t *h_candidates;
} qry_t;



int initResults(void **results, void *queries)
{
	res_t *res = (res_t *) malloc(sizeof(res_t));
	qry_t *qry = (qry_t *) queries;
	uint32_t idCandidate;

	res->h_results = NULL;

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

int loadReference(const unsigned char *fn, void **reference)
{      
    ref_t *ref = (ref_t *) malloc(sizeof(ref_t));
	FILE *fp = NULL;
	unsigned char cadena[256];
	uint numCharacters = 0;
	uint sizeFile = 0;
	uint cleidos = 0;
	uint pos = 0;
	int i = 0;

	//reference init
	ref->size = 0;
	ref->numEntries = 0;
	ref->char_reference = NULL;

	fp = fopen(fn, "rb");
	if (fp==NULL) return (30);

	fseek(fp, 0L, SEEK_END);
	sizeFile = ftell(fp);
	rewind(fp);

	ref->char_reference = (unsigned char*) malloc(sizeFile * sizeof(char));
	if (ref==NULL) return (31);
	
	if ((fgets(cadena, 256, fp) == NULL) || (cadena[0] != '>')) 
		return (32);

	while((!feof(fp)) && (fgets(cadena, 256, fp) != NULL)){
		if (cadena[0] != '>'){
			cleidos = strlen(cadena);
			if(cleidos) cleidos--;
			memcpy((ref->char_reference + pos), cadena, cleidos);
			pos +=  cleidos;
		}
	}

	fclose(fp);

	ref->numEntries = (pos / 32) + ((pos % 32) ? 1 : 0);
	ref->size = pos;

    (*reference) = ref;
	return (0);
}

int loadQueries(const char *fn, void **queries, float distance)
{
    FILE *fp = NULL;
    qry_t *qry = (qry_t*) malloc(sizeof(qry_t));
    size_t result;
    long long int currentPosition;

    fp = fopen(fn, "rb");
    if (fp == NULL) return (2);

    fread(&qry->totalSizeQueries, sizeof(uint32_t), 1, fp);
    fread(&qry->totalQueriesEntries, sizeof(uint32_t), 1, fp);
    fread(&qry->sizeQueries, sizeof(uint32_t), 1, fp);
    fread(&qry->numQueries, sizeof(uint32_t), 1, fp);
    fread(&qry->numCandidates, sizeof(uint32_t), 1, fp);
	
	qry->distance = distance;
    qry->h_candidates = NULL;
    qry->char_queries = NULL;

    qry->h_candidates   = (candInfo_t *) malloc(qry->numCandidates * sizeof(candInfo_t));
        if (qry->h_candidates == NULL) return (31);
    qry->char_queries = (char *) malloc(qry->totalSizeQueries * sizeof(char));
        if (qry->char_queries == NULL) return (31);

    currentPosition = (5 * sizeof(uint32_t)) + (sizeof(qryEntry_t) * qry->totalQueriesEntries);
    fseek(fp,  currentPosition, SEEK_SET);
    result = fread(qry->h_candidates , sizeof(candInfo_t), qry->numCandidates, fp);
        if (result != qry->numCandidates) return (5);
    
    currentPosition += (sizeof(candInfo_t) * qry->numCandidates) + (sizeof(qryInfo_t) * qry->numQueries) + 
						(qry->numCandidates * sizeof(uint32_t));
    fseek(fp, currentPosition, SEEK_SET);
    result = fread(qry->char_queries , sizeof(char), qry->totalSizeQueries, fp);
        if (result != qry->totalSizeQueries) return (7);

    fclose(fp);
    (* queries) = qry;
    return (0);
}

int computeSW(char *char_queries, char *char_reference, candInfo_t *h_candidates, resEntry_t *h_results, 
		 	 uint32_t idCandidate, uint32_t sizeCandidate, uint32_t sizeQuery, uint32_t sizeRef, int32_t *matrix) {

	uint32_t positionRef = h_candidates[idCandidate].position;
	uint32_t idQuery = h_candidates[idCandidate].query;
	char *candidate = char_reference + positionRef, 
         *query = char_queries + (idQuery * sizeQuery);

	int32_t sizeColumn = sizeQuery + 1, numColumns = sizeCandidate + 1;
	int32_t idColumn, i, y, j;
	int32_t cellLeft, cellUpper, cellDiagonal, delta;
	int32_t score, minScore = sizeQuery;
	int32_t minColumn = 0;
	char base;

	if((positionRef < sizeRef) && (sizeRef - positionRef) > sizeCandidate){

		// Horizontal initialization 
		for(i = 0; i < numColumns; i++)
			matrix[i] = 0;

		// Vertical initialization 
		for(i = 0; i < sizeColumn; i++)
			matrix[i * numColumns] = i;

		// Compute SW-MATRIX
		for(idColumn = 1; idColumn < numColumns; idColumn++){

			base = candidate[idColumn - 1];
			for(y = 1; y < sizeColumn; y++){

				delta = (base != query[y - 1]);
				cellLeft = matrix[y * numColumns + (idColumn - 1)] + 1;
				cellUpper = matrix[(y - 1) * numColumns + idColumn] + 1;
				cellDiagonal = matrix[(y - 1) * numColumns + (idColumn - 1)] + delta;
				
				score = MIN(cellDiagonal, MIN(cellLeft, cellUpper));
				matrix[y * numColumns + idColumn] = score;
			}

			if(score < minScore){
				minScore = score;
				minColumn = idColumn - 1;
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

	int32_t *matrix = NULL;
	uint32_t sizeCandidate = qry->sizeQueries * (1 + 2 * qry->distance);
	uint32_t idCandidate;

	matrix = (int32_t *) malloc((qry->sizeQueries + 1) * (sizeCandidate + 1) * sizeof(int32_t));
		if (matrix == NULL) return (1);

	#pragma omp for schedule (static)
	for (idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){
		 computeSW(qry->char_queries, ref->char_reference, qry->h_candidates, res->h_results, 
		 			  idCandidate, sizeCandidate, qry->sizeQueries, ref->size, matrix);
	}

	free(matrix);
	return(0);
}


int freeQueries(void *queries)
{   
    qry_t *qry = (qry_t *) queries;  
	
    if(qry->char_queries != NULL){
        free(qry->char_queries);
        qry->char_queries = NULL;
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

    if(ref->char_reference != NULL){
        free(ref->char_reference);
        ref->char_reference=NULL;
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

	sprintf(resFileOut, "%s.res.sw", fn);

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
	printf("ERROR: %d\n", e);
    switch(e) {
        case 0:  return "No error"; break; 
        case 1:  return "Error opening reference file"; break; 
        case 2:  return "Error opening queries file"; break; 
        case 7:  return "Error reading queries file - char queries"; break; 
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
	
	ts=sampleTime();
	#pragma omp parallel private(n)
	{
		for(n=0; n < iter; n++)
			computeAllQueriesCPU(reference, queries, results);
	}
	ts1=sampleTime();

	total = (ts1 - ts) / iter;
	printf("TIME: \t %f \n", total);

    error = saveResults(qryFile, results);
    CATCH_ERROR(error);
    
    error = freeReference(reference);
    CATCH_ERROR(error);

    error = freeQueries(queries);
    CATCH_ERROR(error);

    error = freeResults(results);
    CATCH_ERROR(error);

    return (0);
}
