#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define     NUM_BASES       5
#define     SIZE_HW_WORD    32


#define CATCH_ERROR(error) {{if (error) { fprintf(stderr, "%s\n", processError(error)); exit(EXIT_FAILURE); }}}
#ifndef MIN
    #define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN */



typedef struct {
    uint32_t bitmap[NUM_BASES];
} qryEntry_t;

typedef struct {
    uint32_t query;
    uint32_t position;
} candInfo_t;

typedef struct {
    uint posEntry;
    uint size;
} qryInfo_t;

typedef struct {
    uint32_t column;
    uint32_t score;
} resEntry_t;


typedef struct {
    float distance;
    uint32_t numResults; 
    resEntry_t* h_results;
} res_t;

typedef struct {
    uint32_t totalSizeQueries;
    uint32_t totalQueriesEntries;
    uint32_t sizeQueries;
    uint32_t numQueries;
    uint32_t numCandidates;

    candInfo_t *h_candidates;
    uint32_t *original_results;
} qry_t;

int loadQueries(const char *fn, void **queries)
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

    qry->h_candidates = NULL;
    qry->original_results = NULL;

    qry->h_candidates   = (candInfo_t *) malloc(qry->numCandidates * sizeof(candInfo_t));
        if (qry->h_candidates == NULL) return (31);
    qry->original_results = (uint32_t *) malloc(qry->numCandidates * sizeof(uint32_t));
        if (qry->original_results == NULL) return (31);

    currentPosition = (5 * sizeof(uint32_t)) + (sizeof(qryEntry_t) * qry->totalQueriesEntries);
    fseek(fp,  currentPosition, SEEK_SET);
    result = fread(qry->h_candidates , sizeof(candInfo_t), qry->numCandidates, fp);
        if (result != qry->numCandidates) return (5);
    
    currentPosition += (sizeof(candInfo_t) * qry->numCandidates) + (sizeof(qryInfo_t) * qry->numQueries);
    fseek(fp, currentPosition, SEEK_SET);
    result = fread(qry->original_results , sizeof(uint32_t), qry->numCandidates, fp);
        if (result != qry->numCandidates) return (6);

    fclose(fp);
    (* queries) = qry;
    return (0);
}

int loadResults(const char *fn, void **results, float distance)
{
    FILE *fp = NULL;
    res_t *res = (res_t *) malloc(sizeof(res_t));
    uint32_t idResult, column, score, idCandidate;
    size_t result;

    fp = fopen(fn, "r");
    if (fp == NULL) return (2);

    res->h_results = NULL;
    res->distance = distance;
    fscanf(fp, "%u", &res->numResults);

    res->h_results = (resEntry_t *) malloc(res->numResults * sizeof(resEntry_t));
        if (res->h_results == NULL) return (45);

    for(idResult = 0; idResult < res->numResults; idResult++){
        fscanf(fp, "%u %u %u", &idCandidate, &score, &column);
        res->h_results[idResult].column = column;
        res->h_results[idResult].score = score;
    }

    fclose(fp);
    (* results) = res;
    return (0);
}

int checkResults(void *queries, void *results)
{

    qry_t *qry = (qry_t *) queries;
    res_t *res = (res_t *) results;
    uint threshold = res->distance * 100;
    uint idCandidate, checkResult, candidateScore;

    printf("Differents results with threshold %u:\n", threshold); 
    for(idCandidate = 0; idCandidate < qry->numCandidates; idCandidate++){

		candidateScore = res->h_results[idCandidate].score;
		if (candidateScore > threshold) candidateScore = -1;
		checkResult = qry->original_results[idCandidate];
		
		if(candidateScore != checkResult){
           printf("C: %d; Q: %d - Position: %u, Res.column: %d, Res.score %d - ORes %d\n", 
            idCandidate, qry->h_candidates[idCandidate].query, 
            qry->h_candidates[idCandidate].position, res->h_results[idCandidate].column, res->h_results[idCandidate].score,
            qry->original_results[idCandidate]); 
        }
    }

    return(0);
}

char *processError(int e){ 
    printf("ERROR: %d\n", e);
    switch(e) {
        case 0:  return "No error"; break; 
        case 1:  return "Error opening reference file"; break; 
        case 2:  return "Error opening queries file"; break;
		case 5:  return "Error reading candidates"; break; 
		case 6:  return "Error reading original results"; break; 
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

int freeQueries(void *queries)
{   
    qry_t *qry = (qry_t *) queries;  
    
    if(qry->original_results != NULL){
        free(qry->original_results);
        qry->original_results = NULL;
    }

    if(qry->h_candidates != NULL){
        free(qry->h_candidates);
        qry->h_candidates = NULL;
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

int main(int argc, char *argv[])
{
    void *queries, *results;
    int32_t error;
    float distance         = atof(argv[1]);
    unsigned char *qryFile = argv[2];
    unsigned char *resFile = argv[3];


    printf("Loading queries ... \n");
    error = loadQueries(qryFile, &queries);
    CATCH_ERROR(error);

    printf("Loading results ...\n");
    error = loadResults(resFile, &results, distance);
    CATCH_ERROR(error);

    printf("Checking results ...\n");
    checkResults(queries, results);
    printf("Done\n");

    error = freeQueries(queries);
    CATCH_ERROR(error);

    error = freeResults(results);
    CATCH_ERROR(error);

    return (0);
}
