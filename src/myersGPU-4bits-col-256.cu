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
#define		BASES_PER_THREAD		256
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

#if (REG == 2)
	#define REG_PH	Ph_C
	#define REG_MH	Mh_C
#endif

#if (REG == 3)
	#define REG_PH	Ph_D
	#define REG_MH	Mh_D
#endif

#if (REG == 4)
	#define REG_PH	Ph_E
	#define REG_MH	Mh_E
#endif

#if (REG == 5)
	#define REG_PH	Ph_F
	#define REG_MH	Mh_F
#endif

#if (REG == 6)
	#define REG_PH	Ph_G
	#define REG_MH	Mh_G
#endif

#if (REG == 7)
	#define REG_PH	Ph_H
	#define REG_MH	Mh_H
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
inline __device__ void shared_collaborative_shift(uint32_t value_A, uint32_t value_B, uint32_t value_C, uint32_t value_D, uint32_t value_E, uint32_t value_F, uint32_t value_G, uint32_t value_H, 
					       						  const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff,
					       						  uint32_t* res_A, uint32_t* res_B, uint32_t* res_C, uint32_t* res_D, uint32_t* res_E, uint32_t* res_F, uint32_t* res_G, uint32_t* res_H)
{
	uint32_t carry;
	#ifdef FUNNEL
		interBuff[intraWarpIdx + 1] = value_H;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		value_H = __funnelshift_lc(value_G, value_H, 1);
		value_G = __funnelshift_lc(value_F, value_G, 1);
		value_F = __funnelshift_lc(value_E, value_F, 1);
		value_E = __funnelshift_lc(value_D, value_E, 1);
		value_D = __funnelshift_lc(value_C, value_D, 1);
		value_C = __funnelshift_lc(value_B, value_C, 1);
		value_B = __funnelshift_lc(value_A, value_B, 1);
		value_A = __funnelshift_lc(carry,   value_A, 1);
	#else
		interBuff[intraWarpIdx + 1] = value_H;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		value_H = (value_G >> 31) | (value_H << 1);
		value_G = (value_F >> 31) | (value_G << 1);
		value_F = (value_E >> 31) | (value_F << 1);
		value_E = (value_D >> 31) | (value_E << 1);
		value_D = (value_C >> 31) | (value_D << 1);
		value_C = (value_B >> 31) | (value_C << 1);
		value_B = (value_A >> 31) | (value_B << 1);
		value_A = (carry   >> 31) | (value_A << 1);

	#endif

	(* res_A) = value_A;
	(* res_B) = value_B;
	(* res_C) = value_C;
	(* res_D) = value_D;
	(* res_E) = value_E;
	(* res_F) = value_F;
	(* res_G) = value_G;
	(* res_H) = value_H;
}

inline __device__ void shared_collaborative_sum(const uint32_t a_A, const uint32_t a_B, const uint32_t a_C, const uint32_t a_D, const uint32_t a_E, const uint32_t a_F, const uint32_t a_G, const uint32_t a_H,
					 					 		const uint32_t b_A, const uint32_t b_B, const uint32_t b_C, const uint32_t b_D, const uint32_t b_E, const uint32_t b_F, const uint32_t b_G, const uint32_t b_H, 
												const uint32_t localThreadIdx, const uint32_t intraWarpIdx, volatile uint32_t *interBuff,
					 							uint32_t* sum_A, uint32_t* sum_B, uint32_t* sum_C, uint32_t* sum_D, uint32_t* sum_E, uint32_t* sum_F, uint32_t* sum_G, uint32_t* sum_H)
{

	uint32_t carry, c_A, c_B, c_C, c_D, c_E, c_F, c_G, c_H;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY_OUT(c_C, a_C, b_C)
	UADD__IN_CARRY_OUT(c_D, a_D, b_D)
	UADD__IN_CARRY_OUT(c_E, a_E, b_E)
	UADD__IN_CARRY_OUT(c_F, a_F, b_F)
	UADD__IN_CARRY_OUT(c_G, a_G, b_G)
	UADD__IN_CARRY_OUT(c_H, a_H, b_H)
	UADD__IN_CARRY    (carry, 0,   0)
/*
	interBuff[intraWarpIdx + 1] = carry;
	carry = interBuff[intraWarpIdx];
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY_OUT(c_C, c_C, 0)
	UADD__IN_CARRY_OUT(c_D, c_D, 0)
	UADD__IN_CARRY_OUT(c_E, c_E, 0)
	UADD__IN_CARRY_OUT(c_F, c_F, 0)
	UADD__IN_CARRY_OUT(c_G, c_G, 0)
	UADD__IN_CARRY_OUT(c_H, c_H, 0)
	UADD__IN_CARRY    (carry, 0, 0)
*/
	while(__any(carry)){
		interBuff[intraWarpIdx + 1] = carry;
		carry = interBuff[intraWarpIdx];
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY_OUT(c_C, c_C, 0)
		UADD__IN_CARRY_OUT(c_D, c_D, 0)
		UADD__IN_CARRY_OUT(c_E, c_E, 0)
		UADD__IN_CARRY_OUT(c_F, c_F, 0)
		UADD__IN_CARRY_OUT(c_G, c_G, 0)
		UADD__IN_CARRY_OUT(c_H, c_H, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
	(* sum_C) = c_C;
	(* sum_D) = c_D;
	(* sum_E) = c_E;
	(* sum_F) = c_F;
	(* sum_G) = c_G;
	(* sum_H) = c_H;
}
#endif

#ifdef SHUFFLE
inline __device__ void shuffle_collaborative_shift(uint32_t value_A, uint32_t value_B, uint32_t value_C, uint32_t value_D, uint32_t value_E, uint32_t value_F, uint32_t value_G, uint32_t value_H, 
					       						   const uint32_t localThreadIdx, 
					       						   uint32_t* res_A, uint32_t* res_B, uint32_t* res_C, uint32_t* res_D, uint32_t* res_E, uint32_t* res_F, uint32_t* res_G, uint32_t* res_H)
{
	uint32_t carry;
	#ifdef FUNNEL
		carry = __shfl_up((int) value_H, 1);
		carry = (localThreadIdx) ? carry : 0;
		value_H = __funnelshift_lc(value_G, value_H, 1);
		value_G = __funnelshift_lc(value_F, value_G, 1);
		value_F = __funnelshift_lc(value_E, value_F, 1);
		value_E = __funnelshift_lc(value_D, value_E, 1);
		value_D = __funnelshift_lc(value_C, value_D, 1);
		value_C = __funnelshift_lc(value_B, value_C, 1);
		value_B = __funnelshift_lc(value_A, value_B, 1);
		value_A = __funnelshift_lc(carry,   value_A, 1);
	#else
		carry = __shfl_up((int) value_H, 1);
		carry = (localThreadIdx) ? carry : 0;
		value_H = (value_G >> 31) | (value_H << 1);
		value_G = (value_F >> 31) | (value_G << 1);
		value_F = (value_E >> 31) | (value_F << 1);
		value_E = (value_D >> 31) | (value_E << 1);
		value_D = (value_C >> 31) | (value_D << 1);
		value_C = (value_B >> 31) | (value_C << 1);
		value_B = (value_A >> 31) | (value_B << 1);
		value_A = (carry   >> 31) | (value_A << 1);
	#endif
	(* res_A) = value_A;
	(* res_B) = value_B;
	(* res_C) = value_C;
	(* res_D) = value_D;
	(* res_E) = value_E;
	(* res_F) = value_F;
	(* res_G) = value_G;
	(* res_H) = value_H;
}

inline __device__ void shuffle_collaborative_sum(const uint32_t a_A, const uint32_t a_B, const uint32_t a_C, const uint32_t a_D, const uint32_t a_E, const uint32_t a_F, const uint32_t a_G, const uint32_t a_H,
					 					 		 const uint32_t b_A, const uint32_t b_B, const uint32_t b_C, const uint32_t b_D, const uint32_t b_E, const uint32_t b_F, const uint32_t b_G, const uint32_t b_H, 
					 							 const uint32_t localThreadIdx,
					 					 		 uint32_t* sum_A, uint32_t* sum_B, uint32_t* sum_C, uint32_t* sum_D, uint32_t* sum_E, uint32_t* sum_F, uint32_t* sum_G, uint32_t* sum_H)
{

	uint32_t carry, c_A, c_B, c_C, c_D, c_E, c_F, c_G, c_H;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY_OUT(c_C, a_C, b_C)
	UADD__IN_CARRY_OUT(c_D, a_D, b_D)
	UADD__IN_CARRY_OUT(c_E, a_E, b_E)
	UADD__IN_CARRY_OUT(c_F, a_F, b_F)
	UADD__IN_CARRY_OUT(c_G, a_G, b_G)
	UADD__IN_CARRY_OUT(c_H, a_H, b_H)
	UADD__IN_CARRY    (carry, 0,   0)
/*
	carry = __shfl_up((int) (carry), 1);
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY_OUT(c_C, c_C, 0)
	UADD__IN_CARRY_OUT(c_D, c_D, 0)
	UADD__IN_CARRY_OUT(c_E, c_E, 0)
	UADD__IN_CARRY_OUT(c_F, c_F, 0)
	UADD__IN_CARRY_OUT(c_G, c_G, 0)
	UADD__IN_CARRY_OUT(c_H, c_H, 0)
	UADD__IN_CARRY    (carry, 0, 0)
*/
	while(__any(carry)){
		carry = __shfl_up((int) (carry), 1);
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY_OUT(c_C, c_C, 0)
		UADD__IN_CARRY_OUT(c_D, c_D, 0)
		UADD__IN_CARRY_OUT(c_E, c_E, 0)
		UADD__IN_CARRY_OUT(c_F, c_F, 0)
		UADD__IN_CARRY_OUT(c_G, c_G, 0)
		UADD__IN_CARRY_OUT(c_H, c_H, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
	(* sum_C) = c_C;
	(* sum_D) = c_D;
	(* sum_E) = c_E;
	(* sum_F) = c_F;
	(* sum_G) = c_G;
	(* sum_H) = c_H;
}
#endif

#ifdef BALLOT
inline __device__ void ballot_collaborative_shift(uint32_t value_A, uint32_t value_B, uint32_t value_C, uint32_t value_D, uint32_t value_E, uint32_t value_F, uint32_t value_G, uint32_t value_H,
					       						  const uint32_t localThreadIdx, const uint32_t intraWarpIdx,
					       						  uint32_t* res_A, uint32_t* res_B, uint32_t* res_C, uint32_t* res_D, uint32_t* res_E, uint32_t* res_F, uint32_t* res_G, uint32_t* res_H)
{
	uint32_t carry;
	carry = ((__ballot(value_H >> 31) << 1) & (1 << intraWarpIdx)) != 0;
	carry = (localThreadIdx) ? carry : 0;
	value_H = (value_G >> 31) | (value_H << 1);
	value_G = (value_F >> 31) | (value_G << 1);
	value_F = (value_E >> 31) | (value_F << 1);
	value_E = (value_D >> 31) | (value_E << 1);
	value_D = (value_C >> 31) | (value_D << 1);
	value_C = (value_B >> 31) | (value_C << 1);
	value_B = (value_A >> 31) | (value_B << 1);
	value_A =  carry          | (value_A << 1);

	(* res_A) = value_A;
	(* res_B) = value_B;
	(* res_C) = value_C;
	(* res_D) = value_D;
	(* res_E) = value_E;
	(* res_F) = value_F;
	(* res_G) = value_G;
	(* res_H) = value_H;
}

inline __device__ void ballot_collaborative_sum(const uint32_t a_A, const uint32_t a_B, const uint32_t a_C, const uint32_t a_D, const uint32_t a_E, const uint32_t a_F, const uint32_t a_G, const uint32_t a_H, 
					 					 		const uint32_t b_A, const uint32_t b_B, const uint32_t b_C, const uint32_t b_D, const uint32_t b_E, const uint32_t b_F, const uint32_t b_G, const uint32_t b_H, 
					 							const uint32_t localThreadIdx, const uint32_t intraWarpIdx,
					 							uint32_t* sum_A, uint32_t* sum_B, uint32_t* sum_C, uint32_t* sum_D, uint32_t* sum_E, uint32_t* sum_F, uint32_t* sum_G, uint32_t* sum_H)
{
	uint32_t carry, c_A, c_B, c_C, c_D, c_E, c_F, c_G, c_H;

	UADD__CARRY_OUT   (c_A, a_A, b_A)
	UADD__IN_CARRY_OUT(c_B, a_B, b_B)
	UADD__IN_CARRY_OUT(c_C, a_C, b_C)
	UADD__IN_CARRY_OUT(c_D, a_D, b_D)
	UADD__IN_CARRY_OUT(c_E, a_E, b_E)
	UADD__IN_CARRY_OUT(c_F, a_F, b_F)
	UADD__IN_CARRY_OUT(c_G, a_G, b_G)
	UADD__IN_CARRY_OUT(c_H, a_H, b_H)
	UADD__IN_CARRY    (carry, 0,   0)
/*
	carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
	carry = (localThreadIdx) ? carry : 0;

	UADD__CARRY_OUT   (c_A, c_A, carry)
	UADD__IN_CARRY_OUT(c_B, c_B, 0)
	UADD__IN_CARRY_OUT(c_C, c_C, 0)
	UADD__IN_CARRY_OUT(c_D, c_D, 0)
	UADD__IN_CARRY_OUT(c_E, c_E, 0)
	UADD__IN_CARRY_OUT(c_F, c_F, 0)
	UADD__IN_CARRY_OUT(c_G, c_G, 0)
	UADD__IN_CARRY_OUT(c_H, c_H, 0)
	UADD__IN_CARRY    (carry, 0, 0)
*/
	while(__any(carry)){
		carry = ((__ballot(carry) << 1) & (1 << intraWarpIdx)) != 0;
		carry = (localThreadIdx) ? carry : 0;
		UADD__CARRY_OUT   (c_A, c_A, carry)
		UADD__IN_CARRY_OUT(c_B, c_B, 0)
		UADD__IN_CARRY_OUT(c_C, c_C, 0)
		UADD__IN_CARRY_OUT(c_D, c_D, 0)
		UADD__IN_CARRY_OUT(c_E, c_E, 0)
		UADD__IN_CARRY_OUT(c_F, c_F, 0)
		UADD__IN_CARRY_OUT(c_G, c_G, 0)
		UADD__IN_CARRY_OUT(c_H, c_H, 0)
		UADD__IN_CARRY    (carry, 0, 0)
	}

	(* sum_A) = c_A;
	(* sum_B) = c_B;
	(* sum_C) = c_C;
	(* sum_D) = c_D;
	(* sum_E) = c_E;
	(* sum_F) = c_F;
	(* sum_G) = c_G;
	(* sum_H) = c_H;
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

__global__ void myersKernel(const qryEntry_t *d_queries, const uint32_t * __restrict d_reference, const candInfo_t *d_candidates, resEntry_t *d_results,
			    const uint32_t sizeCandidate, const uint32_t sizeQueries, const uint32_t sizeRef, const uint32_t numEntriesPerQuery,
			    const uint32_t numEntriesPerCandidate, const uint32_t numCandidates, const uint32_t numThreads)
{

	const uint32_t * __restrict localCandidate;

	uint32_t Ph_A, Mh_A, Pv_A, Mv_A, Xv_A, Xh_A, Eq_A, tEq_A;
	uint32_t Ph_B, Mh_B, Pv_B, Mv_B, Xv_B, Xh_B, Eq_B, tEq_B;
	uint32_t Ph_C, Mh_C, Pv_C, Mv_C, Xv_C, Xh_C, Eq_C, tEq_C;
	uint32_t Ph_D, Mh_D, Pv_D, Mv_D, Xv_D, Xh_D, Eq_D, tEq_D;
	uint32_t Ph_E, Mh_E, Pv_E, Mv_E, Xv_E, Xh_E, Eq_E, tEq_E;
	uint32_t Ph_F, Mh_F, Pv_F, Mv_F, Xv_F, Xh_F, Eq_F, tEq_F;
	uint32_t Ph_G, Mh_G, Pv_G, Mv_G, Xv_G, Xh_G, Eq_G, tEq_G;
	uint32_t Ph_H, Mh_H, Pv_H, Mv_H, Xv_H, Xh_H, Eq_H, tEq_H;

	uint32_t Eq0_A, Eq1_A, Eq2_A, Eq3_A, Eq4_A;
	uint32_t Eq0_B, Eq1_B, Eq2_B, Eq3_B, Eq4_B;
	uint32_t Eq0_C, Eq1_C, Eq2_C, Eq3_C, Eq4_C;
	uint32_t Eq0_D, Eq1_D, Eq2_D, Eq3_D, Eq4_D;
	uint32_t Eq0_E, Eq1_E, Eq2_E, Eq3_E, Eq4_E;
	uint32_t Eq0_F, Eq1_F, Eq2_F, Eq3_F, Eq4_F;
	uint32_t Eq0_G, Eq1_G, Eq2_G, Eq3_G, Eq4_G;
	uint32_t Eq0_H, Eq1_H, Eq2_H, Eq3_H, Eq4_H;

	uint32_t sum_A, sum_B, sum_C, sum_D, sum_E, sum_F, sum_G, sum_H;

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

			Pv_C = MAX_VALUE;
			Mv_C = 0;

			Pv_D = MAX_VALUE;
			Mv_D = 0;

			Pv_E = MAX_VALUE;
			Mv_E = 0;

			Pv_F = MAX_VALUE;
			Mv_F = 0;

			Pv_G = MAX_VALUE;
			Mv_G = 0;

			Pv_H = MAX_VALUE;
			Mv_H = 0;

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

			Eq0_C = d_queries[entry + 2].bitmap[0];
			Eq1_C = d_queries[entry + 2].bitmap[1];
			Eq2_C = d_queries[entry + 2].bitmap[2];
			Eq3_C = d_queries[entry + 2].bitmap[3];
			Eq4_C = d_queries[entry + 2].bitmap[4];

			Eq0_D = d_queries[entry + 3].bitmap[0];
			Eq1_D = d_queries[entry + 3].bitmap[1];
			Eq2_D = d_queries[entry + 3].bitmap[2];
			Eq3_D = d_queries[entry + 3].bitmap[3];
			Eq4_D = d_queries[entry + 3].bitmap[4];

			Eq0_E = d_queries[entry + 4].bitmap[0];
			Eq1_E = d_queries[entry + 4].bitmap[1];
			Eq2_E = d_queries[entry + 4].bitmap[2];
			Eq3_E = d_queries[entry + 4].bitmap[3];
			Eq4_E = d_queries[entry + 4].bitmap[4];

			Eq0_F = d_queries[entry + 5].bitmap[0];
			Eq1_F = d_queries[entry + 5].bitmap[1];
			Eq2_F = d_queries[entry + 5].bitmap[2];
			Eq3_F = d_queries[entry + 5].bitmap[3];
			Eq4_F = d_queries[entry + 5].bitmap[4];

			Eq0_G = d_queries[entry + 6].bitmap[0];
			Eq1_G = d_queries[entry + 6].bitmap[1];
			Eq2_G = d_queries[entry + 6].bitmap[2];
			Eq3_G = d_queries[entry + 6].bitmap[3];
			Eq4_G = d_queries[entry + 6].bitmap[4];

			Eq0_H = d_queries[entry + 7].bitmap[0];
			Eq1_H = d_queries[entry + 7].bitmap[1];
			Eq2_H = d_queries[entry + 7].bitmap[2];
			Eq3_H = d_queries[entry + 7].bitmap[3];
			Eq4_H = d_queries[entry + 7].bitmap[4];

			for(idEntry = 0; idEntry < numEntriesPerCandidate; idEntry++){

				candidate = localCandidate[idEntry]; 

				for(intraBase = 0; intraBase < BASES_PER_ENTRY; intraBase++){	
					
					indexBase = candidate & 0x07;
					Eq_A = selectEq(indexBase, Eq0_A, Eq1_A, Eq2_A, Eq3_A, Eq4_A);
					Eq_B = selectEq(indexBase, Eq0_B, Eq1_B, Eq2_B, Eq3_B, Eq4_B);
					Eq_C = selectEq(indexBase, Eq0_C, Eq1_C, Eq2_C, Eq3_C, Eq4_C);
					Eq_D = selectEq(indexBase, Eq0_D, Eq1_D, Eq2_D, Eq3_D, Eq4_D);
					Eq_E = selectEq(indexBase, Eq0_E, Eq1_E, Eq2_E, Eq3_E, Eq4_E);
					Eq_F = selectEq(indexBase, Eq0_F, Eq1_F, Eq2_F, Eq3_F, Eq4_F);
					Eq_G = selectEq(indexBase, Eq0_G, Eq1_G, Eq2_G, Eq3_G, Eq4_G);
					Eq_H = selectEq(indexBase, Eq0_H, Eq1_H, Eq2_H, Eq3_H, Eq4_H);

					Xv_A = Eq_A | Mv_A;
					Xv_B = Eq_B | Mv_B;
					Xv_C = Eq_C | Mv_C;
					Xv_D = Eq_D | Mv_D;
					Xv_E = Eq_E | Mv_E;
					Xv_F = Eq_F | Mv_F;
					Xv_G = Eq_G | Mv_G;
					Xv_H = Eq_H | Mv_H;

					tEq_A = Eq_A & Pv_A;
					tEq_B = Eq_B & Pv_B;
					tEq_C = Eq_C & Pv_C;
					tEq_D = Eq_D & Pv_D;
					tEq_E = Eq_E & Pv_E;
					tEq_F = Eq_F & Pv_F;
					tEq_G = Eq_G & Pv_G;
					tEq_H = Eq_H & Pv_H;

					#ifdef SHUFFLE
						shuffle_collaborative_sum(tEq_A, tEq_B, tEq_C, tEq_D, tEq_E, tEq_F, tEq_G, tEq_H,  
									   			   Pv_A,  Pv_B,  Pv_C,  Pv_D,  Pv_E,  Pv_F,  Pv_G,  Pv_H, 
												  intraQueryThreadIdx, 
									  			  &sum_A, &sum_B, &sum_C, &sum_D, &sum_E, &sum_F, &sum_G, &sum_H);
					#else
						#ifdef BALLOT
							ballot_collaborative_sum(tEq_A, tEq_B, tEq_C, tEq_D, tEq_E, tEq_F, tEq_G, tEq_H,  
									   				  Pv_A,  Pv_B,  Pv_C,  Pv_D,  Pv_E,  Pv_F,  Pv_G,  Pv_H, 
													 intraQueryThreadIdx, intraWarpIdx,
									  				 &sum_A, &sum_B, &sum_C, &sum_D, &sum_E, &sum_F, &sum_G, &sum_H);
						#else
							shared_collaborative_sum(tEq_A, tEq_B, tEq_C, tEq_D, tEq_E, tEq_F, tEq_G, tEq_H,  
									   				  Pv_A,  Pv_B,  Pv_C,  Pv_D,  Pv_E,  Pv_F,  Pv_G,  Pv_H, 
													 intraQueryThreadIdx, intraWarpIdx, localInterBuff, 
									  				 &sum_A, &sum_B, &sum_C, &sum_D, &sum_E, &sum_F, &sum_G, &sum_H);
						#endif
					#endif

					Xh_A = (sum_A ^ Pv_A) | Eq_A;
					Xh_B = (sum_B ^ Pv_B) | Eq_B;
					Xh_C = (sum_C ^ Pv_C) | Eq_C;
					Xh_D = (sum_D ^ Pv_D) | Eq_D;
					Xh_E = (sum_E ^ Pv_E) | Eq_E;
					Xh_F = (sum_F ^ Pv_F) | Eq_F;
					Xh_G = (sum_G ^ Pv_G) | Eq_G;
					Xh_H = (sum_H ^ Pv_H) | Eq_H;

					Ph_A = Mv_A | ~(Xh_A | Pv_A);
					Ph_B = Mv_B | ~(Xh_B | Pv_B);
					Ph_C = Mv_C | ~(Xh_C | Pv_C);
					Ph_D = Mv_D | ~(Xh_D | Pv_D);
					Ph_E = Mv_E | ~(Xh_E | Pv_E);
					Ph_F = Mv_F | ~(Xh_F | Pv_F);
					Ph_G = Mv_G | ~(Xh_G | Pv_G);
					Ph_H = Mv_H | ~(Xh_H | Pv_H);

					Mh_A = Pv_A & Xh_A;
					Mh_B = Pv_B & Xh_B;
					Mh_C = Pv_C & Xh_C;
					Mh_D = Pv_D & Xh_D;
					Mh_E = Pv_E & Xh_E;
					Mh_F = Pv_F & Xh_F;
					Mh_G = Pv_G & Xh_G;
					Mh_H = Pv_H & Xh_H;

					score += ((REG_PH & mask) != 0) - ((REG_MH & mask) != 0);

					#ifdef SHUFFLE
						shuffle_collaborative_shift(Ph_A, Ph_B, Ph_C, Ph_D, Ph_E, Ph_F, Ph_G, Ph_H, 
													intraQueryThreadIdx, 
													&Ph_A, &Ph_B, &Ph_C, &Ph_D, &Ph_E, &Ph_F, &Ph_G, &Ph_H);
						shuffle_collaborative_shift(Mh_A, Mh_B, Mh_C, Mh_D, Mh_E, Mh_F, Mh_G, Mh_H, 
													intraQueryThreadIdx, 
													&Mh_A, &Mh_B, &Mh_C, &Mh_D, &Mh_E, &Mh_F, &Mh_G, &Mh_H);
					#else
						#ifdef BALLOT
							ballot_collaborative_shift(Ph_A, Ph_B, Ph_C, Ph_D, Ph_E, Ph_F, Ph_G, Ph_H, 
													   intraQueryThreadIdx, intraWarpIdx, 
													   &Ph_A, &Ph_B, &Ph_C, &Ph_D, &Ph_E, &Ph_F, &Ph_G, &Ph_H);
							ballot_collaborative_shift(Mh_A, Mh_B, Mh_C, Mh_D, Mh_E, Mh_F, Mh_G, Mh_H, 
													   intraQueryThreadIdx, intraWarpIdx, 
													   &Mh_A, &Mh_B, &Mh_C, &Mh_D, &Mh_E, &Mh_F, &Mh_G, &Mh_H);
						#else
							shared_collaborative_shift(Ph_A, Ph_B, Ph_C, Ph_D, Ph_E, Ph_F, Ph_G, Ph_H, 
													   intraQueryThreadIdx, intraWarpIdx, localInterBuff, 
													   &Ph_A, &Ph_B, &Ph_C, &Ph_D, &Ph_E, &Ph_F, &Ph_G, &Ph_H);
							shared_collaborative_shift(Mh_A, Mh_B, Mh_C, Mh_D, Mh_E, Mh_F, Mh_G, Mh_H, 
													   intraQueryThreadIdx, intraWarpIdx, localInterBuff, 
													   &Mh_A, &Mh_B, &Mh_C, &Mh_D, &Mh_E, &Mh_F, &Mh_G, &Mh_H);
						#endif
					#endif

					Pv_A = Mh_A | ~(Xv_A | Ph_A);
					Pv_B = Mh_B | ~(Xv_B | Ph_B);
					Pv_C = Mh_C | ~(Xv_C | Ph_C);
					Pv_D = Mh_D | ~(Xv_D | Ph_D);
					Pv_E = Mh_E | ~(Xv_E | Ph_E);
					Pv_F = Mh_F | ~(Xv_F | Ph_F);
					Pv_G = Mh_G | ~(Xv_G | Ph_G);
					Pv_H = Mh_H | ~(Xv_H | Ph_H);

					Mv_A = Ph_A & Xv_A;
					Mv_B = Ph_B & Xv_B;
					Mv_C = Ph_C & Xv_C;
					Mv_D = Ph_D & Xv_D;
					Mv_E = Ph_E & Xv_E;
					Mv_F = Ph_F & Xv_F;
					Mv_G = Ph_G & Xv_G;
					Mv_H = Ph_H & Xv_H;

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
} carry-ou
