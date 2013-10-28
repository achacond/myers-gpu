/*
 * PROJECT: GEM-Tools library
 * FILE: gt_map_align.h
 * DATE: 20/08/2012
 * AUTHOR(S): Santiago Marco-Sola <santiagomsola@gmail.com>
 * DESCRIPTION: // TODO
 */

#ifndef GT_MAP_ALIGN_H_
#define GT_MAP_ALIGN_H_

#include "gt_commons.h"
#include "gt_map.h"
#include "gt_sequence_archive.h"

/*
 * Error Codes
 */
#define GT_MAP_CHECK_ALG_MATCH_OUT_OF_SEQ 5
#define GT_MAP_CHECK_ALG_NO_MISMS 10
#define GT_MAP_CHECK_ALG_BAD_MISMS 11
#define GT_MAP_CHECK_ALG_MISMATCH 12
#define GT_MAP_CHECK_ALG_MISMS_OUT_OF_SEQ 13
#define GT_MAP_CHECK_ALG_INS_OUT_OF_SEQ 20
#define GT_MAP_CHECK_ALG_DEL_OUT_OF_SEQ 30

/*
 * Map check/recover operators
 */
GT_INLINE gt_status gt_map_block_check_alignment(
    gt_map* const map,char* const pattern,const uint64_t pattern_length,
    char* const sequence,const uint64_t sequence_length);
GT_INLINE gt_status gt_map_check_alignment_sa(
    gt_map* const map,gt_string* const pattern,gt_sequence_archive* const sequence_archive);

GT_INLINE gt_status gt_map_block_recover_mismatches(
    gt_map* const map,char* const pattern,const uint64_t pattern_length,
    char* const sequence,const uint64_t sequence_length);
GT_INLINE gt_status gt_map_recover_mismatches_sa(
    gt_map* const map,gt_string* const pattern,gt_sequence_archive* const sequence_archive);

/*
 * Map (Re)alignment operators
 */
GT_INLINE gt_status gt_map_block_realign_hamming(
    gt_map* const map,char* const pattern,char* const sequence,const uint64_t length);
GT_INLINE gt_status gt_map_realign_hamming_sa(
    gt_map* const map,gt_string* const pattern,gt_sequence_archive* const sequence_archive);

GT_INLINE void gt_map_realign_levenshtein_get_distance(
    char* const pattern,const uint64_t pattern_length,
    char* const sequence,const uint64_t sequence_length,
    const bool ends_free,uint64_t* const position,uint64_t* const distance,
    gt_vector* const buffer);
GT_INLINE gt_status gt_map_block_realign_levenshtein(
    gt_map* const map,char* const pattern,const uint64_t pattern_length,
    char* const sequence,const uint64_t sequence_length,
    const bool ends_free,gt_vector* const buffer);
GT_INLINE gt_status gt_map_realign_levenshtein_sa(
    gt_map* const map,gt_string* const pattern,gt_sequence_archive* const sequence_archive);

GT_INLINE gt_status gt_map_block_realign_weighted(
    gt_map* const map,char* const pattern,const uint64_t pattern_length,
    char* const sequence,const uint64_t sequence_length,int32_t (*gt_weigh_fx)(char*,char*));
GT_INLINE gt_status gt_map_realign_weighted_sa(
    gt_map* const map,gt_string* const pattern,
    gt_sequence_archive* const sequence_archive,int32_t (*gt_weigh_fx)(char*,char*));

/*
 * Bit-compressed (Re)alignment
 *   BMP[BitParalellMayers] - Myers' Fast Bit-Vector algorithm (Levenshtein)
 */
typedef struct {
  uint64_t *peq; // Pattern equalities (Bit vector for Myers-DP)
  uint64_t pattern_length;    // Length
  uint64_t pattern_num_words; // ceil(Length / |w|)
  uint64_t pattern_mod;       // Length % |w|
  uint64_t peq_length;        // ceil(Length / |w|) * |w|
  // Auxiliary data
  uint64_t* P;
  uint64_t* M;
  uint64_t* level_mask;
  int64_t* score;
  int64_t* init_score;
  // Auxiliary buffer
  gt_vector* internal_buffer;
} gt_bpm_pattern;

#define GT_BPM_PATTERN_CHECK(bpm_pattern) GT_NULL_CHECK(bpm_pattern->peq)

GT_INLINE gt_bpm_pattern* gt_map_bpm_pattern_new();
GT_INLINE void gt_map_bpm_pattern_compile(gt_bpm_pattern* const bpm_pattern,char* const pattern,const uint64_t pattern_length);
GT_INLINE void gt_map_bpm_pattern_delete(gt_bpm_pattern* const bpm_pattern);

GT_INLINE bool gt_map_block_bpm_get_distance(
    gt_bpm_pattern* const bpm_pattern,char* const sequence,const uint64_t sequence_length,
    uint64_t* const position,uint64_t* const distance,const uint64_t max_levenshtein_distance);
GT_INLINE gt_status gt_map_block_bpm_realign(
    gt_map* const map,gt_bpm_pattern* const bpm_pattern,char* const sequence,const uint64_t sequence_length);
GT_INLINE gt_status gt_map_bpm_realign_sa(
    gt_map* const map,gt_bpm_pattern* const bpm_pattern,gt_sequence_archive* const sequence_archive);
GT_INLINE void gt_map_bpm_search_alignment(
    gt_bpm_pattern* const bpm_pattern,
    char* const sequence,const uint64_t sequence_length,const uint64_t max_levenshtein_distance,
    gt_vector* const map_vector,const uint64_t max_num_matches);

/*
 * Error Messages
 */
#define GT_ERROR_MAP_ALG_WRONG_ALG "(Re)Aligning Map. Wrong alignment"
#define GT_ERROR_MAP_RECOVER_MISMS_WRONG_BASE_ALG "Recovering mismatches from map. Wrong initial alignment"

#endif /* GT_MAP_ALIGN_H_ */
