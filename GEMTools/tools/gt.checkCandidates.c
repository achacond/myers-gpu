/*
 * PROJECT: GEM-Tools library
 * FILE: gt.checkCandidates.c
 * DATE: 08/10/2013
 * AUTHOR(S): Santiago Marco-Sola <santiagomsola@gmail.com>
 * DESCRIPTION:
 */

#include "gem_tools.h"

#include <omp.h>

/*
 * Constants
 */
#define GT_CC_EXPANSION_FACTOR 0.20

#define GT_GPROF_THREAD_ALL 0
#define GT_GPROF_NUM_CANDIDATES 1
#define GT_GPROF_RETRIEVE_TEXT 2
#define GT_GPROF_DP_DISTANCE 3
#define GT_GPROF_BPM_DISTANCE 4
#define GT_GPROF_GLOBAL 5

/*
 * Profile
 */
gt_profile** tprof;

/*
 * Parameters
 */
typedef struct {
  /* I/O */
  bool paired_end;
  char* name_input_file;
  char* name_output_file;
  char* name_reference_file;
  char* name_gem_index_file;
  /* Misc */
  uint64_t num_threads;
  bool verbose;
} gt_check_candidates_args;
gt_check_candidates_args parameters = {
  /* I/O */
  .name_input_file=NULL,
  .paired_end=false,
  .name_output_file=NULL,
  .name_reference_file=NULL,
  .name_gem_index_file=NULL,
  /* Misc */
  .num_threads=1,
  .verbose=false,
};
/*
 * Candidates Job
 */
typedef struct {
  gt_string* reference_read;
  gt_vector* candidates; // (gt_map*)
} gt_candidates_job;

GT_INLINE gt_candidates_job* gt_candidates_job_new() {
  gt_candidates_job* const candidates_job = gt_alloc(gt_candidates_job);
  candidates_job->reference_read = gt_string_new(1000);
  candidates_job->candidates = gt_vector_new(200,sizeof(gt_map*));
  return candidates_job;
}
GT_INLINE void gt_candidates_job_clean(gt_candidates_job* const candidates_job) {
  gt_string_clear(candidates_job->reference_read);
  GT_VECTOR_ITERATE(candidates_job->candidates,candidate,position,gt_map*) {
    gt_map_delete(*candidate);
  }
  gt_vector_clear(candidates_job->candidates);
}
GT_INLINE void gt_candidates_job_delete(gt_candidates_job* const candidates_job) {
  gt_candidates_job_clean(candidates_job);
  gt_string_delete(candidates_job->reference_read);
  gt_vector_delete(candidates_job->candidates);
}
GT_INLINE void gt_candidates_job_add_candidate(gt_candidates_job* const candidates_job,gt_map* const map) {
  gt_vector_insert(candidates_job->candidates,map,gt_map*);
}
/*
 * Cheching candidates
 */
GT_INLINE void gt_check_candidates_align(
    const uint64_t thread_id,
    gt_sequence_archive* const sequence_archive,gt_candidates_job* const candidates_job,
    gt_vector* const buffer,gt_bpm_pattern* const bpm_pattern) {
  // Prepare placeholder for reference sequences
  const uint64_t reference_length = gt_string_get_length(candidates_job->reference_read);
  const uint64_t extra_length =
      (uint64_t) ceil(GT_CC_EXPANSION_FACTOR*(float)reference_length);
  gt_string* const candidates_sequence = gt_string_new(reference_length+2*extra_length+1);
  // Compile pattern for Myers
  gt_map_bpm_pattern_compile(bpm_pattern,
      gt_string_get_string(candidates_job->reference_read),gt_string_get_length(candidates_job->reference_read));
  // Align all candidates
  GT_VECTOR_ITERATE(candidates_job->candidates,candidate_ptr,pos,gt_map*) {
    GPROF_INC_COUNTER(tprof[thread_id],GT_GPROF_NUM_CANDIDATES);
    gt_map* const candidate = *candidate_ptr;

    // Retrieve reference sequence (candidate text)
    GPROF_INC_COUNTER(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT);
    GPROF_START_TIMER(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT);
    if (gt_sequence_archive_retrieve_sequence_chunk(sequence_archive,
        gt_map_get_seq_name(candidate),FORWARD,gt_map_get_position(candidate),
        reference_length+extra_length,extra_length,candidates_sequence)) {
      gt_fatal_error_msg("Couldn't retrieve reference sequence");
    }
    GPROF_STOP_TIMER(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT);

    // Calculate distance using DP-levenshtein (reference test)
    GPROF_INC_COUNTER(tprof[thread_id],GT_GPROF_DP_DISTANCE);
    GPROF_START_TIMER(tprof[thread_id],GT_GPROF_DP_DISTANCE);
    uint64_t dp_position, dp_distance;
    gt_map_realign_levenshtein_get_distance(
        gt_string_get_string(candidates_job->reference_read),gt_string_get_length(candidates_job->reference_read),
        gt_string_get_string(candidates_sequence),gt_string_get_length(candidates_sequence),
        true,&dp_position,&dp_distance,buffer);
    GPROF_STOP_TIMER(tprof[thread_id],GT_GPROF_DP_DISTANCE);

    // Calculate distance using BPM (Myers)
    GPROF_INC_COUNTER(tprof[thread_id],GT_GPROF_BPM_DISTANCE);
    GPROF_START_TIMER(tprof[thread_id],GT_GPROF_BPM_DISTANCE);
    uint64_t bpm_position, bpm_distance;
    gt_map_block_bpm_get_distance(bpm_pattern,
        gt_string_get_string(candidates_sequence),gt_string_get_length(candidates_sequence),
        &bpm_position,&bpm_distance,gt_string_get_length(candidates_job->reference_read));
    GPROF_STOP_TIMER(tprof[thread_id],GT_GPROF_BPM_DISTANCE);

    // Check consistency
    if (bpm_position!=dp_position || bpm_distance!=dp_distance) {
      gt_error_msg("Alignment algorithms don't match => GEM  %lu\t"PRIgts":+:%lu\n"
          "  (distance,position) := Lev(%lu,%lu)!=BPM(%lu,%lu)\n"
          "  REF::"PRIgts"\n  CAN::"PRIgts"\n",
          candidate->gt_score,PRIgts_content(candidate->seq_name),candidate->position,
          dp_distance,dp_position,bpm_distance,dp_position,
          PRIgts_content(candidates_job->reference_read),PRIgts_content(candidates_sequence));
    } else if (candidate->gt_score!=UINT64_MAX && dp_distance!=candidate->gt_score) {
      gt_error_msg("GEM thinks otherwise => %lu\t"PRIgts":+:%lu\n"
          "  (GEMdistance,DPDistance) = (%lu,%lu)\n"
          "  REF::"PRIgts"\n  CAN::"PRIgts"\n",
          candidate->gt_score,PRIgts_content(candidate->seq_name),candidate->position,
          candidate->gt_score,dp_distance,
          PRIgts_content(candidates_job->reference_read),PRIgts_content(candidates_sequence));
    }

    // Realign using DP-levenshtein (reference test)
//    gt_map_block_realign_levenshtein(candidate,
//        gt_string_get_string(candidates_job->reference_read),gt_string_get_length(candidates_job->reference_read),
//        gt_string_get_string(candidates_sequence),gt_string_get_length(candidates_sequence),true,buffer);
//    printf("LEV  %lu  \n",gt_map_get_levenshtein_distance(candidate));

  }
  // Free
  gt_string_delete(candidates_sequence);
}
/*
 * Display Profile
 */
GT_INLINE void gt_check_candidates_show_profile(const uint64_t thread_id) {
  fprintf(stderr,"[Thread-%lu]Total.Time %2.3f\n",thread_id,GPROF_GET_TIMER(tprof[thread_id],GT_GPROF_THREAD_ALL));
  fprintf(stderr,"  --> Retrieve.Text %2.3f (%2.3f%%)\t(%lu calls)\t(%2.3f ms/call)\n",
      GPROF_GET_TIMER(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT),
      GPROF_TIME_PERCENTAGE(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT,GT_GPROF_THREAD_ALL),
      GPROF_GET_COUNTER(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT),
      GPROF_TIME_PER_CALL(tprof[thread_id],GT_GPROF_RETRIEVE_TEXT,GT_GPROF_RETRIEVE_TEXT));
  fprintf(stderr,"  --> DP.distance   %2.3f (%2.3f%%)\t(%lu calls)\t(%2.3f ms/call)\n",
      GPROF_GET_TIMER(tprof[thread_id],GT_GPROF_DP_DISTANCE),
      GPROF_TIME_PERCENTAGE(tprof[thread_id],GT_GPROF_DP_DISTANCE,GT_GPROF_THREAD_ALL),
      GPROF_GET_COUNTER(tprof[thread_id],GT_GPROF_DP_DISTANCE),
      GPROF_TIME_PER_CALL(tprof[thread_id],GT_GPROF_DP_DISTANCE,GT_GPROF_DP_DISTANCE));
  fprintf(stderr,"  --> BPM.distance  %2.3f (%2.3f%%)\t(%lu calls)\t(%2.3f ms/call)\n",
      GPROF_GET_TIMER(tprof[thread_id],GT_GPROF_BPM_DISTANCE),
      GPROF_TIME_PERCENTAGE(tprof[thread_id],GT_GPROF_BPM_DISTANCE,GT_GPROF_THREAD_ALL),
      GPROF_GET_COUNTER(tprof[thread_id],GT_GPROF_BPM_DISTANCE),
      GPROF_TIME_PER_CALL(tprof[thread_id],GT_GPROF_BPM_DISTANCE,GT_GPROF_BPM_DISTANCE));
  fprintf(stderr,"\n");
}
GT_INLINE void gt_check_candidates_show_general_profile() {
  GPROF_SUM_OVERLAP(tprof,parameters.num_threads);
  fprintf(stderr,"[GeneralProfile]\n");
  fprintf(stderr,"Total.Time %2.3f\t(%lu candidates)\n",
      GPROF_GET_TIMER(tprof[0],GT_GPROF_GLOBAL),
      GPROF_GET_COUNTER(tprof[0],GT_GPROF_NUM_CANDIDATES));
  fprintf(stderr,"  --> DP.distance   %2.3f (%2.3f%%)\t(%lu calls)\t(%2.3f ms/call)\n",
      GPROF_GET_TIMER(tprof[0],GT_GPROF_DP_DISTANCE),
      GPROF_TIME_PERCENTAGE(tprof[0],GT_GPROF_DP_DISTANCE,GT_GPROF_GLOBAL),
      GPROF_GET_COUNTER(tprof[0],GT_GPROF_DP_DISTANCE),
      GPROF_TIME_PER_CALL(tprof[0],GT_GPROF_DP_DISTANCE,GT_GPROF_DP_DISTANCE));
  fprintf(stderr,"  --> BPM.distance   %2.3f (%2.3f%%)\t(%lu calls)\t(%2.3f ms/call)\n",
      GPROF_GET_TIMER(tprof[0],GT_GPROF_BPM_DISTANCE),
      GPROF_TIME_PERCENTAGE(tprof[0],GT_GPROF_BPM_DISTANCE,GT_GPROF_GLOBAL),
      GPROF_GET_COUNTER(tprof[0],GT_GPROF_BPM_DISTANCE),
      GPROF_TIME_PER_CALL(tprof[0],GT_GPROF_BPM_DISTANCE,GT_GPROF_BPM_DISTANCE));
}
/*
 * I/O Work Loop
 */
GT_INLINE void gt_check_candidates_parse_candidates_line(
      const char** const text_line,gt_candidates_job* const candidates_job) {
  // Reset candidates_job
  gt_candidates_job_clean(candidates_job);
  /*
   * Parse line:
   *   TCAGATGCATCG.....CGAACAG chr10:+:38880860:-1 chr10:+:42383932:-1
   */
  // Read
  gt_input_parse_field(text_line,TAB,candidates_job->reference_read);
  // Candidates
  while (!gt_input_parse_eol(text_line)) {
    gt_map* map = gt_map_new();
    gt_input_parse_field(text_line,COLON,map->seq_name); // chr10
    gt_input_parse_skip_chars(text_line,2); // Skip "+:"
    gt_input_parse_integer(text_line,(int64_t*)&map->position); // 38880860
    gt_input_parse_next_char(text_line); // ":"
    gt_input_parse_integer(text_line,(int64_t*)&map->gt_score); // -1
    gt_input_parse_field(text_line,TAB,NULL); // Skip the rest
    // Add the candidate
    gt_candidates_job_add_candidate(candidates_job,map);
  }
}
GT_INLINE gt_status gt_check_candidates_parse_candidates(
    gt_buffered_input_file* const buffered_input,gt_candidates_job* const candidates_job) {
  gt_status error_code;
  // Check the end_of_block. Reload buffer if needed
  if (gt_buffered_input_file_eob(buffered_input)) {
    if ((error_code=gt_buffered_input_file_reload(buffered_input,1000))!=GT_INPUT_STATUS_OK) return error_code;
  }
  // Parse alignment
  gt_check_candidates_parse_candidates_line(gt_buffered_input_file_get_text_line(buffered_input),candidates_job);
  // Next record
  gt_buffered_input_file_skip_line(buffered_input);
  // OK
  return GT_INPUT_STATUS_OK;
}
void gt_check_candidates_read__write() {
  // Allocate profiles
  uint64_t i;
  tprof = gt_malloc(sizeof(gt_profile*)*parameters.num_threads);
  for (i=0;i<parameters.num_threads;++i) tprof[i] = GPROF_NEW(100);

  // Open I/O files
  gt_input_file* const input_file = gt_tools_open_input_file(parameters.name_input_file,GT_COMPRESSION_NONE);
  gt_output_file* const output_file = gt_tools_open_output_file(parameters.name_output_file,GT_COMPRESSION_NONE);

  // Open reference file
  gt_sequence_archive* sequence_archive =
      gt_tools_open_sequence_archive(parameters.name_gem_index_file,parameters.name_reference_file,true);

  // Parallel reading+process
  GPROF_START_TIMER(tprof[0],GT_GPROF_GLOBAL);
  #pragma omp parallel num_threads(parameters.num_threads)
  {
    // Thread ID
    const uint64_t thread_id = omp_get_thread_num();

    // Prepare IN/OUT buffers & printers
    gt_buffered_input_file* const buffered_input = gt_buffered_input_file_new(input_file);
    gt_buffered_output_file* const buffered_output = gt_buffered_output_file_new(output_file);
    gt_buffered_input_file_attach_buffered_output(buffered_input,buffered_output);

    /*
     * READ + PROCCESS Loop
     */
    gt_vector* const buffer = gt_vector_new(1000,8);
    gt_bpm_pattern* const bpm_pattern = gt_map_bpm_pattern_new();
    gt_candidates_job* const candidates_job = gt_candidates_job_new();
    while (gt_check_candidates_parse_candidates(buffered_input,candidates_job)) {

      // Do the job
      GPROF_START_TIMER(tprof[thread_id],GT_GPROF_THREAD_ALL);
      gt_check_candidates_align(thread_id,sequence_archive,candidates_job,buffer,bpm_pattern);
      GPROF_STOP_TIMER(tprof[thread_id],GT_GPROF_THREAD_ALL);

    }

    // Clean
    gt_vector_delete(buffer);
    gt_map_bpm_pattern_delete(bpm_pattern);
    gt_candidates_job_delete(candidates_job);
    gt_buffered_input_file_close(buffered_input);
    gt_buffered_output_file_close(buffered_output);
  }
  GPROF_STOP_TIMER(tprof[0],GT_GPROF_GLOBAL);

  // Global stats
  for (i=0;i<parameters.num_threads;++i) gt_check_candidates_show_profile(i);
  gt_check_candidates_show_general_profile();
  // Release archive & Clean
  gt_sequence_archive_delete(sequence_archive);
  gt_input_file_close(input_file);
  gt_output_file_close(output_file);
  for (i=0;i<parameters.num_threads;++i) GPROF_DELETE(tprof[i]);
  gt_free(tprof);
}
/*
 * Parse arguments
 */
gt_option gt_check_candidates_options[] = {
  /* I/O */
  { 'i', "input", GT_OPT_REQUIRED, GT_OPT_STRING, 2 , true, "<file>" , "" },
  { 'o', "output", GT_OPT_REQUIRED, GT_OPT_STRING, 2 , true, "<file>" , "" },
  { 'r', "reference", GT_OPT_REQUIRED, GT_OPT_STRING, 2 , true, "<file> (MultiFASTA/FASTA)" , "" },
  { 'I', "gem-index", GT_OPT_REQUIRED, GT_OPT_STRING, 2 , true, "<file> (GEM2-Index)" , "" },
  /* Misc */
  { 't', "threads", GT_OPT_REQUIRED, GT_OPT_INT, 3 , true, "" , "" },
  { 'h', "help", GT_OPT_NO_ARGUMENT, GT_OPT_NONE, 3 , true, "" , "" },
  {  0, "", 0, 0, 0, false, "", ""}
};
char* gt_check_candidates_groups[] = {
  /*  0 */ "Null",
  /*  1 */ "Unclassified",
  /*  2 */ "I/O",
  /*  3 */ "Misc"
};
void parse_arguments(int argc,char** argv) {
  GT_OPTIONS_ITERATE_BEGIN(check_candidates,option) {
    /* I/O */
    case 'i':
      parameters.name_input_file = optarg;
      break;
    case 'o':
      parameters.name_output_file = optarg;
      break;
    case 'r':
      parameters.name_reference_file = optarg;
      break;
    case 'I':
      parameters.name_gem_index_file = optarg;
      break;
    /* Misc */
    case 't': // threads
      parameters.num_threads = atol(optarg);
      break;
    case 'h': // help
      fprintf(stderr, "USE: ./gt.checkCandidates [ARGS]...\n");
      gt_options_fprint_menu(stderr,gt_filter_options,gt_filter_groups,false,false);
      exit(1);
    case '?':
    default:
      gt_fatal_error_msg("Option not recognized");
    }
  } GT_OPTIONS_ITERATE_END;
  /*
   * Parameters check
   */
  if (parameters.name_reference_file==NULL && parameters.name_gem_index_file==NULL) {
    gt_fatal_error_msg("Reference file required");
  }
}
/*
 * Main
 */
int main(int argc,char** argv) {
  // GT error handler
  gt_handle_error_signals();

  // Parsing command-line options
  parse_arguments(argc,argv);

  // Filter !!
  gt_check_candidates_read__write();

  return 0;
}

