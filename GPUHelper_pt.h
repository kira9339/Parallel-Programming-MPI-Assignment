#ifndef GPU_HELPER_H
#define GPU_HELPER_H

#include <string>
#include <vector>
#include "PCFG.h"


extern "C" {
    void cuda_upload_segment(const char** values_array, const int* value_lengths,
                             int num_values, int type, int seg_id);
    void cuda_generate_single(int seg_type, int seg_id, char* output_buffer,
                              const int* output_offsets, int num_values);
    void cuda_generate_multi(int seg_type, int seg_id, const char* prefix, int prefix_len,
                             char* output_buffer, const int* output_offsets, int num_values);
    void cuda_free_all_caches();
    void cuda_init_buffers(int max_values, int max_total_size);
    void cuda_free_buffers();
}


void generateGuessesGPU_Single(const segment* seg, int seg_type, int seg_id,
                                std::vector<std::string>& guesses, int& total_guesses);
void generateGuessesGPU_Multi(const std::string& prefix, const segment* seg, int seg_type, int seg_id,
                               std::vector<std::string>& guesses, int& total_guesses);


void initGPUBuffers(int max_num_values, int max_total_size);
void freeGPUBuffers();

#endif
