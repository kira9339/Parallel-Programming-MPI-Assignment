#ifndef CUDA_GUESSES_H
#define CUDA_GUESSES_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_upload_segment(const char** values_array, const int* value_lengths,
                         int num_values, int type, int seg_id);

void cuda_init_buffers(int max_values, int max_output_size);
void cuda_free_buffers();

void cuda_generate_single_fast(int seg_type, int seg_id, char* output_buffer,
                                const int* output_offsets, int num_values);
void cuda_generate_multi_fast(int seg_type, int seg_id, const char* prefix, int prefix_len,
                               char* output_buffer, const int* output_offsets, int num_values);

void cuda_free_all_caches();

#ifdef __cplusplus
}
#endif

#endif
