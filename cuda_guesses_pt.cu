// cuda_guesses.cu - 纯C风格的CUDA代码
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>

#define MAX_SEGMENTS 10000

typedef struct {
    char* d_values;
    int*  d_lengths;
    int*  d_offsets;
    int   num_values;
    int   initialized;
} GPUSegmentData;

static GPUSegmentData g_gpu_letters[MAX_SEGMENTS];
static GPUSegmentData g_gpu_digits[MAX_SEGMENTS];
static GPUSegmentData g_gpu_symbols[MAX_SEGMENTS];

//  固定输出缓冲区
static char* g_d_output = NULL;
static int*  g_d_offsets = NULL;
static int   g_output_capacity = 0;
static int   g_offsets_capacity = 0;

// 单segment 
__global__ void generateSingleKernel(
    char*       d_output,
    const int*  d_output_offsets,
    const char* d_values,
    const int*  d_lengths,
    const int*  d_offsets,
    int         num_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    int out_pos = d_output_offsets[idx];
    int val_pos = d_offsets[idx];
    int len = d_lengths[idx];

    for (int i = 0; i < len; i++) {
        d_output[out_pos + i] = d_values[val_pos + i];
    }
    d_output[out_pos + len] = '\0';
}

// 多segment 
__global__ void generateMultiKernel(
    char*       d_output,
    const int*  d_output_offsets,
    const char* d_prefix,
    int         prefix_len,
    const char* d_values,
    const int*  d_lengths,
    const int*  d_offsets,
    int         num_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    int out_pos = d_output_offsets[idx];

    for (int i = 0; i < prefix_len; i++) {
        d_output[out_pos + i] = d_prefix[i];
    }

    int val_pos = d_offsets[idx];
    int len = d_lengths[idx];
    for (int i = 0; i < len; i++) {
        d_output[out_pos + prefix_len + i] = d_values[val_pos + i];
    }
    d_output[out_pos + prefix_len + len] = '\0';
}


static GPUSegmentData* getCachePtr(int type, int id) {
    if (type == 1) {
        if (id >= 0 && id < MAX_SEGMENTS) return &g_gpu_letters[id];
    } else if (type == 2) {
        if (id >= 0 && id < MAX_SEGMENTS) return &g_gpu_digits[id];
    } else if (type == 3) {
        if (id >= 0 && id < MAX_SEGMENTS) return &g_gpu_symbols[id];
    }
    return NULL;
}

static void uploadSegmentToGPU(
    const char** values_array,
    const int*   value_lengths,
    int          num_values,
    GPUSegmentData* cache
) {
    if (cache->initialized) {
        if (cache->d_values)  cudaFree(cache->d_values);
        if (cache->d_lengths) cudaFree(cache->d_lengths);
        if (cache->d_offsets) cudaFree(cache->d_offsets);
    }

    cache->num_values = num_values;
    if (num_values == 0) {
        cache->initialized = 1;
        cache->d_values = NULL;
cache->d_lengths = NULL;
cache->d_offsets = NULL;
        return;
    }

    int* h_offsets = (int*)malloc(num_values * sizeof(int));
    int total_chars = 0;
    for (int i = 0; i < num_values; i++) {
        h_offsets[i] = total_chars;
        total_chars += value_lengths[i];
    }

    char* h_values = (char*)malloc(total_chars * sizeof(char));
    for (int i = 0; i < num_values; i++) {
        memcpy(h_values + h_offsets[i], values_array[i], value_lengths[i]);
    }

    cudaMalloc((void**)&cache->d_values,  total_chars * sizeof(char));
    cudaMalloc((void**)&cache->d_lengths, num_values * sizeof(int));
    cudaMalloc((void**)&cache->d_offsets, num_values * sizeof(int));

    cudaMemcpy(cache->d_values,  h_values,       total_chars * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(cache->d_lengths, value_lengths,  num_values * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(cache->d_offsets, h_offsets,      num_values * sizeof(int),   cudaMemcpyHostToDevice);

    cache->initialized = 1;

    free(h_values);
    free(h_offsets);
}


#ifdef __cplusplus
extern "C" {
#endif


void cuda_upload_segment(
    const char** values_array,
    const int*   value_lengths,
    int          num_values,
    int          type,
    int          seg_id
) {
    GPUSegmentData* cache = getCachePtr(type, seg_id);
    if (cache == NULL) return;
    uploadSegmentToGPU(values_array, value_lengths, num_values, cache);
}

// 初始化固定输出缓冲区
void cuda_init_buffers(int max_values, int max_output_size) {
    if (g_d_output)  cudaFree(g_d_output);
    if (g_d_offsets) cudaFree(g_d_offsets);

    cudaMalloc((void**)&g_d_output,  max_output_size * sizeof(char));
    cudaMalloc((void**)&g_d_offsets, max_values * sizeof(int));
    g_output_capacity  = max_output_size;
    g_offsets_capacity = max_values;
}

// 释放固定缓冲区
void cuda_free_buffers() {
    if (g_d_output)  { cudaFree(g_d_output);  g_d_output = NULL; }
    if (g_d_offsets) { cudaFree(g_d_offsets); g_d_offsets = NULL; }
    g_output_capacity = g_offsets_capacity = 0;
}


void cuda_generate_single_fast(
    int         seg_type,
    int         seg_id,
    char*       output_buffer,    // 主机端输出缓冲区
    const int*  output_offsets,   // 主机端偏移数组
    int         num_values
) {
    GPUSegmentData* cache = getCachePtr(seg_type, seg_id);
    if (cache == NULL || !cache->initialized || cache->num_values != num_values) return;

    if (num_values > g_offsets_capacity) return;

    // 拷贝偏移到GPU
    cudaMemcpy(g_d_offsets, output_offsets, num_values * sizeof(int), cudaMemcpyHostToDevice);

    // 计算输出总大小
    int last_offset = output_offsets[num_values - 1];
    int last_len;
    cudaMemcpy(&last_len, &cache->d_lengths[num_values - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int total_size = last_offset + last_len + 1;

    if (total_size > g_output_capacity) return;

    int threadsPerBlock = 256;
    int blocksPerGrid   = (num_values + threadsPerBlock - 1) / threadsPerBlock;

    generateSingleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        g_d_output, g_d_offsets,
        cache->d_values, cache->d_lengths, cache->d_offsets,
        num_values
    );

    cudaMemcpy(output_buffer, g_d_output, total_size * sizeof(char), cudaMemcpyDeviceToHost);
}


void cuda_generate_multi_fast(
    int         seg_type,
    int         seg_id,
    const char* prefix,
    int         prefix_len,
    char*       output_buffer,
    const int*  output_offsets,
    int         num_values
) {
    GPUSegmentData* cache = getCachePtr(seg_type, seg_id);
    if (cache == NULL || !cache->initialized || cache->num_values != num_values) return;

    if (num_values > g_offsets_capacity) return;

    // 拷贝偏移到GPU
    cudaMemcpy(g_d_offsets, output_offsets, num_values * sizeof(int), cudaMemcpyHostToDevice);

    // 计算输出总大小
    int last_offset = output_offsets[num_values - 1];
    int last_len;
    cudaMemcpy(&last_len, &cache->d_lengths[num_values - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int total_size = last_offset + prefix_len + last_len + 1;

    if (total_size > g_output_capacity) return;

    // 前缀拷贝到GPU固定缓冲区的前面部分
    char* d_prefix_gpu;
    cudaMalloc((void**)&d_prefix_gpu, prefix_len * sizeof(char));
    cudaMemcpy(d_prefix_gpu, prefix, prefix_len * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (num_values + threadsPerBlock - 1) / threadsPerBlock;

    generateMultiKernel<<<blocksPerGrid, threadsPerBlock>>>(
        g_d_output, g_d_offsets,
        d_prefix_gpu, prefix_len,
        cache->d_values, cache->d_lengths, cache->d_offsets,
        num_values
    );

    cudaMemcpy(output_buffer, g_d_output, total_size * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_prefix_gpu);
}

// 释放所有GPU缓存
void cuda_free_all_caches() {
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (g_gpu_letters[i].initialized) {
            if (g_gpu_letters[i].d_values)  cudaFree(g_gpu_letters[i].d_values);
            if (g_gpu_letters[i].d_lengths) cudaFree(g_gpu_letters[i].d_lengths);
            if (g_gpu_letters[i].d_offsets) cudaFree(g_gpu_letters[i].d_offsets);
            g_gpu_letters[i].initialized = 0;
        }
        if (g_gpu_digits[i].initialized) {
            if (g_gpu_digits[i].d_values)  cudaFree(g_gpu_digits[i].d_values);
            if (g_gpu_digits[i].d_lengths) cudaFree(g_gpu_digits[i].d_lengths);
            if (g_gpu_digits[i].d_offsets) cudaFree(g_gpu_digits[i].d_offsets);
            g_gpu_digits[i].initialized = 0;
        }
        if (g_gpu_symbols[i].initialized) {
            if (g_gpu_symbols[i].d_values)  cudaFree(g_gpu_symbols[i].d_values);
            if (g_gpu_symbols[i].d_lengths) cudaFree(g_gpu_symbols[i].d_lengths);
            if (g_gpu_symbols[i].d_offsets) cudaFree(g_gpu_symbols[i].d_offsets);
            g_gpu_symbols[i].initialized = 0;
        }
    }
}

#ifdef __cplusplus
}
#endif
