#include "GPUHelper.h"
#include "cuda_guesses.h"
#include <cstring>
#include <iostream>

using namespace std;

// GPU缓冲区管理
static bool g_buffers_initialized = false;
static int  g_max_num_values = 0;
static int  g_max_total_size = 0;
static const int GPU_THRESHOLD = 5000;  // 只有value数超过此值才用GPU

// 初始化GPU缓冲区
void initGPUBuffers(int max_num_values, int max_total_size) {
    g_max_num_values = max_num_values;
    g_max_total_size = max_total_size;
    cuda_init_buffers(max_num_values, max_total_size);
    g_buffers_initialized = true;
}

// 释放GPU缓冲区
void freeGPUBuffers() {
    cuda_free_buffers();
    g_buffers_initialized = false;
}


static void ensureSegmentOnGPU(const segment* seg, int seg_type, int seg_id) {
    const vector<string>& values = seg->ordered_values;
    int n = (int)values.size();
    if (n == 0) return;

    const char** c_values = new const char*[n];
    int* c_lengths = new int[n];
    for (int i = 0; i < n; i++) {
        c_values[i] = values[i].c_str();
        c_lengths[i] = (int)values[i].length();
    }

    cuda_upload_segment(c_values, c_lengths, n, seg_type, seg_id);

    delete[] c_values;
    delete[] c_lengths;
}


void generateGuessesGPU_Single(const segment* seg, int seg_type, int seg_id,
                                vector<string>& guesses, int& total_guesses)
{
    const vector<string>& values = seg->ordered_values;
    int n = (int)values.size();
    if (n == 0) return;

    // 阈值判断
    if (n < GPU_THRESHOLD) {
        // CPU直接处理
        int old_size = (int)guesses.size();
        guesses.resize(old_size + n);
        for (int i = 0; i < n; i++) {
            guesses[old_size + i] = values[i];
        }
        total_guesses += n;
        return;
    }

    // GPU处理
    ensureSegmentOnGPU(seg, seg_type, seg_id);

    vector<int> h_offsets(n);
    int total_size = 0;
    for (int i = 0; i < n; i++) {
        h_offsets[i] = total_size;
        total_size += (int)values[i].length() + 1;
    }

    char* output_buffer = new char[total_size];
    

    cuda_generate_single_fast(seg_type, seg_id, output_buffer, h_offsets.data(), n);

    int old_size = (int)guesses.size();
    guesses.resize(old_size + n);
    for (int i = 0; i < n; i++) {
        guesses[old_size + i] = string(output_buffer + h_offsets[i]);
    }
    total_guesses += n;

    delete[] output_buffer;
}


void generateGuessesGPU_Multi(const string& prefix, const segment* seg, int seg_type, int seg_id,
                               vector<string>& guesses, int& total_guesses)
{
    const vector<string>& values = seg->ordered_values;
    int n = (int)values.size();
    if (n == 0) return;

    int prefix_len = (int)prefix.length();

    if (n < GPU_THRESHOLD) {
        // CPU直接处理
        int old_size = (int)guesses.size();
        guesses.resize(old_size + n);
        for (int i = 0; i < n; i++) {
            guesses[old_size + i] = prefix + values[i];
        }
        total_guesses += n;
        return;
    }

    // GPU处理
    ensureSegmentOnGPU(seg, seg_type, seg_id);

    vector<int> h_offsets(n);
    int total_size = 0;
    for (int i = 0; i < n; i++) {
        h_offsets[i] = total_size;
        total_size += prefix_len + (int)values[i].length() + 1;
    }

    char* output_buffer = new char[total_size];
    
    cuda_generate_multi_fast(seg_type, seg_id, prefix.c_str(), prefix_len,
                              output_buffer, h_offsets.data(), n);

    int old_size = (int)guesses.size();
    guesses.resize(old_size + n);
    for (int i = 0; i < n; i++) {
        guesses[old_size + i] = string(output_buffer + h_offsets[i]);
    }
    total_guesses += n;

    delete[] output_buffer;
}

void PriorityQueue::initGPU() {
    // 计算所有segment中最大的value数量和最大猜测长度
    int max_values = 0;
    int max_guess_len = 0;
    
    for (auto& seg : m.letters) {
        if ((int)seg.ordered_values.size() > max_values)
            max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) {
            if ((int)v.length() > max_guess_len)
                max_guess_len = (int)v.length();
        }
    }
    for (auto& seg : m.digits) {
        if ((int)seg.ordered_values.size() > max_values)
            max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) {
            if ((int)v.length() > max_guess_len)
                max_guess_len = (int)v.length();
        }
    }
    for (auto& seg : m.symbols) {
        if ((int)seg.ordered_values.size() > max_values)
            max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) {
            if ((int)v.length() > max_guess_len)
                max_guess_len = (int)v.length();
        }
    }
    
    cout << "GPU max_values: " << max_values << ", max_guess_len: " << max_guess_len << endl;
    
    // 初始化GPU缓冲区
    int total_size = max_values * (max_guess_len + 100);
    cuda_init_buffers(max_values, total_size);
}


void PriorityQueue::cleanupGPU() {
    cuda_free_all_caches();
    cuda_free_buffers();
}
