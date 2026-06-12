#include "PCFG.h"
using namespace std;


extern "C" {
    void cuda_upload_segment(const char** values_array, const int* value_lengths,
                             int num_values, int type, int seg_id);
    void cuda_free_all_caches();
    void cuda_init_buffers(int max_values, int max_output_size);
    void cuda_free_buffers();
    void cuda_generate_single_fast(int seg_type, int seg_id, char* output_buffer,
                                    const int* output_offsets, int num_values);
    void cuda_generate_multi_fast(int seg_type, int seg_id, const char* prefix, int prefix_len,
                                   char* output_buffer, const int* output_offsets, int num_values);
}


static const int BATCH_SIZE = 32;  // 每次批量处理32个PT

// 批处理条目
struct BatchItem {
    segment* seg;       // 最后一个segment指针
    int seg_type;
    int seg_id;
    string prefix;      // 多segment时的前缀
    bool is_single;     // 是否单segment
    int num_values;     // value数量
};

static vector<BatchItem> gpu_batch;

// 上传segment到GPU
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


void PriorityQueue::processGPUBatch() {
    if (gpu_batch.empty()) return;
    
    int n = (int)gpu_batch.size();
    
    // 1. 把所有segment上传到GPU
    for (int i = 0; i < n; i++) {
        ensureSegmentOnGPU(gpu_batch[i].seg, gpu_batch[i].seg_type, gpu_batch[i].seg_id);
    }
    
    // 2. 对每个PT在GPU上生成猜测（用CUDA kernel并行）
    // 注：这里用CPU循环调用GPU kernel来模拟"多PT队列GPU并行"的效果
    // 每个PT的猜测生成都在GPU上完成
    for (int i = 0; i < n; i++) {
        auto& item = gpu_batch[i];
        const vector<string>& values = item.seg->ordered_values;
        int num_vals = item.num_values;
        
        if (num_vals == 0) continue;
        
        // 计算偏移
        int prefix_len = item.is_single ? 0 : (int)item.prefix.length();
        vector<int> h_offsets(num_vals);
        int total_size = 0;
        for (int j = 0; j < num_vals; j++) {
            h_offsets[j] = total_size;
            total_size += prefix_len + (int)values[j].length() + 1;
        }
        
        // CPU分配缓冲区
        char* output_buffer = new char[total_size];
        

        if (item.is_single) {
            cuda_generate_single_fast(item.seg_type, item.seg_id,
                                       output_buffer, h_offsets.data(), num_vals);
        } else {
            cuda_generate_multi_fast(item.seg_type, item.seg_id,
                                      item.prefix.c_str(), prefix_len,
                                      output_buffer, h_offsets.data(), num_vals);
        }
        
        // 结果拷贝回guesses
        int old_size = (int)guesses.size();
        guesses.resize(old_size + num_vals);
        for (int j = 0; j < num_vals; j++) {
            guesses[old_size + j] = string(output_buffer + h_offsets[j]);
        }
        total_guesses += num_vals;
        
        delete[] output_buffer;
    }
    
    gpu_batch.clear();
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}


void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}


void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else a = &m.symbols[m.FindSymbol(pt.content[0])];

        int n = pt.max_indices[0];
        int old_size = (int)guesses.size();
        guesses.resize(old_size + n);
        for (int i = 0; i < n; i++) {
            guesses[old_size + i] = a->ordered_values[i];
        }
        total_guesses += n;
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        segment *a;
        int last = (int)pt.content.size() - 1;
        if (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else a = &m.symbols[m.FindSymbol(pt.content[last])];

        int n = pt.max_indices[last];
        int old_size = (int)guesses.size();
        guesses.resize(old_size + n);
        for (int i = 0; i < n; i++) {
            guesses[old_size + i] = guess + a->ordered_values[i];
        }
        total_guesses += n;
    }
}


void PriorityQueue::PopNext()
{
    PT& front_pt = priority.front();
    
    // 准备BatchItem
    BatchItem item;
    item.is_single = (front_pt.content.size() == 1);
    
    if (item.is_single) {
        if (front_pt.content[0].type == 1) {
            item.seg = &m.letters[m.FindLetter(front_pt.content[0])];
            item.seg_type = 1;
            item.seg_id = m.FindLetter(front_pt.content[0]);
        } else if (front_pt.content[0].type == 2) {
            item.seg = &m.digits[m.FindDigit(front_pt.content[0])];
            item.seg_type = 2;
            item.seg_id = m.FindDigit(front_pt.content[0]);
        } else {
            item.seg = &m.symbols[m.FindSymbol(front_pt.content[0])];
            item.seg_type = 3;
            item.seg_id = m.FindSymbol(front_pt.content[0]);
        }
        item.num_values = front_pt.max_indices[0];
    } else {
        // 构建前缀
        string guess;
        int seg_idx = 0;
        for (int idx : front_pt.curr_indices) {
            if (front_pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(front_pt.content[seg_idx])].ordered_values[idx];
            else if (front_pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(front_pt.content[seg_idx])].ordered_values[idx];
            else
                guess += m.symbols[m.FindSymbol(front_pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == (int)front_pt.content.size() - 1) break;
        }
        item.prefix = guess;
        
        int last = (int)front_pt.content.size() - 1;
        if (front_pt.content[last].type == 1) {
            item.seg = &m.letters[m.FindLetter(front_pt.content[last])];
            item.seg_type = 1;
            item.seg_id = m.FindLetter(front_pt.content[last]);
        } else if (front_pt.content[last].type == 2) {
            item.seg = &m.digits[m.FindDigit(front_pt.content[last])];
            item.seg_type = 2;
            item.seg_id = m.FindDigit(front_pt.content[last]);
        } else {
            item.seg = &m.symbols[m.FindSymbol(front_pt.content[last])];
            item.seg_type = 3;
            item.seg_id = m.FindSymbol(front_pt.content[last]);
        }
        item.num_values = front_pt.max_indices[last];
    }
    

    gpu_batch.push_back(item);
    

    if ((int)gpu_batch.size() >= BATCH_SIZE || priority.size() <= 1) {
        processGPUBatch();
    }


    vector<PT> new_pts = front_pt.NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}


vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;
    
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i += 1)
    {
        curr_indices[i] += 1;
        if (curr_indices[i] < max_indices[i]) {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i] -= 1;
    }
    pivot = init_pivot;
    return res;
}

void PriorityQueue::initGPU() {
    int max_values = 0;
    int max_guess_len = 0;
    for (auto& seg : m.letters) {
        if ((int)seg.ordered_values.size() > max_values) max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) if ((int)v.length() > max_guess_len) max_guess_len = (int)v.length();
    }
    for (auto& seg : m.digits) {
        if ((int)seg.ordered_values.size() > max_values) max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) if ((int)v.length() > max_guess_len) max_guess_len = (int)v.length();
    }
    for (auto& seg : m.symbols) {
        if ((int)seg.ordered_values.size() > max_values) max_values = (int)seg.ordered_values.size();
        for (auto& v : seg.ordered_values) if ((int)v.length() > max_guess_len) max_guess_len = (int)v.length();
    }
    cout << "[GPU] Batch size: " << BATCH_SIZE << ", max_values: " << max_values << endl;
    cuda_init_buffers(max_values, max_values * (max_guess_len + 100));
}

void PriorityQueue::cleanupGPU() {
    cuda_free_all_caches();
    cuda_free_buffers();
    gpu_batch.clear();
}
