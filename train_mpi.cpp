
#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <omp.h> 
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <mpi.h>          // 新增 MPI 头文件
#include <cstring>        // 用于 memcpy
using namespace std;
// 或者保留 std:: 前缀，但加上 using 更简单
// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 *
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 *
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 *
 */

 // 辅助函数：合并 unordered_map<int, int>
void merge_freq_map(unordered_map<int, int>& local_map, int rank, int size)
{
    vector<int> local_buf;
    for (auto& kv : local_map) {
        local_buf.push_back(kv.first);
        local_buf.push_back(kv.second);
    }
    int local_count = local_buf.size();

    vector<int> all_counts(size, 0);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> global_buf;
    vector<int> displs(size, 0);
    int total = 0;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) { displs[p] = total; total += all_counts[p]; }
        global_buf.resize(total);
    }
    MPI_Gatherv(local_buf.data(), local_count, MPI_INT,
        global_buf.data(), all_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        local_map.clear();
        for (int p = 0; p < size; ++p) {
            int* ptr = global_buf.data() + displs[p];
            int n = all_counts[p] / 2;
            for (int i = 0; i < n; ++i) local_map[ptr[2 * i]] += ptr[2 * i + 1];
        }
    }

    vector<int> merged_buf;
    int merged_count = 0;
    if (rank == 0) {
        for (auto& kv : local_map) { merged_buf.push_back(kv.first); merged_buf.push_back(kv.second); }
        merged_count = merged_buf.size();
    }
    MPI_Bcast(&merged_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) merged_buf.resize(merged_count);
    MPI_Bcast(merged_buf.data(), merged_count, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        local_map.clear();
        for (int i = 0; i < merged_count; i += 2) local_map[merged_buf[i]] = merged_buf[i + 1];
    }
}

// 辅助函数：按 length 合并 segments（一次性收发，不循环广播）
// seg_type: 1=letters, 2=digits, 3=symbols
void merge_segments_by_length(vector<segment>& segs, unordered_map<int, int>& seg_freqs, int seg_type, int rank, int size)
{
    // 序列化本进程 segments（包含 type, length, freq, values 和 freqs）
    // 格式：[n_segs][len1][freq1][n_vals1][strlen1][str1][val_freq1]...[len2]...
    vector<char> local_buf;
    int n = segs.size();
    local_buf.insert(local_buf.end(), (char*)&n, (char*)&n + 4);
    for (int i = 0; i < n; ++i) {
        int len = segs[i].length;
        int freq = (seg_freqs.find(i) != seg_freqs.end()) ? seg_freqs[i] : 0;
        local_buf.insert(local_buf.end(), (char*)&len, (char*)&len + 4);
        local_buf.insert(local_buf.end(), (char*)&freq, (char*)&freq + 4);
        int nv = segs[i].values.size();
        local_buf.insert(local_buf.end(), (char*)&nv, (char*)&nv + 4);
        for (auto& kv : segs[i].values) {
            int sl = kv.first.size();
            int f = segs[i].freqs[kv.second];
            local_buf.insert(local_buf.end(), (char*)&sl, (char*)&sl + 4);
            local_buf.insert(local_buf.end(), kv.first.begin(), kv.first.end());
            local_buf.insert(local_buf.end(), (char*)&f, (char*)&f + 4);
        }
    }

    // 收集到 rank 0
    int local_sz = local_buf.size();
    vector<int> all_sz(size, 0);
    MPI_Gather(&local_sz, 1, MPI_INT, all_sz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<char> all_data;
    vector<int> displ(size, 0);
    if (rank == 0) {
        int tot = 0;
        for (int i = 0; i < size; ++i) { displ[i] = tot; tot += all_sz[i]; }
        all_data.resize(tot);
    }
    MPI_Gatherv(local_buf.data(), local_sz, MPI_CHAR,
        all_data.data(), all_sz.data(), displ.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    // rank 0 按 length 合并
    // 用 vector 按 length 排序存储，确保广播后所有进程顺序一致
    unordered_map<int, pair<int, unordered_map<string, int>>> merged; // length -> {total_freq, {value->freq}}
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            char* ptr = all_data.data() + displ[p];
            int rem = all_sz[p];
            if (rem < 4) continue;
            int pn; memcpy(&pn, ptr, 4); ptr += 4; rem -= 4;
            for (int s = 0; s < pn; ++s) {
                if (rem < 12) break;
                int len; memcpy(&len, ptr, 4); ptr += 4; rem -= 4;
                int freq; memcpy(&freq, ptr, 4); ptr += 4; rem -= 4;
                int nv; memcpy(&nv, ptr, 4); ptr += 4; rem -= 4;
                merged[len].first += freq;
                for (int v = 0; v < nv; ++v) {
                    if (rem < 4) break;
                    int sl; memcpy(&sl, ptr, 4); ptr += 4; rem -= 4;
                    if (rem < sl + 4) break;
                    string val(ptr, sl); ptr += sl; rem -= sl;
                    int vf; memcpy(&vf, ptr, 4); ptr += 4; rem -= 4;
                    merged[len].second[val] += vf;
                }
            }
        }
    }

    // 序列化合并结果（按 length 升序，保证所有进程一致）
    vector<char> merged_buf;
    int n_merged = 0;
    if (rank == 0) {
        // 收集并排序 length
        vector<int> sorted_lens;
        for (auto& kv : merged) sorted_lens.push_back(kv.first);
        sort(sorted_lens.begin(), sorted_lens.end());

        n_merged = sorted_lens.size();
        merged_buf.insert(merged_buf.end(), (char*)&n_merged, (char*)&n_merged + 4);

        for (int len : sorted_lens) {
            int total_freq = merged[len].first;
            auto& val_freq = merged[len].second;
            int nv = val_freq.size();
            merged_buf.insert(merged_buf.end(), (char*)&len, (char*)&len + 4);
            merged_buf.insert(merged_buf.end(), (char*)&total_freq, (char*)&total_freq + 4);
            merged_buf.insert(merged_buf.end(), (char*)&nv, (char*)&nv + 4);

            // 按 value 排序确保确定性
            vector<pair<string, int>> sorted_vf(val_freq.begin(), val_freq.end());
            for (auto& vf : sorted_vf) {
                int sl = vf.first.size();
                int f = vf.second;
                merged_buf.insert(merged_buf.end(), (char*)&sl, (char*)&sl + 4);
                merged_buf.insert(merged_buf.end(), vf.first.begin(), vf.first.end());
                merged_buf.insert(merged_buf.end(), (char*)&f, (char*)&f + 4);
            }
        }
    }

    // 只广播一次！
    int merged_sz = merged_buf.size();
    MPI_Bcast(&merged_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) merged_buf.resize(merged_sz);
    MPI_Bcast(merged_buf.data(), merged_sz, MPI_CHAR, 0, MPI_COMM_WORLD);

    // 所有进程重建 segs 和 seg_freqs
    segs.clear();
    seg_freqs.clear();

    char* ptr = merged_buf.data();
    int rem = merged_sz;
    if (rem < 4) return;
    int mn; memcpy(&mn, ptr, 4); ptr += 4; rem -= 4;

    for (int i = 0; i < mn; ++i) {
        if (rem < 12) break;
        int len; memcpy(&len, ptr, 4); ptr += 4; rem -= 4;
        int total_freq; memcpy(&total_freq, ptr, 4); ptr += 4; rem -= 4;
        int nv; memcpy(&nv, ptr, 4); ptr += 4; rem -= 4;

        segment new_seg(seg_type, len);
        unordered_map<string, int> vm;
        unordered_map<int, int> fm;
        for (int v = 0; v < nv; ++v) {
            if (rem < 4) break;
            int sl; memcpy(&sl, ptr, 4); ptr += 4; rem -= 4;
            if (rem < sl + 4) break;
            string val(ptr, sl); ptr += sl; rem -= sl;
            int vf; memcpy(&vf, ptr, 4); ptr += 4; rem -= 4;
            vm[val] = vm.size();
            fm[vm[val]] = vf;
        }
        new_seg.values = vm;
        new_seg.freqs = fm;
        segs.push_back(new_seg);
        seg_freqs[i] = total_freq;
    }
}

// 训练的wrapper，实际上就是读取训练集
void model::train(string path)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string pw;
    ifstream train_set(path);
    int lines = 0;
    if (rank == 0) {
        cout << "Training..." << endl;
        cout << "Training phase 1: reading and parsing passwords..." << endl;
    }
    while (train_set >> pw)
    {
        lines += 1;
        if (lines % 10000 == 0)
        {
            cout << "Process " << rank << ": Lines processed: " << lines << endl;
            // 在这里更改读取的训练集口令上限
            if (lines > 3000000)
            {
                break;
            }
        }
        // MPI 并行：按行号轮转分配，每个进程只处理 1/size 的口令
        if ((lines - 1) % size == rank)
        {
            // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
            parse(pw);
        }
    }
    train_set.close();

    // 合并各进程的统计结果
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << "Merging statistical data..." << endl;

    // 1. 合并 total_preterm
    long long lpt = total_preterm;
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &lpt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        total_preterm = (int)lpt;
    }
    else {
        MPI_Reduce(&lpt, &lpt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&total_preterm, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. 合并 preterm_freq
    merge_freq_map(preterm_freq, rank, size);

    // 3-5. 合并 letters/digits/symbols（按 length 合并，type 字段正确设置）
    merge_segments_by_length(letters, letters_freq, 1, rank, size);
    merge_segments_by_length(digits, digits_freq, 2, rank, size);
    merge_segments_by_length(symbols, symbols_freq, 3, rank, size);
}








// #include "PCFG.h"
// #include <fstream>
// #include <cctype>
// #include <algorithm>

// // 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// // 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// // 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// // 所以如果你愿意，也可以尝试用多线程加速训练过程

// /**
//  * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
//  * 
//  * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
//  * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
//  * 
//  * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
//  * 
//  */

// // 训练的wrapper，实际上就是读取训练集
// void model::train(string path)
// {
//     string pw;
//     ifstream train_set(path);
//     int lines = 0;
//     cout<<"Training..."<<endl;
//     cout<<"Training phase 1: reading and parsing passwords..."<<endl;
//     while (train_set >> pw)
//     {
//         lines += 1;
//         if (lines % 10000 == 0)
//         {
//             cout <<"Lines processed: "<< lines << endl;
//             // 在这里更改读取的训练集口令上限
//             if (lines > 3000000)
//             {
//                 break;
//             }
//         }
//         // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
//         parse(pw);
//     }
// }

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(PT pt)
{
    for (int id = 0; id < preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }
        else
        {
            bool equal_flag = true;
            for (int idx = 0; idx < preterminals[id].content.size(); idx += 1)
            {
                if (preterminals[id].content[idx].type != pt.content[idx].type || preterminals[id].content[idx].length != pt.content[idx].length)
                {
                    equal_flag = false;
                    break;
                }
            }
            if (equal_flag == true)
            {
                return id;
            }
        }
    }
    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(segment seg)
{
    for (int id = 0; id < letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(segment seg)
{
    for (int id = 0; id < digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}


void segment::order()
{
    for (pair<string, int> value : values)
    {
        ordered_values.emplace_back(value.first);
    }
    // cout << "value size:" << ordered_values.size() << endl;
    std::sort(ordered_values.begin(), ordered_values.end(),
        [this](const std::string& a, const std::string& b)
        {
            return freqs.at(values[a]) > freqs.at(values[b]);
        });

    // 将排序后的频率存入 ordered_freqs 并计算 total_freq
    for (const std::string& val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
    for (string val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }
        int id = GetNextPretermID();
        // cout << id << endl;
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        // cout << id << endl;
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT& a, const PT& b) {
    return a.preterm_prob > b.preterm_prob;  // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    bool swapped;
    cout << "total pts" << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        // cout << i << endl;
        letters[i].order();
    }
    cout << "Ordering digits" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].order();
    }
    cout << "ordering symbols" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].order();
    }
}