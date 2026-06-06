#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
using namespace std;


vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; ++i) {
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


void PriorityQueue::CalProb(PT& pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}


void PriorityQueue::init() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int pt_idx = 0;
    for (PT pt : m.ordered_pts) {
        if (pt_idx % size == rank) {
            for (segment seg : pt.content) {
                if (seg.type == 1)
                    pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
                else if (seg.type == 2)
                    pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
                else
                    pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
            pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
            CalProb(pt);
            priority.emplace_back(pt);
        }
        pt_idx++;
    }
}

void PriorityQueue::Generate(PT pt) {
    CalProb(pt);  

    if (pt.content.size() == 1) {
        segment* a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        for (int i = 0; i < pt.max_indices[0]; ++i) {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses++;
        }
    }
    else {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        segment* a = nullptr;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else
            a = &m.symbols[m.FindSymbol(pt.content.back())];

        for (int i = 0; i < pt.max_indices.back(); ++i) {
            guesses.emplace_back(prefix + a->ordered_values[i]);
            total_guesses++;
        }
    }
}

void PriorityQueue::PopNext() {
    const int BATCH_SIZE = 5;   
    int n = std::min(BATCH_SIZE, (int)priority.size());
    if (n == 0) return;

    // 1. 取出前 n 个 PT
    vector<PT> batch;
    batch.reserve(n);
    for (int i = 0; i < n; ++i)
        batch.push_back(priority[i]);
    priority.erase(priority.begin(), priority.begin() + n);

    // 2. 处理这批 PT：生成猜测，并收集所有新 PT
    vector<PT> all_new_pts;
    for (PT& pt : batch) {
        Generate(pt);                      // 生成猜测
        vector<PT> new_pts = pt.NewPTs();  // 生成派生 PT
        for (PT& npt : new_pts) {
            CalProb(npt);                  // 计算派生 PT 的概率
            all_new_pts.push_back(npt);
        }
    }

    // 3. 将所有新 PT 按概率降序插入原队列（保证有序）
    for (PT& npt : all_new_pts) {
        bool inserted = false;
        for (auto it = priority.begin(); it != priority.end(); ++it) {
            if (npt.prob >= it->prob) {
                priority.emplace(it, npt);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            priority.emplace_back(npt);
        }
    }
}