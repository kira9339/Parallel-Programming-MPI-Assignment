#include "PCFG.h"
#include <pthread.h>    
#include <vector>
#include <string>
#include <mpi.h>          
#include <cstring>      
using namespace std;

#define THREAD_ID 8      

void PriorityQueue::CalProb(PT& pt)
{

    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init(int num_gen)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pt_idx = 0;
    for (PT pt : m.ordered_pts)
    {
        // 只取属于本生成器的 PT（按 num_gen 取模）
        if (pt_idx % num_gen == rank)
        {
            for (segment seg : pt.content)
            {
                if (seg.type == 1)
                {
                    pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
                }
                if (seg.type == 2)
                {
                    pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
                }
                if (seg.type == 3)
                {
                    pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
                }
            }
            pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
            CalProb(pt);
            priority.emplace_back(pt);
        }
        pt_idx++;
    }
}

void PriorityQueue::PopNext()
{

    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
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
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}

typedef struct {
    int start;
    int end;
    int base;
    const segment* st;
    string prefix;
    const segment* last_st;
    PriorityQueue* q;
} ThreadArg;

inline void* SingleThread(void* arg) {
    ThreadArg* p = (ThreadArg*)arg;
    const vector<string>& values = p->st->ordered_values;
    vector<string>& guesses = p->q->guesses;
    int base = p->base;
    int start = p->start;
    int end = p->end;
    for (int i = start; i < end; ++i) {
        guesses[base + (i - start)] = values[i];
    }
    return nullptr;
}

inline void* MultiThread(void* arg) {
    ThreadArg* p = (ThreadArg*)arg;
    const string& prefix = p->prefix;
    const vector<string>& values = p->last_st->ordered_values;
    vector<string>& guesses = p->q->guesses;
    int base = p->base;
    int start = p->start;
    int end = p->end;
    for (int i = start; i < end; ++i) {
        guesses[base + (i - start)] = prefix + values[i];
    }
    return nullptr;
}

void PriorityQueue::Generate(PT pt)
{

    CalProb(pt);
    segment* a = nullptr;
    int total = 0;
    string prefix;
    if (pt.content.size() == 1)
    {
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        total = pt.max_indices[0];
    }
    else
    {
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        total = pt.max_indices.back();
    }
    if (total <= 0) return;
    const auto& values = a->ordered_values;
    if (pt.content.size() == 1)
    {
        guesses.reserve(guesses.size() + total);
        for (int i = 0; i < total; ++i)
            guesses.push_back(values[i]);
    }
    else
    {
        guesses.reserve(guesses.size() + total);
        for (int i = 0; i < total; ++i)
            guesses.push_back(prefix + values[i]);
    }
}