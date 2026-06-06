#include "PCFG.h"
#include <pthread.h>      // 保留原头文件，但 MPI 版本不再使用 pthread
#include <vector>
#include <string>
#include <mpi.h>          // 新增 MPI 头文件
#include <cstring>        // 用于 strcpy
using namespace std;

#define THREAD_ID 8       // 保留原宏定义，MPI 版本中不再使用

void PriorityQueue::CalProb(PT& pt)
{
    // 计算 PriorityQueue 里面一个 PT 的流程如下：
    // 1. 首先需要计算一个 PT 本身的概率。例如，L6S1 的概率为 0.15
    // 2. 需要注意的是，Queue 里面的 PT 不是"纯粹的"PT，而是除了最后一个 segment 以外，全部被 value 实例化的 PT
    // 3. 所以，对于 L6S1 而言，其在 Queue 里面的实际 PT 可能是 123456S1，其中"123456"为 L6 的一个具体 value。
    // 4. 这个时候就需要计算 123456 在 L6 中出现的概率了。假设 123456 在所有 L6 segment 中的概率为 0.1，那么 123456S1 的概率就是 0.1*0.15

    // 计算一个 PT 本身的概率。后续所有具体 segment value 的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前 segment 在 PT 中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的 segment
            // m.FindLetter(seg): 找到一个 letter segment 在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个 letter segment 在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个 letter segment 在模型中，所有 value 的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 用所有可能的 PT，按概率降序填满整个优先队列
    int pt_idx = 0;
    for (PT pt : m.ordered_pts)
    {
        // 每个进程只负责 1/size 的 PT（按 rank 取模分配），避免所有进程重复做相同的队列操作
        if (pt_idx % size == rank)
        {
            for (segment seg : pt.content)
            {
                if (seg.type == 1)
                {
                    // 下面这行代码的意义：
                    // max_indices 用来表示 PT 中各个 segment 的可能数目。例如，L6S1 中，假设模型统计到了 100 个 L6，那么 L6 对应的最大下标就是 99
                    // （但由于后面采用了"<"的比较关系，所以其实 max_indices[0]=100）
                    // m.FindLetter(seg): 找到一个 letter segment 在模型中的对应下标
                    // m.letters[m.FindLetter(seg)]：一个 letter segment 在模型中对应的所有统计数据
                    // m.letters[m.FindLetter(seg)].ordered_values：一个 letter segment 在模型中，所有 value 的总数目
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
            // pt.PrintPT();
            // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

            // 计算当前 pt 的概率
            CalProb(pt);
            // 将 PT 放入优先队列
            priority.emplace_back(pt);
        }
        pt_idx++;
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的 PT，首先利用这个 PT 生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的 PT，生成一系列新的 PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的 PT 插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
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

    // 现在队首的 PT 善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新 PT
    vector<PT> res;

    // 假如这个 PT 只有一个 segment
    // 那么这个 segment 的所有 value 在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个 PT 可能对应的口令猜测已经遍历完成，无需生成新的 PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的 pivot 值。我们将更改位置下标大于等于这个 pivot 值的 segment 的值（最后一个 segment 除外），并且一次只更改一个 segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于 init_pivot 值的 segment
        // 注意 i < curr_indices.size() - 1，也就是除去了最后一个 segment（这个 segment 的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各 segment 目前的 value 在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各 segment 在模型中一共有多少个 value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新 pivot 值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解 pivot 的作用、新 PT 生成的过程而言，至关重要
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

// 以下两个函数原本供 pthread 使用，MPI 版本中不再需要，保留原样但不被调用
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

// 这个函数是 PCFG 并行化算法的主要载体
void PriorityQueue::Generate(PT pt)
{
    // 1. 计算 PT 的概率（用于排序，不影响生成逻辑）
    CalProb(pt);

    // 2. 确定最后一个 segment 的指针 a 和需要生成的总数 total
    segment* a = nullptr;
    int total = 0;
    string prefix;  // 最后一个 segment 之前的部分（仅多 segment 时有效）

    if (pt.content.size() == 1)
    {
        // 只有一个 segment：没有前缀
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
        // 多个 segment：先拼接前缀（最后一个 segment 不拼）
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
        // 最后一个 segment
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        total = pt.max_indices.back();
    }

    if (total <= 0) return;

    // 每个 PT 只由一个进程负责（init 已按 rank 分配），直接生成全部猜测，无需索引划分
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