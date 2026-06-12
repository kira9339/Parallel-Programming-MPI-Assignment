#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
using namespace std;
using namespace chrono;

int main()
{
    // 下面代码用于测试MD5哈希的正确性
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };
    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1) {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    q.initGPU();
    cout << "here" << endl;


    mutex hash_mutex;
    condition_variable hash_cv;
    queue<vector<string>> hash_queue;
    atomic<bool> hash_done{false};

    //  哈希线程：从队列中取一批猜测，用CPU做MD5哈希（同时利用OpenMP多核加速）
    thread hasher_thread([&]() {
        while (true) {
            vector<string> batch;
            {
                unique_lock<mutex> lock(hash_mutex);
                hash_cv.wait(lock, [&]() { return !hash_queue.empty() || hash_done; });
                if (hash_queue.empty() && hash_done) break;
                batch = move(hash_queue.front());
                hash_queue.pop();
            }
            // 使用OpenMP多核并行哈希，充分利用所有CPU核心
            auto start_hash = system_clock::now();
            #pragma omp parallel for
            for (int i = 0; i < (int)batch.size(); i++) {
                bit32 s[4];
                MD5Hash(batch[i], s);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        }
    });

    int curr_num = 0;
    auto start = system_clock::now();
    long long history = 0;

    while (!q.priority.empty())
    {
        // 主线程：让GPU生成猜测
        q.PopNext();
        int total_in_buffer = (int)q.guesses.size();

        if (total_in_buffer - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + total_in_buffer << endl;
            curr_num = total_in_buffer;

            if (history + total_in_buffer > 20000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                hash_done = true;
                hash_cv.notify_all();
                break;
            }
        }

        // 累积了足够多的猜测，移交到哈希队列（不阻塞主线程的GPU生成）
        if (total_in_buffer >= 1000000)
        {
            {
                lock_guard<mutex> lock(hash_mutex);
                hash_queue.push(move(q.guesses));  
                q.guesses.clear();
            }
            hash_cv.notify_one();

            history += total_in_buffer;
            curr_num = 0;
        }
    }

    // 等待哈希线程结束
    hash_done = true;
    hash_cv.notify_all();
    hasher_thread.join();

    // 处理剩余未哈希的猜测
    if (!q.guesses.empty()) {
        auto start_hash = system_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < (int)q.guesses.size(); i++) {
            bit32 s[4];
            MD5Hash(q.guesses[i], s);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }

    q.cleanupGPU();
}
