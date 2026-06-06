#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>          // 新增 MPI 头文件
using namespace std;
using namespace chrono;

// 编译指令如下
// mpicxx main.cpp train.cpp guessing.cpp md5.cpp -o main
// mpicxx main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// mpicxx main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// mpirun -np 8 ./main

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //下面代码用于测试MD5哈希的正确性
    if (rank == 0) {
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = { "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111" };
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
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        cout << "MD5Hash test passed!" << endl; //请不要修改这一行
    }

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 所有进程加载测试集（用于验证猜测是否正确）
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw && test_count < 1000000) {
        test_set.insert(pw);
        test_count++;
    }
    test_data.close();

    q.init();
    if (rank == 0) cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    int cracked = 0;  // 本地猜中的口令数
    // std::ofstream a("./files/results.txt");

    // 每个进程独立管理 1/size 的 PT，独立生成所有猜测，无需任何 MPI 通信！
    // 所有进程队列状态相同，Generate 中各进程负责不同的索引范围
    // 每个进程追踪本地猜测总数，达到 (总上限 / 进程数) 时退出
    const long long LOCAL_LIMIT = 10000000LL / size;

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            if (rank == 0) {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
            }
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > LOCAL_LIMIT)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                // 汇总所有进程的 crack 数
                int global_cracked = 0;
                MPI_Reduce(&cracked, &global_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;//请不要修改这一行
                    cout << "Hash time:" << time_hash << "seconds" << endl;//请不要修改这一行
                    cout << "Train time:" << time_train << "seconds" << endl;//请不要修改这一行
                    cout << "Cracked:" << global_cracked << endl;
                }
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pwd : q.guesses)
            {
                // 在哈希循环中顺便做 crack 统计（一次遍历完成两件事，避免额外循环）
                if (test_set.find(pwd) != test_set.end()) {
                    cracked++;
                }

                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pwd, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    MPI_Finalize();
    return 0;
}