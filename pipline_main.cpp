#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 打包
vector<char> pack_batch(const vector<string>& batch, int& total_size) {
    int n = batch.size();

    vector<int> lens(n);
    int total_chars = 0;
    for (int i = 0; i < n; i++) {
        lens[i] = batch[i].size();
        total_chars += lens[i];
    }
    int header_size = 4 + n * 4; // n + lens
    total_size = header_size + total_chars;

    vector<char> buf(total_size);
    memcpy(buf.data(), &n, 4);
    int offset = 4;
    for (int i = 0; i < n; i++) {
        memcpy(buf.data() + offset, &lens[i], 4);
        offset += 4;
    }
    for (int i = 0; i < n; i++) {
        memcpy(buf.data() + offset, batch[i].c_str(), lens[i]);
        offset += lens[i];
    }
    return buf;
}

// 从 buffer 解包一批猜测
vector<string> unpack_batch(const vector<char>& buf) {
    int n;
    memcpy(&n, buf.data(), 4);
    vector<int> lens(n);
    int offset = 4;
    for (int i = 0; i < n; i++) {
        memcpy(&lens[i], buf.data() + offset, 4);
        offset += 4;
    }
    vector<string> batch(n);
    for (int i = 0; i < n; i++) {
        batch[i].assign(buf.data() + offset, lens[i]);
        offset += lens[i];
    }
    return batch;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int NUM_GEN = 4;
    const int BATCH_SIZE = 1000000;
    const bool generator = (rank < NUM_GEN);
    const bool hasher = (rank >= NUM_GEN);
    const int NUM_HASH = size - NUM_GEN;


    // MD5 正确性测试
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
        cout << "MD5Hash test passed!" << endl;
    }

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    int cracked = 0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 所有进程加载测试集
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw && test_count < 1000000) {
        test_set.insert(pw);
        test_count++;
    }
    test_data.close();


    if (generator)
    {
        q.init(NUM_GEN);
        if (rank == 0) cout << "here" << endl;

        auto start = system_clock::now();

        long long history = 0;
        const long long LOCAL_LIMIT = 10000000LL / NUM_GEN;

        // 每个生成器固定发送给一个哈希器
        int hasher_id = NUM_GEN + (rank % NUM_HASH);

        vector<string> pending;

        vector<MPI_Request> pending_requests;
        vector<vector<char>> pending_buffers; // 保持 buffer 存活直到发送完成

        // 检查并清理已完成的非阻塞发送
        auto cleanup_sends = [&]() {
            for (int i = pending_requests.size() - 1; i >= 0; i--) {
                int flag = 0;
                MPI_Test(&pending_requests[i], &flag, MPI_STATUS_IGNORE);
                if (flag) {
                    // 发送完成，移除
                    pending_requests.erase(pending_requests.begin() + i);
                    pending_buffers.erase(pending_buffers.begin() + i);
                }
            }
            };

        while (!q.priority.empty())
        {
            q.PopNext();
            if (!q.guesses.empty()) {
                pending.insert(pending.end(), q.guesses.begin(), q.guesses.end());
                q.guesses.clear();
            }

            // 当 pending 足够多，或队列快空时，打包发送
            bool need_send = (pending.size() >= (size_t)BATCH_SIZE);
            bool final_send = (q.priority.empty() && !pending.empty());

            while (need_send || final_send)
            {
                int to_send_n = min((int)pending.size(), BATCH_SIZE);
                vector<string> to_send(pending.begin(), pending.begin() + to_send_n);
                pending.erase(pending.begin(), pending.begin() + to_send_n);

                // 打包并发送
                int buf_size;
                vector<char> buf = pack_batch(to_send, buf_size);

                // 使用非阻塞发送，永不阻塞
                MPI_Request req;
                MPI_Isend(buf.data(), buf_size, MPI_CHAR, hasher_id, 0, MPI_COMM_WORLD, &req);

                // 保存请求和 buffer（buffer 必须在发送完成前保持有效）
                pending_requests.push_back(req);
                pending_buffers.push_back(std::move(buf));

                history += to_send_n;
                if (history % 100000 == 0 && rank == 0) {
                    cout << "Guesses generated: " << history << endl;
                }

                // 清理已完成的发送
                cleanup_sends();

                need_send = (pending.size() >= (size_t)BATCH_SIZE);
                final_send = (q.priority.empty() && !pending.empty());

                if (history > LOCAL_LIMIT) break;
            }

            if (history > LOCAL_LIMIT) break;
        }

        // 等待所有未完成的发送完成
        for (size_t i = 0; i < pending_requests.size(); i++) {
            MPI_Wait(&pending_requests[i], MPI_STATUS_IGNORE);
        }
        pending_requests.clear();
        pending_buffers.clear();

        // 通知哈希器：本生成器结束
        int done_signal = -1;
        MPI_Send(&done_signal, 1, MPI_INT, hasher_id, 1, MPI_COMM_WORLD);

        // 等待所有哈希器
        MPI_Barrier(MPI_COMM_WORLD);


        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        double total_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;

        // 汇总破解数
        int global_cracked = 0;
        MPI_Reduce(&cracked, &global_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // 取所有哈希器中最大的哈希耗时
        double global_hash_time = 0;
        MPI_Reduce(&time_hash, &global_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        time_hash = global_hash_time;

        if (rank == 0) {
            // 流水线中生成和哈希重叠，所以：
            // total_time = 实际总耗时
            // time_hash = 哈希耗时
            // "Guess time" ≈ total_time（因为重叠，无法精确分离）
            cout << "Guess time:" << total_time << "seconds" << endl;   // ← 输出总耗时
            cout << "Hash time:" << time_hash << "seconds" << endl;
            cout << "Train time:" << time_train << "seconds" << endl;
            cout << "Cracked:" << global_cracked << endl;
        }

    }

    if (hasher)
    {

        int my_gen_count = 0;
        for (int g = 0; g < NUM_GEN; g++) {
            if (NUM_GEN + (g % NUM_HASH) == rank) {
                my_gen_count++;
            }
        }
        int active_generators = my_gen_count;
        int local_cracked = 0;
        double local_hash_time = 0;

        while (active_generators > 0)
        {
            MPI_Status status;
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (flag)
            {
                if (status.MPI_TAG == 1)
                {

                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    active_generators--;
                }
                else if (status.MPI_TAG == 0)
                {

                    int buf_size;
                    MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_CHAR, &buf_size);

                    vector<char> buf(buf_size);
                    MPI_Recv(buf.data(), buf_size, MPI_CHAR, status.MPI_SOURCE, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    vector<string> batch = unpack_batch(buf);

                    // 哈希
                    auto start_hash = system_clock::now();
                    bit32 state[4];
                    for (const string& pwd : batch)
                    {
                        if (test_set.find(pwd) != test_set.end()) {
                            local_cracked++;
                        }
                        MD5Hash(pwd, state);
                    }
                    auto end_hash = system_clock::now();
                    auto duration = duration_cast<microseconds>(end_hash - start_hash);
                    local_hash_time += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                }
            }
        }

        cracked = local_cracked;
        time_hash = local_hash_time;

        MPI_Barrier(MPI_COMM_WORLD);

        int global_cracked = 0;
        MPI_Reduce(&cracked, &global_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        double global_hash_time = 0;

        MPI_Reduce(&time_hash, &global_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
