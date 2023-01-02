//started with https://stackoverflow.com/a/51400041/1884837
#ifndef SRC_COMMON_TASKPOOL_HPP_
#define SRC_COMMON_TASKPOOL_HPP_

#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>
#include <thread>

#include "../task.hpp"

namespace kb {
namespace viz2d {
namespace detail {

class TaskPool {
private:
    std::vector<std::vector<Task>> collected_;
    std::queue<Task> taskQueue_;
    std::mutex dataLock;
    std::mutex runningLock;
    std::condition_variable dataCondition_;
    std::condition_variable runningCondition_;
    std::atomic<bool> acceptFunctions_;
    std::vector<Task> schedule_;
    std::vector<std::thread> threadPool_;
    size_t running_ = 0;
    static std::mutex id_mtx_;
    static int32_t max_id_;
public:

    TaskPool(const size_t& numWorkers = std::thread::hardware_concurrency());
    ~TaskPool();
    void addTasks(const std::vector<Task>& collected);
    void prepare();
    void run();
    void finish();
    void join();
    void infinite_loop_func();
private:
    void push(const Task& func);
};

} /* namespace detail */
} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_TASKPOOL_HPP_ */
