#include "taskpool.hpp"
#include "../../ext/timetracker.hpp"

#include <iostream>
#include "../viz2dworker.hpp"

namespace kb {
namespace viz2d {
namespace detail {

std::mutex TaskPool::id_mtx_;
int32_t TaskPool::max_id_ = 0;

TaskPool::TaskPool(const size_t& numWorkers) :
        taskQueue_(), dataLock(), dataCondition_(), runningCondition_(), acceptFunctions_(true), threadPool_(numWorkers) {
    for(size_t i = 0; i < numWorkers; ++i) {
        threadPool_[i] = std::thread(&TaskPool::infinite_loop_func, this);
    }
}

TaskPool::~TaskPool() {
}

void TaskPool::addTasks(const std::vector<Task>& tasks) {
    collected_.push_back(tasks);
}

void TaskPool::prepare() {
    schedule_.clear();

    for(size_t i = 0; i < collected_.size(); ++i) {
        assert(collected_[0].size() == collected_[i].size());
    }

    for(size_t i = 0; i < collected_[0].size(); ++i) {
        for(size_t j = 0; j < collected_.size(); ++j) {
            collected_[j][i].id_ = j;
            schedule_.push_back(collected_[j][i]);
        }
    }
}

void TaskPool::run() {
    for(const auto& t : schedule_) {
        push(t);
    }
}

void TaskPool::push(const Task& t) {
    std::unique_lock<std::mutex> lock(dataLock);
    taskQueue_.push(t);
    // when we send the notification immediately, the consumer will try to get the lock , so unlock asap
    lock.unlock();
    dataCondition_.notify_one();
}

void TaskPool::finish() {
    std::unique_lock<std::mutex> lock(dataLock);
    acceptFunctions_ = false;
    lock.unlock();
    // when we send the notification immediately, the consumer will try to get the lock , so unlock asap
    dataCondition_.notify_all();
    //notify all waiting threads.
}

void TaskPool::join() {
    std::unique_lock<std::mutex> lock(dataLock);
    runningCondition_.wait(lock, [this]() {
        return taskQueue_.empty() && running_ == 0;
    });
}

void TaskPool::infinite_loop_func() {
    Task task;
    int id;
    {
        std::unique_lock lock(id_mtx_);
        id = max_id_++;
    }
    while (true) {
        try {
            {
                std::unique_lock<std::mutex> lock(dataLock);
                dataCondition_.wait(lock, [this]() {
                    return !taskQueue_.empty() || !acceptFunctions_;
                });
                if (!acceptFunctions_ && taskQueue_.empty()) {
                    //lock will be release automatically.
                    //finish the thread loop and let it join in the main thread.
                    return;
                }
                task = taskQueue_.front();
                if(task.id_ != id) {
                    lock.unlock();
                    dataCondition_.notify_one();
                    continue;
                }
                taskQueue_.pop();
                ++running_;
                if(task.mutex_) {
                    task.owner_->makeCurrent();
                    TimeTracker::getInstance()->execute(task.name_, task.func_);
                    task.owner_->makeNonCurrent();
                    --running_;
                }
            }
            if(!task.mutex_) {
                TimeTracker::getInstance()->execute(task.name_, task.func_);
                --running_;
            }
        } catch(std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
        runningCondition_.notify_all();
    }
}

} /* namespace detail */
} /* namespace viz2d */
} /* namespace kb */
