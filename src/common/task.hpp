#ifndef SRC_COMMON_DETAIL_TASK_HPP_
#define SRC_COMMON_DETAIL_TASK_HPP_

#include <cstdint>
#include <functional>
#include <string>

namespace kb {
namespace viz2d {
class Viz2DWorker;
class TaskPool;
class Task {
public:
    std::string name_;
    uint64_t frameIdx_ = 0;
    bool mutex_ = false;
    std::function<void()> func_;
    Viz2DWorker* owner_ = nullptr;
    int32_t id_ = -1;

    Task(){
    }
    Task(std::string name, uint64_t frameIdx, bool mutex, std::function<void()> func, Viz2DWorker* owner) :
    name_(name), frameIdx_(frameIdx), mutex_(mutex), func_(func), owner_(owner){
    }

    void operator()() {
        func_();
    }
};
}
}

#endif /* SRC_COMMON_DETAIL_TASK_HPP_ */
