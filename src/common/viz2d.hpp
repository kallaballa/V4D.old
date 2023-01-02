#ifndef SRC_COMMON_VIZ2D_HPP_
#define SRC_COMMON_VIZ2D_HPP_

#include "viz2dworker.hpp"
#include "detail/taskpool.hpp"


namespace kb {
namespace viz2d {
namespace detail {
    class TaskPool;
}
class Viz2D : public Viz2DWorker {
    const size_t numWorkers_;
    kb::viz2d::detail::TaskPool taskPool_;
    std::vector<cv::Ptr<kb::viz2d::Viz2DWorker>> workers_;
    cv::Ptr<kb::viz2d::Viz2DWorker> source_;
public:
    Viz2D(const size_t& numWorkers, const cv::Size &initialSize, const cv::Size& frameBufferSize, bool offscreen, const string &title, int major = 4, int minor = 6, int samples = 0, bool debug = false);
    virtual ~Viz2D();
    void prepare(std::function<std::vector<Task>(Viz2DWorker&)> plan);
    void work();
    virtual cv::VideoWriter& makeVAWriter(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex) override;
    virtual cv::VideoCapture& makeVACapture(const string &inputFilename, const int vaDeviceIndex) override;
    virtual cv::VideoWriter& makeWriter(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize) override;
    virtual cv::VideoCapture& makeCapture(const string &inputFilename) override;
    template <typename T>
    void propagate(const string& name, const T& value, double scale = 1) {
        storage().properties().propagate<T>(name, value, scale);
        for(size_t i = 0; i < numWorkers_; ++i) {
            workers_[i]->storage().properties().propagate<T>(name, value, scale);
        }
    }
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_VIZ2D_HPP_ */
