#include "viz2d.hpp"
#include "detail/taskpool.hpp"
#include "../ext/timetracker.hpp"

namespace kb {
namespace viz2d {

Viz2D::Viz2D(const size_t& numWorkers, const cv::Size &size, const cv::Size& frameBufferSize, bool offscreen, const string &title, int major, int minor, int samples, bool debug)
        : Viz2DWorker(size, frameBufferSize, offscreen, title, major, minor, samples, debug),
          numWorkers_(numWorkers),
          taskPool_(numWorkers),
          workers_(numWorkers),
          source_(new kb::viz2d::Viz2DWorker(size, frameBufferSize, true, "source")){
    for(size_t i = 0; i < numWorkers_; ++i) {
        //FIXME shared gl context?
        workers_[i] = new kb::viz2d::Viz2DWorker(size, frameBufferSize, true, std::to_string(i));
    }
}

Viz2D::~Viz2D() {
}

void Viz2D::prepare(std::function<std::vector<Task>(Viz2DWorker&)> plan) {
    for(size_t i = 0; i < numWorkers_; ++i)
        taskPool_.addTasks(plan(*workers_[i]));

    taskPool_.prepare();
}

void Viz2D::work() {
    cv::UMat frame;
    for (size_t i = 0; i < numWorkers_; ++i) {
        workers_[i]->properties() = this->properties();

        if (source_->hasCapture()) {
            source_->makeCurrent();
            {
                if (!source_->capture())
                    exit(0);

                source_->write([&](const cv::UMat &videoFrame) {
                    videoFrame.copyTo(frame);
                });
            }
            source_->makeNonCurrent();

            workers_[i]->makeCurrent();
            {
                workers_[i]->capture([&](cv::UMat &videoFrame) {
                    frame.copyTo(videoFrame);
                });
            }
            workers_[i]->makeNonCurrent();
        }
    }
    source_->makeNonCurrent();


    taskPool_.run();
    taskPool_.join();
    if(this->hasWriter()) {
        for(size_t i = 0; i < numWorkers_; ++i) {
            workers_[i]->makeCurrent();
            {
                workers_[i]->write([&](const cv::UMat& videoFrame){
                    videoFrame.copyTo(frame);
                });
            }
            workers_[i]->makeNonCurrent();

            this->makeCurrent();
            {
                this->capture([&](cv::UMat& videoFrame){
                    frame.copyTo(videoFrame);
                });

                this->write();

                if(!this->display())
                    exit(0);
            }
            this->makeNonCurrent();
        }
    }
    TimeTracker::getInstance()->print(std::cerr);
    if(this->getFrameCount() % 10 == 0)
        TimeTracker::getInstance()->newCount();
}

cv::VideoWriter& Viz2D::makeVAWriter(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex) {
    return Viz2DWorker::makeVAWriter(outputFilename, fourcc, fps, frameSize, vaDeviceIndex);
}

cv::VideoCapture& Viz2D::makeVACapture(const string &inputFilename, const int vaDeviceIndex) {
    return source_->makeVACapture(inputFilename, vaDeviceIndex);
}

cv::VideoWriter& Viz2D::makeWriter(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize) {
    return Viz2DWorker::makeWriter(outputFilename, fourcc, fps, frameSize);
}

cv::VideoCapture& Viz2D::makeCapture(const string &inputFilename) {
    return source_->makeCapture(inputFilename);
}

} /* namespace viz2d */
} /* namespace kb */
