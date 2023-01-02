
#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/nvg.hpp"
#include "../common/util.hpp"
#include "../common/detail/taskpool.hpp"
#include "../ext/midiplayback.hpp"
#include "../ext/timetracker.hpp"

#include <cmath>
#include <csignal>
#include <vector>
#include <set>
#include <string>
#include <thread>
#include <random>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core/ocl.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using namespace std::literals::chrono_literals;

enum BackgroundModes {
    GREY,
    COLOR,
    VALUE,
    BLACK
};

enum PostProcModes {
    GLOW,
    BLOOM,
    NONE
};

/** Application parameters **/

#ifndef __EMSCRIPTEN__
constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
#else
constexpr unsigned int WIDTH = 1280;
constexpr unsigned int HEIGHT = 720;
#endif
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr const char* OUTPUT_FILENAME = "optflow-demo.mkv";
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;
constexpr size_t FPS = 30;
#ifndef __EMSCRIPTEN__
const size_t NUM_WORKERS = 4;
#else
const size_t NUM_WORKERS = 4;
#endif

static cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(NUM_WORKERS, cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Sparse Optical Flow Demo");

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#  include <fstream>

using namespace emscripten;

std::string pushImage(std::string filename){
    try {
        std::ifstream fs(filename, std::fstream::in | std::fstream::binary);
        fs.seekg (0, std::ios::end);
        auto length = fs.tellg();
        fs.seekg (0, std::ios::beg);

        source->capture([&](cv::UMat &videoFrame) {
            if(videoFrame.empty())
                videoFrame.create(HEIGHT, WIDTH, CV_8UC4);
            cv::Mat tmp = videoFrame.getMat(cv::ACCESS_WRITE);
            fs.read((char*)(tmp.data), tmp.elemSize() * tmp.total());
            tmp.release();
        });
        return "success";
    } catch(std::exception& ex) {
        return string(ex.what());
    }
}

EMSCRIPTEN_BINDINGS(my_module)
{
    function("push_image", &pushImage);
}
#endif

void postEvents(const std::vector<MidiEvent> &events) {
    for (const auto &ev : events) {
        std::vector<std::string> names = v2d->properties().names();

        for (size_t i = 0; i < NUM_WORKERS; ++i) {
            if (ev.controller_ >= 12 && (size_t(ev.controller_) - 12) < names.size()) {
                cerr << names[ev.controller_ - 12] << ":" << ev.value_ << endl;
                v2d->propagate(names[ev.controller_ - 12], ev.value_, 127.0);
            }
        }
    }
}

void prepare_motion_mask(kb::viz2d::Storage& storage, const cv::UMat& srcGrey, cv::UMat& motionMaskGrey) {
    auto& bgSubtractor = storage.local<cv::Ptr<cv::BackgroundSubtractorMOG2>>("bgSubtractor", cv::createBackgroundSubtractorMOG2(100, 16.0, false));
    auto& element = storage.local<cv::Mat>("element", cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1)));

    bgSubtractor->apply(srcGrey, motionMaskGrey);
    cv::morphologyEx(motionMaskGrey, motionMaskGrey, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
}

void detect_points(kb::viz2d::Storage& storage, const cv::UMat& srcMotionMaskGrey, vector<cv::Point2f>& points) {
    auto& detector = storage.local<cv::Ptr<cv::FastFeatureDetector>>("detector", cv::FastFeatureDetector::create(1, false));
    auto& tmpKeyPoints = storage.local<vector<cv::KeyPoint>>("tmpKeyPoints");

    tmpKeyPoints.clear();
    detector->detect(srcMotionMaskGrey, tmpKeyPoints);

    points.clear();
    for (const auto &kp : tmpKeyPoints) {
        points.push_back(kp.pt);
    }
}

bool detect_scene_change(kb::viz2d::Storage& storage, const cv::UMat& srcMotionMaskGrey, const float& thresh, const float& theshDiff) {
    auto& lastMovement = storage.local<float>("lastMovement");

    float movement = cv::countNonZero(srcMotionMaskGrey) / double(srcMotionMaskGrey.cols * srcMotionMaskGrey.rows);
    float relation = movement > 0 && lastMovement > 0 ? std::max(movement, lastMovement) / std::min(movement, lastMovement) : 0;
    float relM = relation * log10(1.0f + (movement * 9.0));
    float relLM = relation * log10(1.0f + (lastMovement * 9.0));

    bool result = !((movement > 0 && lastMovement > 0 && relation > 0)
            && (relM < thresh && relLM < thresh && fabs(relM - relLM) < theshDiff));
    lastMovement = (lastMovement + movement) / 2.0f;
    return result;
}

void visualize_sparse_optical_flow(kb::viz2d::Storage& storage, const cv::UMat &prevGrey, const cv::UMat &nextGrey, const vector<cv::Point2f> &detectedPoints, const float scaleFactor, const int maxStrokeSize, const cv::Scalar color, const int maxPoints, const float pointLossPercent) {
    auto& hull = storage.local<vector<cv::Point2f>>("hull");
    auto& prevPoints = storage.local<vector<cv::Point2f>>("prevPoints");
    auto& nextPoints = storage.local<vector<cv::Point2f>>("nextPoints");
    auto& newPoints = storage.local<vector<cv::Point2f>>("newPoints");
    auto& upPrevPoints = storage.local<vector<cv::Point2f>>("upPrevPoints");
    auto& upNextPoints = storage.local<vector<cv::Point2f>>("upNextPoints");
    auto& status = storage.local<vector<uchar>>("status");
    auto& err = storage.local<vector<float>>("err");
    auto& rd = storage.local<std::random_device>("rd");
    auto& g = storage.local<cv::Ptr<std::mt19937>>("g", new std::mt19937(rd()));

    if (detectedPoints.size() > 4) {
        cv::convexHull(detectedPoints, hull);
        float area = cv::contourArea(hull);
        if (area > 0) {
            float density = (detectedPoints.size() / area);
            float strokeSize = maxStrokeSize * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
            size_t currentMaxPoints = ceil(density * maxPoints);

            std::shuffle(prevPoints.begin(), prevPoints.end(), *g);
            prevPoints.resize(ceil(prevPoints.size() * (1.0f - (pointLossPercent / 100.0f))));

            size_t copyn = std::min(detectedPoints.size(), (size_t(std::ceil(currentMaxPoints)) - prevPoints.size()));
            if (prevPoints.size() < currentMaxPoints) {
                std::copy(detectedPoints.begin(), detectedPoints.begin() + copyn, std::back_inserter(prevPoints));
            }

            cv::calcOpticalFlowPyrLK(prevGrey, nextGrey, prevPoints, nextPoints, status, err);
            newPoints.clear();
            if (prevPoints.size() > 1 && nextPoints.size() > 1) {
                upNextPoints.clear();
                upPrevPoints.clear();
                for (cv::Point2f pt : prevPoints) {
                    upPrevPoints.push_back(pt /= scaleFactor);
                }

                for (cv::Point2f pt : nextPoints) {
                    upNextPoints.push_back(pt /= scaleFactor);
                }

                using namespace kb::viz2d::nvg;
                beginPath();
                strokeWidth(strokeSize);
                strokeColor(color);

                for (size_t i = 0; i < prevPoints.size(); i++) {
                    if (status[i] == 1 && err[i] < (1.0 / density) && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < nextGrey.rows / scaleFactor && upNextPoints[i].x < nextGrey.cols / scaleFactor) {
                        float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                        if (len > 0 && len < sqrt(area)) {
                            newPoints.push_back(nextPoints[i]);
                            moveTo(upNextPoints[i].x, upNextPoints[i].y);
                            lineTo(upPrevPoints[i].x, upPrevPoints[i].y);
                        }
                    }
                }
                stroke();
            }
            prevPoints = newPoints;
        }
    }
}

void bloom(kb::viz2d::Storage& storage, const cv::UMat& src, cv::UMat &dst, int ksize = 3, int threshValue = 235, float gain = 4) {
    cv::UMat& bgr = storage.local("bgr");
    cv::UMat& hls = storage.local("hls");
    cv::UMat& ls16 = storage.local("ls16");
    cv::UMat& ls = storage.local("ls");
    cv::UMat& blur = storage.local("blur");
    std::vector<cv::UMat>& hlsChannels = storage.local<std::vector<cv::UMat>>("hlsChannels");

    cv::cvtColor(src, bgr, cv::COLOR_BGRA2RGB);
    cv::cvtColor(bgr, hls, cv::COLOR_BGR2HLS);
    cv::split(hls, hlsChannels);
    cv::bitwise_not(hlsChannels[2], hlsChannels[2]);

    cv::multiply(hlsChannels[1], hlsChannels[2], ls16, 1, CV_16U);
    cv::divide(ls16, cv::Scalar(255.0), ls, 1, CV_8U);
    cv::threshold(ls, blur, threshValue, 255, cv::THRESH_BINARY);

    cv::boxFilter(blur, blur, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::cvtColor(blur, blur, cv::COLOR_GRAY2BGRA);

    addWeighted(src, 1.0, blur, gain, 0, dst);
}

void glow_effect(kb::viz2d::Storage& storage, const cv::UMat &src, cv::UMat &dst, const int ksize) {
    cv::UMat& resize = storage.local("resize");
    cv::UMat& blur = storage.local("blur");
    cv::UMat& dst16 = storage.local("dst16");

    cv::bitwise_not(src, dst);

    //Resize for some extra performance
    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, src.size());

    //Multiply the src image with a blurred version of itself
    cv::multiply(dst, blur, dst16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}

void composite_layers(kb::viz2d::Storage& storage, const cv::UMat& background, cv::UMat& foreground, const cv::UMat& frameBuffer, cv::UMat& dst, int kernelSize, float fgLossPercent, BackgroundModes bgMode, PostProcModes ppMode, int bloomThresh, float bloomGain) {
    cv::UMat& backgroundGrey = storage.local("backgroundGrey");
    cv::UMat& newBackground = storage.allocLocal("newBackground", background.size(), background.type(), cv::Scalar::all(0));
    cv::UMat& tmp = storage.local("tmp");
    cv::UMat& post = storage.local("post");
    vector<cv::UMat>& channels = storage.local<vector<cv::UMat>>("hsvChannels");

    cv::subtract(foreground, cv::Scalar::all(255.0f * (fgLossPercent / 100.0f)), foreground);
    cv::add(foreground, frameBuffer, foreground);

    switch (bgMode) {
    case GREY:
        cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(backgroundGrey, newBackground, cv::COLOR_GRAY2BGRA);
        break;
    case VALUE:
        cv::cvtColor(background, tmp, cv::COLOR_BGRA2BGR);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
        split(tmp, channels);
        cv::cvtColor(channels[2], newBackground, cv::COLOR_GRAY2BGRA);
        break;
    case COLOR:
        cv::cvtColor(background, newBackground, cv::COLOR_BGRA2RGBA);
        break;
    case BLACK:
        newBackground = cv::Scalar::all(0);
        break;
    default:
        break;
    }

    switch (ppMode) {
    case GLOW:
        glow_effect(storage, foreground, post, kernelSize);
        break;
    case BLOOM:
        bloom(storage, foreground, post, kernelSize, bloomThresh, bloomGain);
        break;
    case NONE:
        foreground.copyTo(post);
        break;
    default:
        break;
    }

    cv::add(newBackground, post, dst);
}

void setup_gui(cv::Ptr<kb::viz2d::Viz2D> v2d) {
    v2d->addWindow(5, 30, "Effects");

    v2d->addGroup("Foreground");
    v2d->addFormWidget("fgScale", "Scale", 0.5f, 0.1f, 4.0f, true, "", "Generate the foreground at this scale");
    v2d->addFormWidget("fgLoss", "Loss", 2.5f, 0.1f, 99.9f, true, "%", "On every frame the foreground loses on brightness");

    v2d->addGroup("Background");
    v2d->addFormWidget("bgMode", "Mode", GREY, GREY, BLACK, {"Grey", "Color", "Value", "Black"});

    v2d->addGroup("Points");
    v2d->addFormWidget("maxPoints", "Max. Points", 250000, 10, 1000000, true, "", "The theoretical maximum number of points to track which is scaled by the density of detected points and therefor is usually much smaller");
    v2d->addFormWidget("pointLoss", "Point Loss", 25.0f, 0.0f, 100.0f, true, "%", "How many of the tracked points to lose intentionally");

    v2d->addGroup("Optical flow");
    v2d->addFormWidget("maxStroke", "Max. Stroke Size", 14, 1, 100, true, "px", "The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull of tracked points and therefor is usually much smaller");
    v2d->addFormWidget("color", "Color", nanogui::Color(1.0f, 0.75f, 0.4f, 1.0f), "The primary effect color");
    v2d->addFormWidget("alpha", "Alpha", 0.1f, 0.0f, 1.0f, true, "", "The opacity of the effect");

    v2d->addWindow(220, 30, "Post Processing");
    auto* postPocMode = v2d->addFormWidget("ppMode", "Mode",GLOW, GLOW, NONE, {"Glow", "Bloom", "None"});
    auto* kernelSize = v2d->addFormWidget("ksize", "Kernel Size", std::max(int(DIAG / 100 % 2 == 0 ? DIAG / 100 + 1 : DIAG / 100), 1), 1, 63, true, "", "Intensity of glow defined by kernel size");
    kernelSize->set_callback([=](const int& k) {
        static int lastKernelSize = v2d->property<int>("ksize");

        int& ksize = v2d->property<int>("ksize");
        if(k == lastKernelSize)
            return;

        if(k <= lastKernelSize) {
            ksize = std::max(int(k % 2 == 0 ? k - 1 : k), 1);
        } else if(k > lastKernelSize)
            ksize = std::max(int(k % 2 == 0 ? k + 1 : k), 1);

        lastKernelSize = k;
        kernelSize->set_value(ksize);
    });
    auto* thresh = v2d->addFormWidget("bloomThresh", "Threshold", 210, 1, 255, true, "", "The lightness selection threshold", true, false);
    auto* gain = v2d->addFormWidget("bloomGain", "Gain", 3.0f, 0.1f, 20.0f, true, "", "Intensity of the effect defined by gain", true, false);
    postPocMode->set_callback([&,kernelSize, thresh, gain](const int& m) {
        PostProcModes ppm = v2d->property<PostProcModes>("ppMode") = static_cast<PostProcModes>(m);
        if(ppm == BLOOM) {
            thresh->set_enabled(true);
            gain->set_enabled(true);
        } else {
            thresh->set_enabled(false);
            gain->set_enabled(false);
        }

        if(ppm == NONE) {
            kernelSize->set_enabled(false);
        } else {
            kernelSize->set_enabled(true);
        }
    });

    v2d->addWindow(220, 175, "Settings");

    v2d->addGroup("Hardware Acceleration");
    v2d->addFormWidget("hwEnable", "Enable", true, "Enable or disable libva and OpenCL acceleration");

    v2d->addGroup("Scene Change Detection");
    v2d->addFormWidget("sceneThresh", "Threshold", 0.29f, 0.1f, 1.0f, true, "", "Peak threshold. Lowering it makes detection more sensitive");
    v2d->addFormWidget("sceneThreshDiff", "Threshold Diff", 0.1f, 0.1f, 1.0f, true, "", "Difference of peak thresholds. Lowering it makes detection more sensitive");

    v2d->addWindow(8, 16, "Display");

    v2d->addGroup("Display");
    v2d->addFormWidget("showFPS", "Show FPS", true, "Enable or disable the On-screen FPS display");
    v2d->addFormWidget("stretch", "Stretch", false, "Stretch the frame buffer to the window size")->set_callback([=](const bool &s) {
        v2d->setStretching(s);
    });

#ifndef __EMSCRIPTEN__
    v2d->addButton("Fullscreen", [=]() {
        v2d->setFullscreen(!v2d->isFullscreen());
    });

    v2d->addButton("Offscreen", [=]() {
        v2d->setOffscreen(!v2d->isOffscreen());
    });
#endif
}

std::vector<kb::viz2d::Task> plan(kb::viz2d::Viz2DWorker& worker) {
    using namespace kb::viz2d;
    return {
        worker.clgl("prepare", [](Storage& storage, cv::UMat& frameBuffer) {
            cv::UMat& down = storage.output("down");
            cv::UMat& background = storage.output("background");

            const float& fgScale = storage.property<float>("fgScale");

            cv::resize(frameBuffer, down, cv::Size(frameBuffer.size().width * fgScale, frameBuffer.size().height * fgScale));
            frameBuffer.copyTo(background);
        }),
        worker.cl("detect", [](Storage& storage) {
            const cv::UMat& down = storage.input("down");
            cv::UMat& downNextGrey = storage.output("downNextGrey");
            cv::UMat& downMotionMaskGrey = storage.output("downMotionMaskGrey");

            auto& detectedPoints = storage.output<vector<cv::Point2f>>("detectedPoints");

            cv::cvtColor(down, downNextGrey, cv::COLOR_RGBA2GRAY);
            //Subtract the background to create a motion mask
            prepare_motion_mask(storage, downNextGrey, downMotionMaskGrey);
            //Detect trackable points in the motion mask
            detect_points(storage, downMotionMaskGrey, detectedPoints);
        }),
        worker.nvg("optflow", [](Storage& storage, const cv::Size& sz) {
            const cv::UMat& downMotionMaskGrey = storage.input("downMotionMaskGrey");
            const cv::UMat& downPrevGrey = storage.input("downPrevGrey");
            const cv::UMat& downNextGrey = storage.input("downNextGrey");

            const auto& detectedPoints = storage.input<vector<cv::Point2f>>("detectedPoints");

            const float& sceneThresh = storage.property<float>("sceneThresh");
            const float& sceneThreshDiff = storage.property<float>("sceneThreshDiff");
            const float& alpha = storage.property<float>("alpha");
            const float& fgScale = storage.property<float>("fgScale");
            const float& maxStroke = storage.property<int>("maxStroke");
            const float& maxPoints = storage.property<int>("maxPoints");
            const float& pointLoss = storage.property<float>("pointLoss");
            const nanogui::Color& c = storage.property<nanogui::Color>("color");

            nvg::clear();
            if (!downPrevGrey.empty()) {
                //We don't want the algorithm to get out of hand when there is a scene change, so we suppress it when we detect one.
                if (!detect_scene_change(storage, downMotionMaskGrey, sceneThresh, sceneThreshDiff)) {
                    //Visualize the sparse optical flow using nanovg
                    cv::Scalar color = cv::Scalar(c.b() * 255.0f, c.g() * 255.0f, c.r() * 255.0f, alpha * 255.0f);
                    visualize_sparse_optical_flow(storage, downPrevGrey, downNextGrey, detectedPoints, fgScale, maxStroke, color, maxPoints, pointLoss);
                }
            }
        }),
        worker.cl("clone", [](Storage& storage){
            storage.output("downPrevGrey") = storage.input("downNextGrey").clone();
        }),
        worker.clgl("composite", [](Storage& storage, cv::UMat& frameBuffer){
            cv::UMat& foreground = storage.allocSharedOutput("foreground", frameBuffer.size(), frameBuffer.type(), cv::Scalar::all(0));
            const cv::UMat& background = storage.input("background");

            const int& ksize = storage.property<int>("ksize");
            const float& fgLoss = storage.property<float>("fgLoss");
            const int& bloomThresh = storage.property<int>("bloomThresh");
            const float& bloomGain = storage.property<float>("bloomGain");
            const BackgroundModes& bgMode = storage.property<BackgroundModes>("bgMode");
            const PostProcModes& ppMode = storage.property<PostProcModes>("ppMode");

            //Put it all together (OpenCL)
            composite_layers(storage, background, foreground, frameBuffer, frameBuffer, ksize, fgLoss, bgMode, ppMode, bloomThresh, bloomGain);
        })
    };
}

bool done;
static void finish(int ignore) {
    done = true;
}

void start_midi(int port) {
    std::thread midiThread([=]() {
        MidiReceiver midi(port);
        while (!done) {
            postEvents(midi.receive());
            usleep(10000);
        }
    });
    midiThread.detach();
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;
#ifndef __EMSCRIPTEN__
    if (argc != 3) {
        std::cerr << "Usage: optflow <input-video-file> <midi-port>" << endl;
        exit(1);
    }
#endif
    done = false;
    signal(SIGINT, finish);
    signal(SIGTERM, finish);

    print_system_info();

    if(!v2d->isOffscreen()) {
        setup_gui(v2d);
        v2d->setVisible(true);
    }

    start_midi(atoi(argv[2]));

#ifndef __EMSCRIPTEN__
    auto capture = v2d->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        exit(-1);
    }

    float fps = capture.get(cv::CAP_PROP_FPS);
    float width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    float height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(width, height), VA_HW_DEVICE_INDEX);

    v2d->prepare(plan);

    while (!done) {
        v2d->work();
    }
#else
    emscripten_set_main_loop(iteration, -1, false);
#endif


    return 0;
}
