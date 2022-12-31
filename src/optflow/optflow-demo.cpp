#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/nvg.hpp"
#include "../common/util.hpp"
#include "../ext/midiplayback.hpp"

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

std::vector<MidiEvent> EVENTS;
std::mutex EV_MTX;

cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Sparse Optical Flow Demo");
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

        v2d->capture([&](cv::UMat &videoFrame) {
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
    std::vector<std::string> names = v2d->names();
    for(const auto& ev: events) {
        if(ev.controller_ - 12 < names.size()) {
            cerr << names[ev.controller_ - 12] << ":" << ev.value_ << endl;
            v2d->propagate(names[ev.controller_ - 12], ev.value_, 127.0);
        }
    }
}

void prepare_motion_mask(const cv::UMat& srcGrey, cv::UMat& motionMaskGrey) {
    static cv::Ptr<cv::BackgroundSubtractor> bg_subtrator = cv::createBackgroundSubtractorMOG2(100, 16.0, false);
    static int morph_size = 1;
    static cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));

    bg_subtrator->apply(srcGrey, motionMaskGrey);
    cv::morphologyEx(motionMaskGrey, motionMaskGrey, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
}

void detect_points(const cv::UMat& srcMotionMaskGrey, vector<cv::Point2f>& points) {
    static cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(1, false);
    static vector<cv::KeyPoint> tmpKeyPoints;

    tmpKeyPoints.clear();
    detector->detect(srcMotionMaskGrey, tmpKeyPoints);

    points.clear();
    for (const auto &kp : tmpKeyPoints) {
        points.push_back(kp.pt);
    }
}

bool detect_scene_change(const cv::UMat& srcMotionMaskGrey, const float& thresh, const float& theshDiff) {
    static float last_movement = 0;

    float movement = cv::countNonZero(srcMotionMaskGrey) / double(srcMotionMaskGrey.cols * srcMotionMaskGrey.rows);
    float relation = movement > 0 && last_movement > 0 ? std::max(movement, last_movement) / std::min(movement, last_movement) : 0;
    float relM = relation * log10(1.0f + (movement * 9.0));
    float relLM = relation * log10(1.0f + (last_movement * 9.0));

    bool result = !((movement > 0 && last_movement > 0 && relation > 0)
            && (relM < thresh && relLM < thresh && fabs(relM - relLM) < theshDiff));
    last_movement = (last_movement + movement) / 2.0f;
    return result;
}

void visualize_sparse_optical_flow(const cv::UMat &prevGrey, const cv::UMat &nextGrey, vector<cv::Point2f> &detectedPoints, const float scaleFactor, const int maxStrokeSize, const cv::Scalar color, const int maxPoints, const float pointLossPercent) {
    static vector<cv::Point2f> hull, prevPoints, nextPoints, newPoints;
    static vector<cv::Point2f> upPrevPoints, upNextPoints;
    static std::vector<uchar> status;
    static std::vector<float> err;
    static std::random_device rd;
    static std::mt19937 g(rd());

    if (detectedPoints.size() > 4) {
        cv::convexHull(detectedPoints, hull);
        float area = cv::contourArea(hull);
        if (area > 0) {
            float density = (detectedPoints.size() / area);
            float strokeSize = maxStrokeSize * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
            size_t currentMaxPoints = ceil(density * maxPoints);

            std::shuffle(prevPoints.begin(), prevPoints.end(), g);
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

void bloom(const cv::UMat& src, cv::UMat &dst, int ksize = 3, int threshValue = 235, float gain = 4) {
    static cv::UMat bgr;
    static cv::UMat hls;
    static cv::UMat ls16;
    static cv::UMat ls;
    static cv::UMat blur;
    static std::vector<cv::UMat> hlsChannels;

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

void glow_effect(const cv::UMat &src, cv::UMat &dst, const int ksize) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat dst16;

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

void composite_layers(cv::UMat& background, const cv::UMat& foreground, const cv::UMat& frameBuffer, cv::UMat& dst, int kernelSize, float fgLossPercent, BackgroundModes bgMode, PostProcModes ppMode, int bloomThresh, float bloomGain) {
    static cv::UMat tmp;
    static cv::UMat post;
    static cv::UMat backgroundGrey;
    static vector<cv::UMat> channels;

    cv::subtract(foreground, cv::Scalar::all(255.0f * (fgLossPercent / 100.0f)), foreground);
    cv::add(foreground, frameBuffer, foreground);

    switch (bgMode) {
    case GREY:
        cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(backgroundGrey, background, cv::COLOR_GRAY2BGRA);
        break;
    case VALUE:
        cv::cvtColor(background, tmp, cv::COLOR_BGRA2BGR);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
        split(tmp, channels);
        cv::cvtColor(channels[2], background, cv::COLOR_GRAY2BGRA);
        break;
    case COLOR:
        cv::cvtColor(background, background, cv::COLOR_BGRA2RGBA);
        break;
    case BLACK:
        background = cv::Scalar::all(0);
        break;
    default:
        break;
    }

    switch (ppMode) {
    case GLOW:
        glow_effect(foreground, post, kernelSize);
        break;
    case BLOOM:
        bloom(foreground, post, kernelSize, bloomThresh, bloomGain);
        break;
    case NONE:
        foreground.copyTo(post);
        break;
    default:
        break;
    }

    cv::add(background, post, dst);
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

void iteration() {
    using namespace kb::viz2d;

//    if(v2d->isAccelerated() != v2d->property<bool>("hwEnable"))
//        v2d->setAccelerated(false);

#ifndef __EMSCRIPTEN__
    if(!v2d->capture())
        exit(0);
#endif
    v2d->clgl([](Viz2D& v2d, cv::UMat& frameBuffer) {
        cv::UMat& down = v2d.buffer("down");
        cv::UMat& background = v2d.buffer("background");

        const float& fgScale = v2d.property<float>("fgScale");

        cv::resize(frameBuffer, down, cv::Size(frameBuffer.size().width * fgScale, frameBuffer.size().height * fgScale));
        frameBuffer.copyTo(background);
    });

    v2d->cl([](Viz2D& v2d) {
        cv::UMat& down = v2d.buffer("down");
        cv::UMat& downNextGrey = v2d.buffer("downNextGrey");
        cv::UMat& downMotionMaskGrey = v2d.buffer("downMotionMaskGrey");

        auto& detectedPoints = v2d.variable<vector<cv::Point2f>>("detectedPoints");

        cv::cvtColor(down, downNextGrey, cv::COLOR_RGBA2GRAY);
        //Subtract the background to create a motion mask
        prepare_motion_mask(downNextGrey, downMotionMaskGrey);
        //Detect trackable points in the motion mask
        detect_points(downMotionMaskGrey, detectedPoints);
    });

    v2d->nvg([](Viz2D& v2d, const cv::Size& sz) {
        cv::UMat& downMotionMaskGrey = v2d.buffer("downMotionMaskGrey");
        cv::UMat& downPrevGrey = v2d.buffer("downPrevGrey");
        cv::UMat& downNextGrey = v2d.buffer("downNextGrey");

        auto& detectedPoints = v2d.variable<vector<cv::Point2f>>("detectedPoints");

        const float& sceneThresh = v2d.property<float>("sceneThresh");
        const float& sceneThreshDiff = v2d.property<float>("sceneThreshDiff");
        const float& alpha = v2d.property<float>("alpha");
        const float& fgScale = v2d.property<float>("fgScale");
        const float& maxStroke = v2d.property<int>("maxStroke");
        const float& maxPoints = v2d.property<int>("maxPoints");
        const float& pointLoss = v2d.property<float>("pointLoss");
        const nanogui::Color& c = v2d.property<nanogui::Color>("color");

        nvg::clear();
        if (!downPrevGrey.empty()) {
            //We don't want the algorithm to get out of hand when there is a scene change, so we suppress it when we detect one.
            if (!detect_scene_change(downMotionMaskGrey, sceneThresh, sceneThreshDiff)) {
                //Visualize the sparse optical flow using nanovg
                cv::Scalar color = cv::Scalar(c.b() * 255.0f, c.g() * 255.0f, c.r() * 255.0f, alpha * 255.0f);
                visualize_sparse_optical_flow(downPrevGrey, downNextGrey, detectedPoints, fgScale, maxStroke, color, maxPoints, pointLoss);
            }
        }
    });

    v2d->cl([](Viz2D& v2d){
        v2d.buffer("downPrevGrey") = v2d.buffer("downNextGrey").clone();
    });

    v2d->clgl([](Viz2D& v2d, cv::UMat& frameBuffer){
        cv::UMat& foreground = v2d.allocate_once("foreground", frameBuffer.size(), frameBuffer.type(), cv::Scalar::all(0));
        cv::UMat& background = v2d.buffer("background");
        cv::UMat& menuFrame = v2d.buffer("menuFrame");

        const int& ksize = v2d.property<int>("ksize");
        const float& fgLoss = v2d.property<float>("fgLoss");
        const int& bloomThresh = v2d.property<int>("bloomThresh");
        const float& bloomGain = v2d.property<float>("bloomGain");
        const BackgroundModes& bgMode = v2d.property<BackgroundModes>("bgMode");
        const PostProcModes& ppMode = v2d.property<PostProcModes>("ppMode");

        //Put it all together (OpenCL)
        composite_layers(background, foreground, frameBuffer, frameBuffer, ksize, fgLoss, bgMode, ppMode, bloomThresh, bloomGain);
#ifndef __EMSCRIPTEN__
        cvtColor(frameBuffer, menuFrame, cv::COLOR_BGRA2RGB);
#endif
    });

    update_fps(v2d, v2d->property<bool>("showFPS"));

#ifndef __EMSCRIPTEN__
    v2d->write();
#endif

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if(!v2d->display())
        exit(0);
}

bool done;
static void finish(int ignore) {
    done = true;
}

void startMidi(int port) {
    std::thread midiThread([=]() {
        MidiReceiver midi(port);
        long long cnt = 0;
        auto epoch = std::chrono::system_clock::now().time_since_epoch();
        auto firstFrameTime = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();

        while (!done) {
            epoch = std::chrono::system_clock::now().time_since_epoch();
            auto start = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
            {
                postEvents(midi.receive());
            }

            epoch = std::chrono::system_clock::now().time_since_epoch();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count() - start;

            if (dur < (1000000.0 / FPS))
                usleep((1000000.0 / FPS) - dur);
            else
                std::cerr << "Underrun: " << (dur - (1000000.0 / FPS)) / 1000.0 << std::endl;
            ++cnt;
        }
        epoch = std::chrono::system_clock::now().time_since_epoch();
        auto lastFrameTime = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
        auto total = lastFrameTime - firstFrameTime;
        auto perfect = cnt * (1000000.0 / FPS);

        std::cerr << "skew: " << (double) perfect / total << std::endl;
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

    startMidi(atoi(argv[2]));

    if(!v2d->isOffscreen()) {
        setup_gui(v2d);
        v2d->setVisible(true);
    }

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
    while (!done) {
        iteration();
    }
#else
    emscripten_set_main_loop(iteration, -1, false);
#endif


    return 0;
}
