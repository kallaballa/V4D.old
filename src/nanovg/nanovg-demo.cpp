#define CL_TARGET_OPENCL_VERSION 200

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "nanovg-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;

#include "../common/subsystems.hpp"
#include <csignal>
#include <iomanip>

using std::cerr;
using std::endl;

void drawColorwheel(NVGcontext *vg, float x, float y, float w, float h, float hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    int i;
    float r0, r1, ax, ay, bx, by, cx, cy, aeps, r;
    NVGpaint paint;

    nvgSave(vg);

    /*  nvgBeginPath(vg);
     nvgRect(vg, x,y,w,h);
     nvgFillColor(vg, nvgRGBA(255,0,0,128));
     nvgFill(vg);*/

    cx = x + w * 0.5f;
    cy = y + h * 0.5f;
    r1 = (w < h ? w : h) * 0.5f - 5.0f;
    r0 = r1 - 20.0f;
    aeps = 0.5f / r1;   // half a pixel arc length in radians (2pi cancels out).

    for (i = 0; i < 6; i++) {
        float a0 = (float) i / 6.0f * NVG_PI * 2.0f - aeps;
        float a1 = (float) (i + 1.0f) / 6.0f * NVG_PI * 2.0f + aeps;
        nvgBeginPath(vg);
        nvgArc(vg, cx, cy, r0, a0, a1, NVG_CW);
        nvgArc(vg, cx, cy, r1, a1, a0, NVG_CCW);
        nvgClosePath(vg);
        ax = cx + cosf(a0) * (r0 + r1) * 0.5f;
        ay = cy + sinf(a0) * (r0 + r1) * 0.5f;
        bx = cx + cosf(a1) * (r0 + r1) * 0.5f;
        by = cy + sinf(a1) * (r0 + r1) * 0.5f;
        paint = nvgLinearGradient(vg, ax, ay, bx, by, nvgHSLA(a0 / (NVG_PI * 2), 1.0f, 0.55f, 255), nvgHSLA(a1 / (NVG_PI * 2), 1.0f, 0.55f, 255));
        nvgFillPaint(vg, paint);
        nvgFill(vg);
    }

    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, r0 - 0.5f);
    nvgCircle(vg, cx, cy, r1 + 0.5f);
    nvgStrokeColor(vg, nvgRGBA(0, 0, 0, 64));
    nvgStrokeWidth(vg, 1.0f);
    nvgStroke(vg);

    // Selector
    nvgSave(vg);
    nvgTranslate(vg, cx, cy);
    nvgRotate(vg, hue * NVG_PI * 2);

    // Marker on
    nvgStrokeWidth(vg, 2.0f);
    nvgBeginPath(vg);
    nvgRect(vg, r0 - 1, -3, r1 - r0 + 2, 6);
    nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 192));
    nvgStroke(vg);

    paint = nvgBoxGradient(vg, r0 - 3, -5, r1 - r0 + 6, 10, 2, 4, nvgRGBA(0, 0, 0, 128), nvgRGBA(0, 0, 0, 0));
    nvgBeginPath(vg);
    nvgRect(vg, r0 - 2 - 10, -4 - 10, r1 - r0 + 4 + 20, 8 + 20);
    nvgRect(vg, r0 - 2, -4, r1 - r0 + 4, 8);
    nvgPathWinding(vg, NVG_HOLE);
    nvgFillPaint(vg, paint);
    nvgFill(vg);

    // Center triangle
    r = r0 - 6;
    ax = cosf(120.0f / 180.0f * NVG_PI) * r;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r;
    bx = cosf(-120.0f / 180.0f * NVG_PI) * r;
    by = sinf(-120.0f / 180.0f * NVG_PI) * r;
    nvgBeginPath(vg);
    nvgMoveTo(vg, r, 0);
    nvgLineTo(vg, ax, ay);
    nvgLineTo(vg, bx, by);
    nvgClosePath(vg);
    paint = nvgLinearGradient(vg, r, 0, ax, ay, nvgHSLA(hue, 1.0f, 0.5f, 255), nvgRGBA(255, 255, 255, 255));
    nvgFillPaint(vg, paint);
    nvgFill(vg);
    paint = nvgLinearGradient(vg, (r + ax) * 0.5f, (0 + ay) * 0.5f, bx, by, nvgRGBA(0, 0, 0, 0), nvgRGBA(0, 0, 0, 255));
    nvgFillPaint(vg, paint);
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBA(0, 0, 0, 64));
    nvgStroke(vg);

    // Select circle on triangle
    ax = cosf(120.0f / 180.0f * NVG_PI) * r * 0.3f;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r * 0.4f;
    nvgStrokeWidth(vg, 2.0f);
    nvgBeginPath(vg);
    nvgCircle(vg, ax, ay, 5);
    nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 192));
    nvgStroke(vg);

    paint = nvgRadialGradient(vg, ax, ay, 7, 9, nvgRGBA(0, 0, 0, 64), nvgRGBA(0, 0, 0, 0));
    nvgBeginPath(vg);
    nvgRect(vg, ax - 20, ay - 20, 40, 40);
    nvgCircle(vg, ax, ay, 7);
    nvgPathWinding(vg, NVG_HOLE);
    nvgFillPaint(vg, paint);
    nvgFill(vg);

    nvgRestore(vg);

    nvgRestore(vg);
}

bool done = false;
static void finish(int ignore) {
    std::cerr << endl;
    done = true;
}

int main(int argc, char **argv) {
    done = false;
    signal(SIGINT, finish);

    using namespace kb;
    //Initialize OpenCL Context for VAAPI
    va::init();

    if (argc != 2) {
        cerr << "Usage: nanovg-demo <video-file>" << endl;
        exit(1);
    }

    //Initialize MJPEG HW decoding using VAAPI
    cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    // check if we succeeded
    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);

    //Initialize VP9 HW encoding using VAAPI
    cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat videoFrame;
    cv::UMat videoFrameBGRA;
    cv::UMat videoFrameHSV;
    cv::UMat hueChannel;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    while (!done) {
        //Bind the OpenCL context for OpenGL
        gl::bind();
        //Initially aquire the framebuffer so we can write the video frame to it
        gl::acquire_from_gl(frameBuffer);

        //Bind the OpenCL context for VAAPI
        va::bind();
        //Decode a frame on the GPU using VAAPI
        capture >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //we use time to calculated the current hue
        float time = cv::getTickCount() / cv::getTickFrequency();
        //nanovg hue fading between 0.0f and 1.0f
        float nvgHue = (sinf(time*0.12f)+1.0f) / 2.0f;
        //opencv hue fading between 0 and 255
        int cvHue = (42 + uint8_t(std::round(((1.0 - sinf(time*0.12f))+1.0f) * 128.0))) % 255;

        //The frameBuffer is upside-down. Flip videoFrame. (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
        //Color-conversion from RGB to HSV. (OpenCL)
        cv::cvtColor(videoFrame, videoFrameHSV, cv::COLOR_RGB2HSV_FULL);
        //Extract the hue channel
        cv::extractChannel(videoFrameHSV, hueChannel, 0);
        //Set the current hue
        hueChannel.setTo(cvHue);
        //Insert the hue channel
        cv::insertChannel(hueChannel, videoFrameHSV, 0);
        //Color-conversion from HSV to RGB. (OpenCL)
        cv::cvtColor(videoFrameHSV, videoFrame, cv::COLOR_HSV2RGB_FULL);
        //Color-conversion from RGB to BGRA. (OpenCL)
        cv::cvtColor(videoFrame, videoFrameBGRA, cv::COLOR_RGB2BGRA);
        //Resize the frame if necessary. (OpenCL)
        cv::resize(videoFrameBGRA, frameBuffer, cv::Size(WIDTH, HEIGHT));

        //Bind the OpenCL context for OpenGL again for rendering
        gl::bind();
        //Release the frame buffer for use by OpenGL
        gl::release_to_gl(frameBuffer);
        //Render using nanovg;
        nvg::begin();
        drawColorwheel(nvg::vg, WIDTH - 300, HEIGHT - 300, 250.0f, 250.0f, nvgHue);
        nvg::end();

        //Aquire frame buffer from OpenGL
        gl::acquire_from_gl(frameBuffer);
        //Color-conversion from BGRA to RGB. OpenCV/OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Transfer buffer ownership back to OpenGL
        gl::release_to_gl(frameBuffer);

        //Activate the OpenCL context for VAAPI
        va::bind();
        //The frameBuffer is upside-down. Flip videoFrame. (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
        //Encode the frame using VAAPI on the GPU.
        writer.write(videoFrame);

        if (x11::is_initialized()) {
            //Yet again activate the OpenCL context for OpenGL
            gl::bind();
            //Blit the framebuffer we have been working on to the screen
            gl::blit_frame_buffer_to_screen();

            //Check if the x11 window was closed
            if (x11::window_closed()) {
                finish(0);
                break;
            }
            //Transfer the back buffer (which we have been using as frame buffer) to the native window
            gl::swap_buffers();
        }

        //Measure FPS
        if (cnt % uint64(ceil(lastFps)) == 0) {
            int64 tick = cv::getTickCount();
            lastFps = tickFreq / ((tick - start + 1) / cnt);
            cerr << "FPS : " << lastFps << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}