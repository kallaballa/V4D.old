#define CL_TARGET_OPENCL_VERSION 220

//WIDTH and HEIGHT have to be specified before including subsystems.hpp
constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;

#include "../common/subsystems.hpp"
#include <stdlib.h>
#include <string>

using std::cerr;
using std::endl;
using std::string;

//Static stream info. Has to match your capture device/file
constexpr double INPUT_FPS = 30;
constexpr int INPUT_WIDTH = 320;
constexpr int INPUT_HEIGHT = 240;
const string INPUT_FORMAT = "mjpeg";
const string PIXEL_FORMAT = "yuyv422";
const string INPUT_FILENAME = "example.mp4";
const string OUTPUT_FILENAME = "camera-demo.mkv";

//The ffmpeg capture and writer options we need to capture... but don't overwrite the environment variables if they already exist.
const string CAPTURE_OPTIONS = "framerate;" + std::to_string(INPUT_FPS)
        + "|input_format;" + INPUT_FORMAT
        + "|video_size;" + std::to_string(INPUT_WIDTH) + "x" + std::to_string(INPUT_HEIGHT)
        + "|pixel_format;" + PIXEL_FORMAT;

const string WRITER_OPTIONS = "";

constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr bool OFFSCREEN = true;

cv::ocl::OpenCLExecutionContext VA_CONTEXT;
cv::ocl::OpenCLExecutionContext GL_CONTEXT;

void render() {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
    glBindFramebuffer(GL_FRAMEBUFFER, kb::gl::frame_buf);
    glViewport(0, 0, WIDTH , HEIGHT );
    glRotatef(1, 0, 1, 0);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINES);
    for (GLfloat i = -2.5; i <= 2.5; i += 0.25) {
        glVertex3f(i, 0, 2.5);
        glVertex3f(i, 0, -2.5);
        glVertex3f(2.5, 0, i);
        glVertex3f(-2.5, 0, i);
    }
    glEnd();

    glBegin(GL_TRIANGLE_STRIP);
        glColor3f(1, 1, 1);
        glVertex3f(0, 2, 0);
        glColor3f(1, 0, 0);
        glVertex3f(-1, 0, 1);
        glColor3f(0, 1, 0);
        glVertex3f(1, 0, 1);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, -1.4);
        glColor3f(1, 1, 1);
        glVertex3f(0, 2, 0);
        glColor3f(1, 0, 0);
        glVertex3f(-1, 0, 1);
    glEnd();
    glFlush();
}

void glow(cv::UMat &src, int ksize = WIDTH / 85 % 2 == 0 ? WIDTH / 85  + 1 : WIDTH / 85) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat src16;

    cv::bitwise_not(src, src);

    //Resize for some extra performance
    cv::resize(src, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, cv::Size(WIDTH, HEIGHT));

    //Multiply the src image with a blurred version of itself
    cv::multiply(src, blur, src16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(src16, cv::Scalar::all(255.0), src, 1, CV_8U);

    cv::bitwise_not(src, src);
}

int main(int argc, char **argv) {
    //The ffmpeg capture and writer options we need to capture... but don't overwrite the environment variables if they already exist.
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", CAPTURE_OPTIONS.c_str(), 0);
    setenv("OPENCV_FFMPEG_WRITER_OPTIONS", WRITER_OPTIONS.c_str(), 0);

    using namespace kb;

    //Initialize OpenCL Context for VAAPI
    va::init_va();
    /*
     * The OpenCLExecutionContext for VAAPI needs to be copied right after init_va().
     * Now everytime you want to do VAAPI interop first bind the context.
     */
    VA_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    //Initialize MJPEG HW decoding using VAAPI
    cv::VideoCapture cap(INPUT_FILENAME, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    //Initialize VP9 HW encoding using VAAPI. We don't need to specify the hardware device twice. only generates a warning.
    cv::VideoWriter video(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), INPUT_FPS, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //If we are rendering offscreen we don't need x11
    if(!OFFSCREEN)
        x11::init_x11();

    //Passing true to init_egl will create a OpenGL debug context
    egl::init_egl();
    //Initialize OpenCL Context for OpenGL
    gl::init_gl();
    /*
     * The OpenCLExecutionContext for OpenGL needs to be copied right after init_gl().
     * Now everytime you want to do OpenGL interop first bind the context.
     */
    GL_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat videoFrame;
    cv::UMat videoFrameRGBA;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = 0;

    while (true) {
        //Activate the OpenCL context for OpenGL
        GL_CONTEXT.bind();
        //Initially aquire the framebuffer so we can write the video frame to it
        gl::acquire_frame_buffer(frameBuffer);

        //Activate the OpenCL context for VAAPI
        VA_CONTEXT.bind();
        //Decode a frame on the GPU using VAAPI
        cap >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //The video is upside-down. Flip it. (OpenCL)
        cv::flip(videoFrame, videoFrame, 0);
        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(videoFrame, videoFrameRGBA, cv::COLOR_RGB2BGRA);
        //Resize the frame. (OpenCL)
        cv::resize(videoFrameRGBA, frameBuffer, cv::Size(WIDTH, HEIGHT));

        GL_CONTEXT.bind();
        //Release the frame buffer for use by OpenGL
        gl::release_frame_buffer(frameBuffer);
        //Render using OpenGL
        render();
        //Aquire the frame buffer for use by OpenCL
        gl::acquire_frame_buffer(frameBuffer);

        //Glow effect (OpenCL)
        glow(frameBuffer);

        //Color-conversion from BGRA to RGB. (OpenCL)
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        cv::flip(videoFrame, videoFrame, 0);

        //Activate the OpenCL context for VAAPI
        VA_CONTEXT.bind();
        //Encode the frame using VAAPI on the GPU.
        video.write(videoFrame);

        if(x11::is_initialized()) {
            //Yet again activate the OpenCL context for OpenGL
            GL_CONTEXT.bind();
            //Release the frame buffer for use by OpenGL
            gl::release_frame_buffer(frameBuffer);
            //Blit the framebuffer we have been working on to the screen
            gl::blit_frame_buffer_to_screen();

            //Check if the x11 window was closed
            if(x11::window_closed())
                break;

            //Transfer the back buffer (which we have been using as frame buffer) to the native window
            gl::swapBuffers();
        }

        //Measure FPS
        if (cnt % uint64(ceil(lastFps == 0 ? INPUT_FPS : lastFps)) == 0) {
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
