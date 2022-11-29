#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "tetra-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int GLOW_KERNEL_SIZE = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;

void init_scene(unsigned long w, unsigned long h) {
    //Initialize the OpenGL scene
    glViewport(0, 0, w, h);
    glColor3f(1.0, 1.0, 1.0);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-2, 2, -1.5, 1.5, 1, 40);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -3);
    glRotatef(50, 1, 0, 0);
    glRotatef(70, 0, 1, 0);
}

void render_scene(unsigned long w, unsigned long h) {
    //Render a tetrahedron using immediate mode because the code is more concise for a demo
    glViewport(0, 0, w, h);
    glRotatef(1, 0, 1, 0);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

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

int main(int argc, char **argv) {
    using namespace kb;

    //Initialize the application
    kb::init(WIDTH, HEIGHT);

    //Initialize VP9 HW encoding using VAAPI
    cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //Copy OpenCL Context for VAAPI. Must be called right after first VideoWriter/VideoCapture initialization.
    va::copy();

    //If we are rendering offscreen we don't need x11
    if(!OFFSCREEN)
        x11::init("tetra-demo");

    //you can set OpenGL-version, multisample-buffer samples and enable debug context using egl::init()
    egl::init();
    //Initialize OpenCL Context for OpenGL
    gl::init();

    init_scene(WIDTH, HEIGHT);

    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    //BGRA
    cv::UMat frameBuffer;
    //RGB
    cv::UMat videoFrame;

    while (true) {
        //Activate the OpenCL context for OpenGL
        gl::bind();
        //Render using OpenGL
        gl::begin();
        render_scene(WIDTH, HEIGHT);
        gl::end();

        //Aquire the frame buffer for use by OpenCL
        gl::acquire_from_gl(frameBuffer);
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, GLOW_KERNEL_SIZE);
        //Color-conversion from BGRA to RGB. OpenCV/OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Release the frame buffer for use by OpenGL
        gl::release_to_gl(frameBuffer);

        //If x11 is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if(!gl::display())
            break;

        //Activate the OpenCL context for VAAPI
        va::bind();
        //Encode the frame using VAAPI on the GPU.
        writer << videoFrame;

        print_fps();
    }

    return 0;
}
