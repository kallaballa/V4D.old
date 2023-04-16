// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"

using std::cerr;
using std::endl;

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "shader-demo.mkv";
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

int glow_kernel_size = std::max(int(DIAG / 200 % 2 == 0 ? DIAG / 200 + 1 : DIAG / 200), 1);

/** mandelbrot control parameters **/
// Red, green, blue and alpha. All from 0.0f to 1.0f
nanogui::Color base_color_val(0.2f, 0.6f, 1.0f, 1.0f);
// Keep alpha separate for the GUI
float alpha = 1.0f; //0.0-1.0
//contrast boost
int contrast_boost = 15; //0.0-255
int max_iterations = 500;
float center_x = -0.119609;
float center_y = 0.13262;
float zoom_factor = 1.0;
float zoom = 1.0;
float zoom_incr = 0.99;
long iterations = 0;
bool manual_navigation = false;

/** GL uniform handles **/
GLint base_color_hdl;
GLint contrast_boost_hdl;
GLint max_iterations_hdl;
GLint center_x_hdl;
GLint center_y_hdl;
GLint zoom_hdl;

/** shader and program handle **/
GLuint shader_program_hdl;

#ifndef V4D_USE_ES3
//vertex array
GLuint VAO;
#endif
GLuint VBO, EBO;

// vertex position, color
float vertices[] =
{
//    x      y      z
    -1.0f, -1.0f, -0.0f,
     1.0f,  1.0f, -0.0f,
    -1.0f,  1.0f, -0.0f,
     1.0f, -1.0f, -0.0f
};

unsigned int indices[] =
{
//  2---,1
//  | .' |
//  0'---3
    0, 1, 2,
    0, 3, 1
};

void load_buffer_data(){

#ifndef V4D_USE_ES3
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
#endif

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
#ifndef V4D_USE_ES3
    glBindVertexArray(0);
#endif
}

//workaround: required with emscripten + nanogui on every iteration before rendering
void rebind_buffers() {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint init_shader(const char* vShader, const char* fShader, const char* outputAttributeName) {
    struct Shader {
        GLenum       type;
        const char*      source;
    }  shaders[2] = {
        { GL_VERTEX_SHADER, vShader },
        { GL_FRAGMENT_SHADER, fShader }
    };

    GLuint program = glCreateProgram();

    for ( int i = 0; i < 2; ++i ) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader( s.type );
        glShaderSource( shader, 1, (const GLchar**) &s.source, NULL );
        glCompileShader( shader );

        GLint  compiled;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &compiled );
        if ( !compiled ) {
            std::cerr << " failed to compile:" << std::endl;
            GLint  logSize;
            glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logSize );
            char* logMsg = new char[logSize];
            glGetShaderInfoLog( shader, logSize, NULL, logMsg );
            std::cerr << logMsg << std::endl;
            delete [] logMsg;

            exit( EXIT_FAILURE );
        }

        glAttachShader( program, shader );
    }
#ifndef V4D_USE_ES3
    /* Link output */
    glBindFragDataLocation(program, 0, outputAttributeName);
#endif
    /* link  and error check */
    glLinkProgram(program);

    GLint  linked;
    glGetProgramiv( program, GL_LINK_STATUS, &linked );
    if ( !linked ) {
        std::cerr << "Shader program failed to link" << std::endl;
        GLint  logSize;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog( program, logSize, NULL, logMsg );
        std::cerr << logMsg << std::endl;
        delete [] logMsg;

        exit( EXIT_FAILURE );
    }

    /* use program object */
    glUseProgram(program);

    return program;
}

//mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
void load_shader(){
#ifndef V4D_USE_ES3
    const string shaderVersion = "330";
#else
    const string shaderVersion = "300 es";
#endif

    const string vert = "    #version " + shaderVersion + R"(
    in vec4 position;
    
    void main()
    {
        gl_Position = vec4(position.xyz, 1.0);
    })";

    const string frag = "    #version " + shaderVersion + R"(
    precision lowp float;

    out vec4 outColor;
    
    uniform vec4 base_color;
    uniform int contrast_boost;
    uniform int max_iterations;
    uniform float zoom;
    uniform float center_x;
    uniform float center_y;

    int get_iterations()
    {
        float pointr = (((gl_FragCoord.x / 1080.0f) - 0.5f) * zoom + center_x) * 5.0f;
        float pointi = (((gl_FragCoord.y / 1080.0f) - 0.5f) * zoom + center_y) * 5.0f;
        const float four = 4.0f;

        int iterations = 0;
        float zi = 0.0f;
        float zr = 0.0f;
        float zrsqr = 0.0f;
        float zisqr = 0.0f;

        while (iterations < max_iterations && zrsqr + zisqr < four) {
           //equals following line as a consequence of binomial expansion: zi = (((zr + zi)*(zr + zi)) - zrsqr) - zisqr
            zi = (zr + zr) * zi;

            zi += pointi;
            zr = (zrsqr - zisqr) + pointr;
    
            zrsqr = zr * zr;
            zisqr = zi * zi;
            ++iterations;
        }
        return iterations;
    }
     
    vec4 return_color()
    {
        int iter = get_iterations();
        if (iter == max_iterations) {   
            return vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        float iterations = float(iter) / float(max_iterations);
        //convert to float
        float cb = float(contrast_boost);

        return vec4(base_color[0] * iterations * cb, base_color[1] * iterations * cb, base_color[2] * iterations * cb, base_color[3]);
    }

    void main()
    {
        outColor = return_color();
    })";

    cerr << "##### Vertex Shader #####" << endl;
    cerr << vert << endl;

    cerr << "##### Fragment Shader #####" << endl;
    cerr << frag << endl;

    shader_program_hdl = init_shader(vert.c_str(),  frag.c_str(), "fragColor");
}

float easeInOutQuint(float x) {
    return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
}

void init_scene(const cv::Size& sz) {
    load_shader();
    load_buffer_data();

    base_color_hdl = glGetUniformLocation(shader_program_hdl, "base_color");
    contrast_boost_hdl = glGetUniformLocation(shader_program_hdl, "contrast_boost");
    max_iterations_hdl = glGetUniformLocation(shader_program_hdl, "max_iterations");
    zoom_hdl = glGetUniformLocation(shader_program_hdl, "zoom");
    center_x_hdl = glGetUniformLocation(shader_program_hdl, "center_x");
    center_y_hdl = glGetUniformLocation(shader_program_hdl, "center_y");

    glViewport(0, 0, WIDTH, HEIGHT);
}

void render_scene(const cv::Size& sz) {
//    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//    glClear(GL_COLOR_BUFFER_BIT);
    if(zoom >= 1) {
        zoom_incr = -0.01;
        iterations = 0;
    } else if(zoom < 2.5e-06) {
        zoom_incr = +0.01;
        iterations = 0;
    }

    glUseProgram(shader_program_hdl);
    glUniform4f(base_color_hdl, base_color_val[0], base_color_val[1], base_color_val[2], alpha);
    glUniform1i(contrast_boost_hdl, contrast_boost);
    glUniform1i(max_iterations_hdl, max_iterations);
    glUniform1f(center_y_hdl, center_y);
    glUniform1f(center_x_hdl, center_x);
    if(!manual_navigation) {
        zoom+=zoom_incr;
        glUniform1f(zoom_hdl, easeInOutQuint(zoom));
    } else {
        zoom = 1.0 / pow(zoom_factor,5.0f);
        glUniform1f(zoom_hdl, zoom);
    }

#ifndef V4D_USE_ES3
    glBindVertexArray(VAO);
#endif
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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

cv::Ptr<cv::viz::V4D> v4d = cv::viz::V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Shader Demo");

void setup_gui(cv::Ptr<cv::viz::V4D> v4d) {
    v4d->nanogui([](cv::viz::FormHelper& form){
    form.makeDialog(5, 30, "Fractal");

    form.makeGroup("Navigation");
    form.makeFormVariable("Iterations", max_iterations, 3, 1000000, true, "", "How deeply to calculate the fractal.");
    auto* cxVar = form.makeFormVariable("X", center_x, -1.0f, 1.0f, true, "", "The x location from -1.0 to 1.0");
    cxVar->number_format("%.7g");
    cxVar->set_value_increment(0.0000001);
    cxVar->set_callback([&, cxVar](const float& value){
        manual_navigation = true;
        cxVar->set_value(value);
        center_x = value;
    });

    auto* cyVar = form.makeFormVariable("Y", center_y, -1.0f, 1.0f, true, "", "The y location from -1.0 to 1.0");
    cyVar->number_format("%.7g");
    cyVar->set_value_increment(0.0000001);
    cyVar->set_callback([&,cyVar](const float &value) {
        manual_navigation = true;
        cyVar->set_value(value);
        center_y = value;
    });

    auto* czVar = form.makeFormVariable("Zoom", zoom_factor, 1.0f, 1000000.0f, true, "", "How much to zoom in on the fractal");
    czVar->set_callback([&,czVar](const float &value) {
        manual_navigation = true;
        czVar->set_value(value);
        zoom_factor = value;
    });

#ifndef __EMSCRIPTEN__
    form.makeGroup("Glow");
    auto* kernelSize = form.makeFormVariable("Kernel Size", glow_kernel_size, 1, 127, true, "", "Intensity of glow defined by kernel size");
    kernelSize->set_callback([=](const int& k) {
        static int lastKernelSize = glow_kernel_size;

        if(k == lastKernelSize)
            return;

        if(k <= lastKernelSize) {
            glow_kernel_size = std::max(int(k % 2 == 0 ? k - 1 : k), 1);
        } else if(k > lastKernelSize)
            glow_kernel_size = std::max(int(k % 2 == 0 ? k + 1 : k), 1);

        lastKernelSize = k;
        kernelSize->set_value(glow_kernel_size);
    });
#endif
    form.makeGroup("Color");
    form.makeColorPicker("Color", base_color_val, "The base color of the fractal visualization",[&](const nanogui::Color &c) {
        base_color_val[0] = c[0];
        base_color_val[1] = c[1];
        base_color_val[2] = c[2];
    });
    form.makeFormVariable("Alpha", alpha, 0.0f, 1.0f, true, "", "The opacity of the fractal visualization");
    form.makeFormVariable("Contrast boost", contrast_boost, 1, 255, true, "", "Boost contrast by this factor");
    });
}

bool iteration() {
    //ignore failed capture attempts
    v4d->capture();

#ifdef __EMSCRIPTEN__
    //required in conjunction with emscripten + nanovg
    rebind_buffers();
#endif
    //Render using OpenGL
    v4d->gl(render_scene);

//To slow for WASM but works
#ifndef __EMSCRIPTEN__
    //Aquire the frame buffer for use by OpenCL
    v4d->fb([](cv::UMat &frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
    });
#endif

    updateFps(v4d, true);

#ifndef __EMSCRIPTEN__
    v4d->write();
#endif
    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if (!v4d->display())
        return false;

    ++iterations;
    return true;
}

int main(int argc, char **argv) {
    using namespace cv::viz;
    try {
        if(argc != 2) {
            cerr << "Usage: shader-demo <video-file>" << endl;
            exit(1);
        }

        printSystemInfo();
        if(!v4d->isOffscreen()) {
            setup_gui(v4d);
            v4d->setVisible(true);
        }
        v4d->gl(init_scene);

#ifndef __EMSCRIPTEN__
        Source src = makeCaptureSource(argv[1]);
        v4d->setSource(src);
        Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT));
        v4d->setSink(sink);
#else
        Source src = makeCaptureSource(WIDTH, HEIGHT);
        v4d->setSource(src);
#endif

        v4d->run(iteration);
    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
