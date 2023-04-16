// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "detail/clvacontext.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/nanovgcontext.hpp"

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif

namespace cv {
namespace viz {
namespace detail {
void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
    int errorCode = glGetError();

    if (errorCode != 0) {
        std::cerr << "GL failed in " << file.filename() << " (" << line << ") : "
                << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode
                << "\n   " << std::endl;
        assert(false);
    }
}

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

bool contains_absolute(nanogui::Widget* w, const nanogui::Vector2i& p) {
    nanogui::Vector2i d = p - w->absolute_position();
    return d.x() >= 0 && d.y() >= 0 && d.x() < w->size().x() && d.y() < w->size().y();
}
}

cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code) {
    cv::Mat tmpIn(1, 1, CV_8UC3);
    cv::Mat tmpOut(1, 1, CV_8UC3);

    tmpIn.at<cv::Vec3b>(0, 0) = cv::Vec3b(src[0], src[1], src[2]);
    cvtColor(tmpIn, tmpOut, code);
    const cv::Vec3b& vdst = tmpOut.at<cv::Vec3b>(0, 0);
    cv::Scalar dst(vdst[0], vdst[1], vdst[2], src[3]);
    return dst;
}

void resizeKeepAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize,
        const cv::Scalar& bgcolor) {
    double h1 = dstSize.width * (src.rows / (double) src.cols);
    double w2 = dstSize.height * (src.cols / (double) src.rows);
    if (h1 <= dstSize.height) {
        cv::resize(src, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize(src, output, cv::Size(w2, dstSize.height));
    }

    int top = (dstSize.height - output.rows) / 2;
    int down = (dstSize.height - output.rows + 1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols + 1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const string& title, bool debug) {
    cv::Ptr<V4D> v4d = new V4D(size, size, false, title, 4, 6, true, 0, debug);
    v4d->setVisible(true);
    return v4d;
}

cv::Ptr<V4D> V4D::make(const cv::Size& initialSize, const cv::Size& frameBufferSize,
        bool offscreen, const string& title, int major, int minor, bool compat, int samples, bool debug) {
    return new V4D(initialSize, frameBufferSize, offscreen, title, major, minor, compat, samples, debug);
}

V4D::V4D(const cv::Size& size, const cv::Size& frameBufferSize, bool offscreen,
        const string& title, int major, int minor, bool compat, int samples, bool debug) :
        initialSize_(size), frameBufferSize_(frameBufferSize), viewport_(0, 0,
                frameBufferSize.width, frameBufferSize.height), scale_(1), mousePos_(0, 0), offscreen_(
                offscreen), stretch_(false), title_(title), major_(major), minor_(minor), compat_(compat), samples_(
                samples), debug_(debug) {
    assert(
            frameBufferSize_.width >= initialSize_.width
                    && frameBufferSize_.height >= initialSize_.height);

    initializeWindowing();
}

V4D::~V4D() {
    //don't delete form_. it is autmatically cleaned up by the base class (nanogui::Screen)
    if (screen_)
        delete screen_;
    if (writer_)
        delete writer_;
    if (capture_)
        delete capture_;
    if (nvgContext_)
        delete nvgContext_;
    if (clvaContext_)
        delete clvaContext_;
    if (clglContext_)
        delete clglContext_;
}

bool V4D::initializeWindowing() {
    if (glfwInit() != GLFW_TRUE)
        return false;

    glfwSetErrorCallback(cv::viz::glfw_error_callback);

    if (debug_)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    if (offscreen_)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwSetTime(0);

#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#elif defined(V4D_USE_ES3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API) ;
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_);
    glfwWindowHint(GLFW_OPENGL_PROFILE, compat_ ? GLFW_OPENGL_COMPAT_PROFILE : GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API) ;
#endif
    glfwWindowHint(GLFW_SAMPLES, samples_);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#else
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
#endif
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    /* I figure we don't need double buffering because the FBO (and the bound texture) is our backbuffer that
     * we blit to the front on every iteration.
     * On X11, wayland and in WASM it works and boosts performance a bit.
     */
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);

    glfwWindow_ = glfwCreateWindow(initialSize_.width, initialSize_.height, title_.c_str(), nullptr,
            nullptr);
    if (glfwWindow_ == NULL) {
        return false;
    }
    glfwMakeContextCurrent(getGLFWWindow());

    screen_ = new nanogui::Screen();
    screen().initialize(getGLFWWindow(), false);
    form_ = new FormHelper(&screen());

    this->setWindowSize(initialSize_);

    glfwSetWindowUserPointer(getGLFWWindow(), this);

    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        v4d->screen().cursor_pos_callback_event(x, y);
        auto cursor = v4d->getMousePosition();
        auto diff = cursor - cv::Vec2f(x, y);
        if (v4d->isMouseDrag()) {
            v4d->pan(diff[0], -diff[1]);
        }
        v4d->setMousePosition(x, y);
    }
    );
    glfwSetMouseButtonCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->screen().mouse_button_callback_event(button, action, modifiers);
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    v4d->setMouseDrag(action == GLFW_PRESS);
                }
            }
    );
    glfwSetKeyCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->screen().key_callback_event(key, scancode, action, mods);
            }
    );
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        v4d->screen().char_callback_event(codepoint);
    }
    );
    glfwSetDropCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int count, const char** filenames) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->screen().drop_callback_event(count, filenames);
            }
    );
    glfwSetScrollCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        std::vector<nanogui::Widget*> widgets;
        find_widgets(&v4d->screen(), widgets);
        for (auto* w : widgets) {
            auto mousePos = nanogui::Vector2i(v4d->getMousePosition()[0] / v4d->getXPixelRatio(), v4d->getMousePosition()[1] / v4d->getYPixelRatio());
    if(contains_absolute(w, mousePos)) {
        v4d->screen().scroll_callback_event(x, y);
        return;
    }
}

        v4d->zoom(y < 0 ? 1.1 : 0.9);
    }
    );

//FIXME resize internal buffers?
//    glfwSetWindowContentScaleCallback(getGLFWWindow(),
//        [](GLFWwindow* glfwWin, float xscale, float yscale) {
//        }
//    );

    glfwSetFramebufferSizeCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int width, int height) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        v4d->screen().resize_callback_event(width, height);
    });

    clglContext_ = new detail::FrameBufferContext(this->getFrameBufferSize());
    clvaContext_ = new detail::CLVAContext(*clglContext_);
    nvgContext_ = new detail::NanoVGContext(*this, getNVGcontext(), *clglContext_);
    return true;
}

cv::ogl::Texture2D& V4D::texture() {
    return clglContext_->getTexture2D();
}

FormHelper& V4D::form() {
    return *form_;
}

void V4D::setKeyboardEventCallback(
        std::function<bool(int key, int scancode, int action, int modifiers)> fn) {
    keyEventCb_ = fn;
}

bool V4D::keyboard_event(int key, int scancode, int action, int modifiers) {
    if (keyEventCb_)
        return keyEventCb_(key, scancode, action, modifiers);

    return screen().keyboard_event(key, scancode, action, modifiers);
}

FrameBufferContext& V4D::fb() {
    assert(clglContext_ != nullptr);
    makeCurrent();
    return *clglContext_;
}

CLVAContext& V4D::clva() {
    assert(clvaContext_ != nullptr);
    return *clvaContext_;
}

NanoVGContext& V4D::nvg() {
    assert(nvgContext_ != nullptr);
    makeCurrent();
    return *nvgContext_;
}

nanogui::Screen& V4D::screen() {
    assert(screen_ != nullptr);
    makeCurrent();
    return *screen_;
}

cv::Size V4D::getVideoFrameSize() {
    return clva().getVideoFrameSize();
}

void V4D::gl(std::function<void(const cv::Size&)> fn) {
    auto fbSize = getFrameBufferSize();
#ifndef __EMSCRIPTEN__
    detail::CLExecScope_t scope(fb().getCLExecContext());
#endif
    detail::FrameBufferContext::GLScope glScope(fb());
    fn(fbSize);
}

void V4D::fb(std::function<void(cv::UMat&)> fn) {
    fb().execute(fn);
}

void V4D::nvg(std::function<void(const cv::Size&)> fn) {
    nvg().render(fn);
}

void V4D::nanogui(std::function<void(FormHelper& form)> fn) {
    fn(form());
    screen().set_visible(true);
    screen().perform_layout();
}

void V4D::run(std::function<bool()> fn) {
#ifndef __EMSCRIPTEN__
    while (keepRunning() && fn())
        ;
#else
    emscripten_set_main_loop(fn, -1, true);
#endif
}

void V4D::setSource(const Source& src) {
    if (!clva().hasContext())
        clva().copyContext();
    source_ = src;
}

void V4D::feed(cv::InputArray& in) {
    this->capture([&](cv::OutputArray& videoFrame) {
        in.getUMat().copyTo(videoFrame);
    });
}

bool V4D::capture() {
    return this->capture([&](cv::UMat& videoFrame) {
        if (source_.isReady())
            source_().second.copyTo(videoFrame);
    });
}

bool V4D::capture(std::function<void(cv::UMat&)> fn) {
    if(futureReader_.valid())
        if(!futureReader_.get())
            return false;

    if(nextReaderFrame_.empty()) {
        if(!clva().capture(fn, nextReaderFrame_))
            return false;
    }

    currentReaderFrame_ = nextReaderFrame_;
    {
        FrameBufferContext::GLScope glScope(*clglContext_);
        FrameBufferContext::FrameBufferScope fbScope(*clglContext_, readerFrameBuffer_);

        currentReaderFrame_.copyTo(readerFrameBuffer_);
    }

    futureReader_ = pool.push([=,this](){
        return clva().capture(fn, nextReaderFrame_);
    });
    return captureSuccessful_;
}

bool V4D::isSourceReady() {
    return source_.isReady();
}

void V4D::setSink(const Sink& sink) {
    if (!clva().hasContext())
        clva().copyContext();
    sink_ = sink;
}

void V4D::write() {
    this->write([&](const cv::UMat& videoFrame) {
        if (sink_.isReady())
            sink_(videoFrame);
    });
}

void V4D::write(std::function<void(const cv::UMat&)> fn) {
    if(futureWriter_.valid())
        futureWriter_.get();

    {
        FrameBufferContext::GLScope glScope(*clglContext_);
        FrameBufferContext::FrameBufferScope fbScope(*clglContext_, writerFrameBuffer_);

        writerFrameBuffer_.copyTo(currentWriterFrame_);
    }
    futureWriter_ = pool.push([=,this](){
        clva().write(fn, currentWriterFrame_);
    });
}

bool V4D::isSinkReady() {
    return sink_.isReady();
}

void V4D::makeCurrent() {
    glfwMakeContextCurrent(getGLFWWindow());
}

void V4D::makeNoneCurrent() {
    glfwMakeContextCurrent(nullptr);
}

void V4D::clear(const cv::Scalar& bgra) {
    const float& b = bgra[0] / 255.0f;
    const float& g = bgra[1] / 255.0f;
    const float& r = bgra[2] / 255.0f;
    const float& a = bgra[3] / 255.0f;
    GL_CHECK(glClearColor(r, g, b, a));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}

void V4D::showGui(bool s) {
    auto children = screen().children();
    for (auto* child : children) {
        child->set_visible(s);
    }
}

void V4D::setMouseDrag(bool d) {
    mouseDrag_ = d;
}

bool V4D::isMouseDrag() {
    return mouseDrag_;
}

void V4D::pan(int x, int y) {
    viewport_.x += x * scale_;
    viewport_.y += y * scale_;
}

void V4D::zoom(float factor) {
    if (scale_ == 1 && viewport_.x == 0 && viewport_.y == 0 && factor > 1)
        return;

    double oldScale = scale_;
    double origW = getFrameBufferSize().width;
    double origH = getFrameBufferSize().height;

    scale_ *= factor;
    if (scale_ <= 0.025) {
        scale_ = 0.025;
        return;
    } else if (scale_ > 1) {
        scale_ = 1;
        viewport_.width = origW;
        viewport_.height = origH;
        if (factor > 1) {
            viewport_.x += log10(((viewport_.x * (1.0 - factor)) / viewport_.width) * 9 + 1.0)
                    * viewport_.width;
            viewport_.y += log10(((viewport_.y * (1.0 - factor)) / viewport_.height) * 9 + 1.0)
                    * viewport_.height;
        } else {
            viewport_.x += log10(((-viewport_.x * (1.0 - factor)) / viewport_.width) * 9 + 1.0)
                    * viewport_.width;
            viewport_.y += log10(((-viewport_.y * (1.0 - factor)) / viewport_.height) * 9 + 1.0)
                    * viewport_.height;
        }
        return;
    }

    cv::Vec2f offset;
    double oldW = (origW * oldScale);
    double oldH = (origH * oldScale);
    viewport_.width = std::min(scale_ * origW, origW);
    viewport_.height = std::min(scale_ * origH, origH);

    float delta_x;
    float delta_y;

    if (factor < 1.0) {
        offset = cv::Vec2f(viewport_.x, viewport_.y)
                - cv::Vec2f(mousePos_[0], origH - mousePos_[1]);
        delta_x = offset[0] / oldW;
        delta_y = offset[1] / oldH;
    } else {
        offset = cv::Vec2f(viewport_.x - (viewport_.width / 2.0),
                viewport_.y - (viewport_.height / 2.0)) - cv::Vec2f(viewport_.x, viewport_.y);
        delta_x = offset[0] / oldW;
        delta_y = offset[1] / oldH;
    }

    float x_offset;
    float y_offset;
    x_offset = delta_x * (viewport_.width - oldW);
    y_offset = delta_y * (viewport_.height - oldH);

    if (factor < 1.0) {
        viewport_.x += x_offset;
        viewport_.y += y_offset;
    } else {
        viewport_.x += x_offset;
        viewport_.y += y_offset;
    }
}

cv::Vec2f V4D::getPosition() {
    makeCurrent();
    int x, y;
    glfwGetWindowPos(getGLFWWindow(), &x, &y);
    return {float(x), float(y)};
}

cv::Vec2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(int x, int y) {
    mousePos_ = { float(x), float(y) };
}

float V4D::getScale() {
    return scale_;
}

cv::Rect V4D::getViewport() {
    return viewport_;
}

cv::Size V4D::getNativeFrameBufferSize() {
    makeCurrent();
    int w, h;
    glfwGetFramebufferSize(getGLFWWindow(), &w, &h);
    return {w, h};
}

cv::Size V4D::getFrameBufferSize() {
    return frameBufferSize_;
}

cv::Size V4D::getWindowSize() {
    makeCurrent();
    int w, h;
    glfwGetWindowSize(getGLFWWindow(), &w, &h);
    return {w, h};
}

cv::Size V4D::getInitialSize() {
    return initialSize_;
}

float V4D::getXPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    return emscripten_get_device_pixel_ratio();
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    return xscale;
#endif
}

float V4D::getYPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    return emscripten_get_device_pixel_ratio();
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    return yscale;
#endif
}

void V4D::setWindowSize(const cv::Size& sz) {
    makeCurrent();
    screen().set_size(nanogui::Vector2i(sz.width / getXPixelRatio(), sz.height / getYPixelRatio()));
}

bool V4D::isFullscreen() {
    makeCurrent();
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void V4D::setFullscreen(bool f) {
    makeCurrent();
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (f) {
        glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height,
                mode->refreshRate);
        setWindowSize(getNativeFrameBufferSize());
    } else {
        glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getInitialSize().width,
                getInitialSize().height, 0);
        setWindowSize(getInitialSize());
    }
}

bool V4D::isResizable() {
    makeCurrent();
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
}

void V4D::setResizable(bool r) {
    makeCurrent();
    glfwWindowHint(GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
}

bool V4D::isVisible() {
    makeCurrent();
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_VISIBLE) == GLFW_TRUE;
}

void V4D::setVisible(bool v) {
    makeCurrent();
    glfwWindowHint(GLFW_VISIBLE, v ? GLFW_TRUE : GLFW_FALSE);
    screen().set_visible(v);
    screen().perform_layout();
}

bool V4D::isOffscreen() {
    return offscreen_;
}

void V4D::setOffscreen(bool o) {
    offscreen_ = o;
    setVisible(!o);
}

void V4D::setStretching(bool s) {
    stretch_ = s;
}

bool V4D::isStretching() {
    return stretch_;
}

void V4D::setDefaultKeyboardEventCallback() {
    setKeyboardEventCallback([&](int key, int scancode, int action, int modifiers) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            setOffscreen(!isOffscreen());
            return true;
        } else if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
            auto children = screen().children();
            for (auto* child : children) {
                child->set_visible(!child->visible());
            }

            return true;
        }
        return false;
    });
}

bool V4D::display() {
    bool result = true;
    if (!offscreen_) {
        makeCurrent();
        screen().draw_contents();
#ifndef __EMSCRIPTEN__
        clglContext_->blitFrameBufferToScreen(getViewport(), getWindowSize(), isStretching());
#else
        clglContext_->blitFrameBufferToScreen(getViewport(), getInitialSize(), isStretching());
#endif
        screen().draw_widgets();
        glfwSwapBuffers(glfwWindow_);
        glfwPollEvents();
        result = !glfwWindowShouldClose(glfwWindow_);
    }

    return result;
}

bool V4D::isClosed() {
    return closed_;

}
void V4D::close() {
    setVisible(false);
    closed_ = true;
}

GLFWwindow* V4D::getGLFWWindow() {
    return glfwWindow_;
}

NVGcontext* V4D::getNVGcontext() {
    return screen().nvg_context();
}
}
}
