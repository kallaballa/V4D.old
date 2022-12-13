#include "nanovgcontext.hpp"

#include "viz2d.hpp"

namespace kb {

NanoVGContext::NanoVGContext(Viz2D &v2d, NVGcontext *context, CLGLContext &fbContext) :
        v2d_(v2d), context_(context), clglContext_(fbContext) {
    nvgCreateFont(context_, "libertine", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for first frame color glitch
    cv::UMat tmp;
    clglContext_.acquireFromGL(tmp);
    clglContext_.releaseToGL(tmp);
}

void NanoVGContext::render(std::function<void(NVGcontext*, const cv::Size&)> fn) {
    CLExecScope_t scope(clglContext_.getCLExecContext());
    begin();
    fn(context_, clglContext_.getSize());
    end();
}

void NanoVGContext::begin() {
    clglContext_.begin();
    float w = v2d_.getVideoFrameSize().width;
    float h = v2d_.getVideoFrameSize().height;
    float r = v2d_.getXPixelRatio();

    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
    GL_CHECK(glViewport(0, 0,w,h));
}

void NanoVGContext::end() {
    nvgEndFrame(context_);
    nvgRestore(context_);
    clglContext_.end();
}
} /* namespace kb */