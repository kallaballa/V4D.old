#ifndef SRC_COMMON_VIZ2DWORKER_HPP_
#define SRC_COMMON_VIZ2DWORKER_HPP_

#include <filesystem>
#include <typeindex>
#include <typeinfo>
#include <thread>
#include <any>
#include <iostream>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <nanogui/nanogui.h>
#ifndef __EMSCRIPTEN__
#include <source_location>
#include <GL/glew.h>
#else
#include <experimental/source_location>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <emscripten.h>
#endif

#include "task.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

namespace kb {
namespace viz2d {
class NVG;
class Viz2DWorker;
cv::Scalar color_convert(const cv::Scalar& src, cv::ColorConversionCodes code);

namespace detail {
class CLGLContext;
class CLVAContext;
class NanoVGContext;


void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression);

#define GL_CHECK(expr)                            \
    expr;                                        \
    kb::viz2d::gl_check_error(__FILE__, __LINE__, #expr);

void error_callback(int error, const char *description);

class MemoryPool {
    std::map<std::thread::id, std::map<string, cv::UMat>> globalMap_;
    std::map<std::thread::id, std::map<string, cv::UMat>> localMap_;
    std::map<std::thread::id, std::map<string, void*>> globalVarMap_;
    std::map<std::thread::id, std::map<string, void*>> localVarMap_;
    static std::map<string, cv::UMat> sharedMap_;
    static std::map<string, void*> sharedVarMap_;

    std::map<string, cv::UMat>& getSharedMap() {
        return sharedMap_;
    }

    std::map<string, cv::UMat>& getGlobalMap() {
        return globalMap_[std::this_thread::get_id()];
    }

    std::map<string, cv::UMat>& getLocalMap() {
        return localMap_[std::this_thread::get_id()];
    }

    std::map<string, void*>& getSharedVarMap() {
        return sharedVarMap_;
    }

    std::map<string, void*>& getGlobalVarMap() {
        return globalVarMap_[std::this_thread::get_id()];
    }

    std::map<string, void*>& getLocalVarMap() {
        return localVarMap_[std::this_thread::get_id()];
    }
public:
    cv::UMat& allocateShared(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        auto& buf = getSharedMap()[name];
        if(buf.empty())
            buf.create(sz, type, usageFlags);

        return buf;
    }

    cv::UMat& allocateGlobal(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        auto& buf = getGlobalMap()[name];
        if(buf.empty())
            buf.create(sz, type, usageFlags);

        return buf;
    }

    cv::UMat& allocateLocal(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        auto& buf = getGlobalMap()[name];
        if(buf.empty())
            buf.create(sz, type, usageFlags);

        return buf;
    }

    cv::UMat& shared(const string& name) {
        return getSharedMap()[name];
    }

    cv::UMat& global(const string& name) {
        return getGlobalMap()[name];
    }

    cv::UMat& local(const string& name) {
        return getLocalMap()[name];
    }

    template <typename T> T& shared(const string& name) {
        auto& map = getSharedVarMap();
        auto it = map.find(name);
        if(it != map.end()) {
            return *static_cast<T*>((*it).second);
        } else {
            auto* p = new T();
            map[name] = p;
            return *p;
        }
    }

    template <typename T> T& global(const string& name) {
        auto& map = getGlobalVarMap();
        auto it = map.find(name);
        if(it != map.end()) {
            return *static_cast<T*>((*it).second);
        } else {
            auto* p = new T();
            map[name] = p;
            return *p;
        }
    }

    template <typename T> T& local(const string& name, const T& init) {
        auto& map = getLocalVarMap();
        auto it = map.find(name);
        if(it != map.end()) {
            return *static_cast<T*>((*it).second);
        } else {
            auto* p = new T(init);
            map[name] = p;
            return *p;
        }
    }

    template <typename T> T& local(const string& name) {
        auto& map = getLocalVarMap();
        auto it = map.find(name);
        if(it != map.end()) {
            return *static_cast<T*>((*it).second);
        } else {
            auto* p = new T();
            map[name] = p;
            return *p;
        }
    }
};


class Property {
    friend class kb::viz2d::Viz2DWorker;
public:
    string name_;
    string label_;
    std::any value_;
    std::any min_;
    std::any max_;

    Property(const string &name, const string &label, const std::any &value, const std::any &min, const std::any &max) :
    name_(name), label_(label), value_(value), min_(min), max_(max){
    }

    template<typename T>
    void propagate(const T& v) {
        value_ = v;
        assert(fn_);
        fn_(std::any(v));
    }
protected:
    std::function<void(std::any)> fn_;
};

class Properties {
    std::map<std::string, std::any*> propMap_;
public:
    Properties() {
    }

    template<typename Tval> Property& create(const string &name, const string &label, const Tval& value, const Tval &min, const Tval &max) {
        propMap_[name] = new std::any(Property(name, label, value, min, max));
        return property(name);
    }

    template<typename Tenum> Property& create(const string &name, const string &label, const Tenum& e) {
        propMap_[name] = new std::any(Property(name, label, e, e, e));
        return property(name);
    }

    Property& create(const string &name, const string &label, const nanogui::Color& color) {
        propMap_[name] = new std::any(Property(name, label, color, nanogui::Color(0,0,0,0), nanogui::Color(255,255,255,255)));
        return property(name);
    }

    Property& create(const string &name, const string &label, const bool& b) {
        propMap_[name] = new std::any(Property(name, label, b, false, true));
        return property(name);
    }

    template<typename T> Property& create(const string &name, const string &label, T& value, T& min, T& max) {
        propMap_[name] = new std::any(Property(name, label, value, min, max));
        return property(name);
    }

    Property& property(const string& name) {
        return std::any_cast<Property&>(*(propMap_[name]));
    }

    std::vector<std::string> names() {
        std::vector<std::string> result;
        std::transform(
                propMap_.begin(),
                propMap_.end(),
            std::back_inserter(result),
            [](auto &kv){ return kv.first;}
        );
        return result;
    }
    template <typename T, typename std::enable_if<std::is_enum<T>::value >::type* = nullptr>
    void propagate(const string& name, const T& newVal, double scale = 1) {
        Property& prop = property(name);
        std::any& val = prop.value_;
        std::any& min = prop.min_;
        std::any& max = prop.max_;
        int& propVal = std::any_cast<int&>(val);
        int& propMin = std::any_cast<int&>(min);
        int& propMax = std::any_cast<int&>(max);

        propVal = propMin + (propVal / scale) * (propMax - propMin);

        T& enumVal = *reinterpret_cast<T*>(&propVal);
        prop.propagate<T>(enumVal);
    }

    template <typename T, typename std::enable_if<!std::is_enum<T>::value >::type* = nullptr>
    void propagate(const string& name, const T& newVal, double scale = 1) {
        Property& prop = property(name);
        std::any& val = prop.value_;
        std::any& min = prop.min_;
        std::any& max = prop.max_;
        if (auto x = std::any_cast<int8_t>(&val)) {
            auto minVal = std::any_cast<int8_t>(min);
            auto maxVal = std::any_cast<int8_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<int8_t>(*x);
        } else if (auto x = std::any_cast<uint8_t>(&val)) {
            auto minVal = std::any_cast<uint8_t>(min);
            auto maxVal = std::any_cast<uint8_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<uint8_t>(*x);
        } else if (auto x = std::any_cast<int16_t>(&val)) {
            auto minVal = std::any_cast<int16_t>(min);
            auto maxVal = std::any_cast<int16_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<int16_t>(*x);
        } else if (auto x = std::any_cast<uint16_t>(&val)) {
            auto minVal = std::any_cast<uint16_t>(min);
            auto maxVal = std::any_cast<uint16_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<uint16_t>(*x);
        } else if (auto x = std::any_cast<int32_t>(&val)) {
            auto minVal = std::any_cast<int32_t>(min);
            auto maxVal = std::any_cast<int32_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<int32_t>(*x);
        } else if (auto x = std::any_cast<uint32_t>(&val)) {
            auto minVal = std::any_cast<uint32_t>(min);
            auto maxVal = std::any_cast<uint32_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<uint32_t>(*x);
        } else if (auto x = std::any_cast<int64_t>(&val)) {
            auto minVal = std::any_cast<int64_t>(min);
            auto maxVal = std::any_cast<int64_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<int64_t>(*x);
        } else if (auto x = std::any_cast<uint64_t>(&val)) {
            auto minVal = std::any_cast<uint64_t>(min);
            auto maxVal = std::any_cast<uint64_t>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<uint64_t>(*x);
        } else if (auto x = std::any_cast<float>(&val)) {
            auto minVal = std::any_cast<float>(min);
            auto maxVal = std::any_cast<float>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<float>(*x);
        } else if (auto x = std::any_cast<double>(&val)) {
            auto minVal = std::any_cast<double>(min);
            auto maxVal = std::any_cast<double>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<double>(*x);
        } else if (auto x = std::any_cast<long double>(&val)) {
            auto minVal = std::any_cast<long double>(min);
            auto maxVal = std::any_cast<long double>(max);
            *x = minVal + ((newVal / scale) * (maxVal - minVal));
            prop.propagate<long double>(*x);
        } else if (auto x = std::any_cast<nanogui::Color>(&val)) {
            cv::Scalar rgb = kb::viz2d::color_convert(cv::Scalar((newVal / scale) * 255, 127, 127, 255), cv::COLOR_HLS2RGB);
            nanogui::Color color(rgb[0], rgb[1], rgb[2], 255);
            prop.propagate<nanogui::Color>(color);
        }
    }
};
}

using namespace kb::viz2d::detail;

class Storage {
    detail::Properties properties_;
    detail::MemoryPool memPool_;
public:
    Properties& properties() {
        return properties_;
    }

    template <typename T>
    void propagate(const string& name, const T& value, double scale = 1) {
        properties_.propagate<T>(name, value, scale);
    }

    template<typename T, typename std::enable_if<std::is_enum<T>::value >::type* = nullptr> T& property(const string& name) {
        int& propVal = std::any_cast<int&>(properties_.property(name).value_);
        return *reinterpret_cast<T*>(&propVal);
    }

    template<typename T, typename std::enable_if<!std::is_enum<T>::value >::type* = nullptr> T& property(const string& name) {
        return std::any_cast<T&>(properties_.property(name).value_);
    }

    cv::UMat& allocSharedOutput(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        return memPool_.allocateShared(name, sz, type, defaultValue, usageFlags);
    }

    const cv::UMat& allocSharedInput(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        return memPool_.allocateShared(name, sz, type, defaultValue, usageFlags);
    }

    cv::UMat& allocOutput(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        return memPool_.allocateGlobal(name, sz, type, defaultValue, usageFlags);
    }

    const cv::UMat& allocInput(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT) {
        return memPool_.allocateGlobal(name, sz, type, defaultValue, usageFlags);
    }

    cv::UMat& allocLocal(const string& name, const cv::Size& sz, int type, const cv::Scalar& defaultValue, cv::UMatUsageFlags usageFlags = cv::USAGE_DEFAULT, const std::source_location& loc = std::source_location::current()) {
        std::ostringstream ss;
        ss << loc.file_name() << "("
                << loc.line() << ":"
                << loc.column() << ") `"
                << loc.function_name() << "`: "
                << name;
        return memPool_.allocateLocal(ss.str(), sz, type, defaultValue, usageFlags);
    }

    cv::UMat& sharedOutput(const string& name) {
        return memPool_.shared(name);
    }

    const cv::UMat& sharedInput(const string& name) {
        return memPool_.shared(name);
    }

    cv::UMat& output(const string& name) {
        return memPool_.global(name);
    }

    const cv::UMat& input(const string& name) {
        return memPool_.global(name);
    }

    cv::UMat& local(const string& name, const std::source_location& loc = std::source_location::current()) {
        std::ostringstream ss;
        ss << loc.file_name() << "("
                << loc.line() << ":"
                << loc.column() << ") `"
                << loc.function_name() << "`: "
                << name;
        return memPool_.local(ss.str());
    }

    template <typename T> T& sharedOutput(const string& name) {
        return memPool_.shared<T>(name);
    }

    template <typename T> const T& sharedInput(const string& name) {
        return memPool_.shared<T>(name);
    }

    template <typename T> T& output(const string& name) {
        return memPool_.global<T>(name);
    }

    template <typename T> const T& input(const string& name) {
        return memPool_.global<T>(name);
    }

    template <typename T> T& local(const string& name, const T& init, const std::source_location& loc = std::source_location::current()) {
        std::ostringstream ss;
        ss << loc.file_name() << "("
                << loc.line() << ":"
                << loc.column() << ") `"
                << loc.function_name() << "`: "
                << name;
        return memPool_.local<T>(ss.str(), init);
    }

    template <typename T> T& local(const string& name, const std::source_location& loc = std::source_location::current()) {
        std::ostringstream ss;
        ss << loc.file_name() << "("
                << loc.line() << ":"
                << loc.column() << ") `"
                << loc.function_name() << "`: "
                << name;
        return memPool_.local<T>(ss.str());
    }
};

class Viz2DWindow : public nanogui::Window {
private:
    static std::function<bool(Viz2DWindow*, Viz2DWindow*)> viz2DWin_Xcomparator;
    static std::set<Viz2DWindow*, decltype(viz2DWin_Xcomparator)> all_windows_xsorted_;
    nanogui::Screen* screen_;
    nanogui::Vector2i lastDragPos_;
    nanogui::Vector2i maximizedPos_;
    nanogui::Button* minBtn_;
    nanogui::Button* maxBtn_;
    nanogui::ref<nanogui::AdvancedGridLayout> oldLayout_;
    nanogui::ref<nanogui::AdvancedGridLayout> newLayout_;
    bool minimized_ = false;
public:
    Viz2DWindow(nanogui::Screen* screen, int x, int y, const string& title);
    virtual ~Viz2DWindow();
    bool isMinimized();
    bool mouse_drag_event(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int mods) override;
};

class Viz2DWorker {
    friend class Viz2D;
    friend class detail::Property;
    friend class NanoVGContext;
private:
    const cv::Size initialSize_;
    cv::Size frameBufferSize_;
    cv::Rect viewport_;
    float scale_;
    cv::Vec2f mousePos_;
    bool offscreen_;
    bool stretch_;
    string title_;
    int major_;
    int minor_;
    int samples_;
    bool debug_;
    std::filesystem::path capturePath_;
    std::filesystem::path writerPath_;
    GLFWwindow* glfwWindow_ = nullptr;
    CLGLContext* clglContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    cv::VideoCapture* capture_ = nullptr;
    cv::VideoWriter* writer_ = nullptr;
    nanogui::FormHelper* form_ = nullptr;
    bool closed_ = false;
    cv::Size videoFrameSize_ = cv::Size(0,0);
    int vaCaptureDeviceIndex_ = 0;
    int vaWriterDeviceIndex_ = 0;
    bool mouseDrag_ = false;
    nanogui::Screen* screen_ = nullptr;
    uint64_t frameCnt_ = 0;
    Storage storage_;
public:
    Viz2DWorker(const cv::Size &initialSize, const cv::Size& frameBufferSize, bool offscreen, const string &title, int major = 4, int minor = 6, int samples = 0, bool debug = false);
    virtual ~Viz2DWorker();
    bool initializeWindowing();
    void makeCurrent();
    void makeNonCurrent();

    cv::ogl::Texture2D& texture();

    Task gl(const string& name, std::function<void(Storage&, const cv::Size&)> fn);
    Task cl(const string& name, std::function<void(Storage&)> fn);
    Task cpu(const string& name, std::function<void(Storage&)> fn);
    Task clgl(const string& name, std::function<void(Storage&, cv::UMat&)> fn);
    Task nvg(const string& name, std::function<void(Storage&, const cv::Size&)> fn);

    virtual bool capture();
    virtual bool capture(std::function<void(cv::UMat&)> fn);
    virtual bool hasCapture();
    virtual void write();
    virtual void write(std::function<void(const cv::UMat&)> fn);
    virtual bool hasWriter();
    virtual cv::VideoWriter& makeVAWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize, const int vaDeviceIndex);
    virtual cv::VideoCapture& makeVACapture(const string& intputFilename, const int vaDeviceIndex);
    virtual cv::VideoWriter& makeWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize);
    virtual cv::VideoCapture& makeCapture(const string& intputFilename);
    void setMouseDrag(bool d);
    bool isMouseDrag();
    void pan(int x, int y);
    void zoom(float factor);
    cv::Vec2f getPosition();
    cv::Vec2f getMousePosition();
    float getScale();
    cv::Rect getViewport();
    void setWindowSize(const cv::Size& sz);
    cv::Size getWindowSize();
    cv::Size getInitialSize();
    void setVideoFrameSize(const cv::Size& sz);
    cv::Size getVideoFrameSize();
    cv::Size getFrameBufferSize();
    cv::Size getNativeFrameBufferSize();
    float getXPixelRatio();
    float getYPixelRatio();
    bool isFullscreen();
    void setFullscreen(bool f);
    bool isResizable();
    void setResizable(bool r);
    bool isVisible();
    void setVisible(bool v);
    bool isOffscreen();
    void setOffscreen(bool o);
    void setStretching(bool s);
    bool isStretching();
    bool isClosed();
    bool isAccelerated();
    void setAccelerated(bool u);
    void close();
    bool display();

    Viz2DWindow* addWindow(int x, int y, const string& title);
    nanogui::Label* addGroup(const string& label);
    nanogui::Button* addButton(const string& caption, std::function<void()> fn);
    template<typename Tval> nanogui::detail::FormWidget<Tval>* addFormWidget(const string &name, const string &label, const Tval& value, const Tval &min, const Tval &max, bool spinnable, const string &unit, const string tooltip, bool visible = true, bool enabled = true) {
        auto& prop = storage().properties().create(name, label, value, min, max);
        auto* var = addVariable(name, std::any_cast<Tval&>(prop.value_), min, max, spinnable, unit, tooltip, visible, enabled);
        prop.fn_ = [=](const std::any& val) {
            var->set_value(std::any_cast<Tval>(val));
        };
        return var;
    }

    template<typename Tenum> nanogui::detail::FormWidget<Tenum>* addFormWidget(const string &name, const string &label, const Tenum& e, const Tenum& emin, const Tenum& emax, const std::vector<string> &items) {
        auto& prop = storage().properties().create(name, label, (int)e, (int)emin, (int)emax);
        int& v = std::any_cast<int&>(prop.value_);
        Tenum& en = *reinterpret_cast<Tenum*>(&v);
        auto *widget = addComboBox(label, en, items);
        prop.fn_ = [=](const std::any& val) {
            widget->set_value((Tenum)std::any_cast<int>(val));
        };
        return widget;
    }

    nanogui::detail::FormWidget<nanogui::Color>* addFormWidget(const string &name, const string &label, const nanogui::Color& color, const string &tooltip = "", bool visible = true, bool enabled = true) {
        auto& prop = storage().properties().create(name, label, color);
        auto *widget = addColorPicker(label, std::any_cast<nanogui::Color&>(prop.value_), tooltip, visible, enabled);
        prop.fn_ = [=](const std::any& color) {
            widget->set_value(std::any_cast<nanogui::Color>(color));
        };
        return widget;
    }

    nanogui::detail::FormWidget<bool>* addFormWidget(const string &name, const string &label, const bool& b, const string &tooltip = "", bool visible = true, bool enabled = true) {
        auto& prop = storage().properties().create(name, label, b);
        auto *widget = addCheckbox(label, std::any_cast<bool&>(prop.value_), tooltip, visible, enabled);
        prop.fn_ = [=](const std::any& b) {
            widget->set_value(std::any_cast<bool>(b));
        };
        return widget;
    }

//    template<typename T> Property& addProperty(const string& name, const string& label, const T& val, const T& min, const T& max, std::function<void(const std::any&)> fn) {
//        auto& prop = storage_.properties().create<T>(name, label, val, min, max);
//        assert(fn);
//        prop.fn_ = fn;
//        return prop;
//    }

    uint64_t getFrameCount() {
        return frameCnt_;
    }

    Properties& properties() {
        return storage().properties();
    }

    template <typename T>
    void propagate(const string& name, const T& value, double scale = 1) {
        storage().properties().propagate<T>(name, value, scale);
    }

    template<typename T, typename std::enable_if<std::is_enum<T>::value >::type* = nullptr> T& property(const string& name) {
        int& propVal = std::any_cast<int&>(storage().properties().property(name).value_);
        return *reinterpret_cast<T*>(&propVal);
    }

    template<typename T, typename std::enable_if<!std::is_enum<T>::value >::type* = nullptr> T& property(const string& name) {
        return std::any_cast<T&>(storage().properties().property(name).value_);
    }
private:
    nanogui::detail::FormWidget<bool>* addVariable(const string &name, bool &v, const string &tooltip = "", bool visible = true, bool enabled = true);
    template<typename T> nanogui::detail::FormWidget<T>* addVariable(const string &name, T &v, const T &min, const T &max, bool spinnable, const string &unit, const string tooltip, bool visible = true, bool enabled = true) {
        auto var = form()->add_variable(name, v);
        var->set_enabled(enabled);
        var->set_visible(visible);
        var->set_spinnable(spinnable);
        var->set_min_value(min);
        var->set_max_value(max);
        if (!unit.empty())
            var->set_units(unit);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    nanogui::detail::FormWidget<nanogui::Color>* addColorPicker(const string& label, nanogui::Color& color, const string& tooltip = "", bool visible = true, bool enabled = true);
    template<typename T> nanogui::detail::FormWidget<T>* addComboBox(const string &label, T& e, const std::vector<string>& items, const string& tooltip = "", bool visible = true, bool enabled = true) {
        auto* var = form()->add_variable(label, e, true);
        var->set_enabled(enabled);
        var->set_visible(visible);
        var->set_items(items);
        return var;
    }

    nanogui::detail::FormWidget<bool>* addCheckbox(const string &label, bool& state, const string& tooltip = "", bool visible = true, bool enabled = true){
        auto var = form()->add_variable(label, state);
        var->set_enabled(enabled);
        var->set_visible(visible);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    Storage& storage() {
        return storage_;
    }

    nanogui::Screen& screen();
    virtual bool keyboard_event(int key, int scancode, int action, int modifiers);
    void setMousePosition(int x, int y);
    nanogui::FormHelper* form();
    CLGLContext& clgl();
    CLVAContext& clva();
    NanoVGContext& nvg();
    GLFWwindow* getGLFWWindow();
    NVGcontext* getNVGcontext();
};
}
} /* namespace kb */

#endif /* SRC_COMMON_VIZ2DWORKER_HPP_ */
