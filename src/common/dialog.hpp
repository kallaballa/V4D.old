#ifndef SRC_COMMON_DIALOG_HPP_
#define SRC_COMMON_DIALOG_HPP_

#include <nanogui/nanogui.h>

#include <set>
#include <string>

namespace kb {
namespace viz2d {

using std::string;

class Dialog : public nanogui::Window {
private:
    static std::function<bool(Dialog*, Dialog*)> viz2DWin_Xcomparator;
    static std::set<Dialog*, decltype(viz2DWin_Xcomparator)> all_windows_xsorted_;
    nanogui::Screen* screen_;
    nanogui::Vector2i lastDragPos_;
    nanogui::Vector2i maximizedPos_;
    nanogui::Button* minBtn_;
    nanogui::Button* maxBtn_;
    nanogui::ref<nanogui::AdvancedGridLayout> oldLayout_;
    nanogui::ref<nanogui::AdvancedGridLayout> newLayout_;
    bool minimized_ = false;
public:
    Dialog(nanogui::Screen* screen, int x, int y, const string& title);
    virtual ~Dialog();
    bool isMinimized();
    bool mouse_drag_event(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int mods) override;
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_DIALOG_HPP_ */
