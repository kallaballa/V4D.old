#ifndef SRC_COMMON_UTIL_HPP_
#define SRC_COMMON_UTIL_HPP_

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace kb {
namespace viz2d {
class Viz2DWorker;
std::string get_gl_info();
std::string get_cl_info();
void print_system_info();
void update_fps(Viz2DWorker& viz2d);
}
}

#endif /* SRC_COMMON_UTIL_HPP_ */
