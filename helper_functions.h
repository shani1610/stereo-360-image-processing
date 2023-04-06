#define HELPER_MATH_H

#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif
#include <math.h>

// inline uchar3 invert(uchar3 src, uchar3 dst, int dst_y, int dst_x)
// {
//     uchar3 val = src(dst_y, dst_x);
//     dst(dst_y, dst_x).x = 255-val.x;
//     dst(dst_y, dst_x).y = 255-val.y;
//     dst(dst_y, dst_x).z = 255-val.z;
//     return dst;
// }

