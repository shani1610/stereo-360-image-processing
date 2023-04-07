#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"
#include "helper_functions.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols )
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  const int width_cube = cols/3;
  const int height_cube = rows/2;

  // Y-
  if (dst_x < cols/3 && dst_y < rows/2)
    {
      uchar3 val = src(dst_y, dst_x);
      //printf("float: %f", dst_y);
      
      // find_local_coordinates
      int dst_y_l = rows/2 - dst_y; // for the second line it will be rows
      int dst_x_l = dst_x;

      // normalize the coordinates
      float dst_y_f = float(dst_y_l)/float(height_cube);
      float dst_x_f = float(dst_x_l)/float(width_cube);

      // move the origin to the center of the cube
      dst_y_f = dst_y_f-0.5;
      dst_x_f = dst_x_f-0.5;

      // 3D cartesian coordinate
      float x_cart = dst_x_f;
      float y_cart = 0.5;
      float z_cart = dst_y_f;
      //printf("float: %f", z_cart);

      //Converting to spherical coordinates
      float R = sqrt(x_cart*x_cart + y_cart*y_cart + z_cart*z_cart);
      float theta = atan2(y_cart, x_cart);
      float phi = acos(z_cart/R);

      // convert our spherical coordinates back to u,v
      float u = theta/2*M_PI;
      float v = phi/M_PI;
      //printf("float: %f", u);

      int yy = v * rows/2;
      int xx = u * cols/3;

      // final step
      dst(yy, xx).x = val.x;
      dst(yy, xx).y = val.y;
      dst(yy, xx).z = val.z;

    }
  
  // X+
  if (dst_x >= cols/3 && dst_x < (2*cols)/3 && dst_y < rows/2)
  {
    uchar3 val = src(dst_y, dst_x);
    dst(dst_y, dst_x).x = 255-val.x;
    dst(dst_y, dst_x).y = 255-val.y;
    dst(dst_y, dst_x).z = 255-val.z;
  }

  // Y+
  if (dst_x >= (2*cols)/3 && dst_x < cols && dst_y < rows/2)
  {
    uchar3 val = src(dst_y, dst_x);
    dst(dst_y, dst_x).x = 255-val.x;
    dst(dst_y, dst_x).y = 255-val.y;
    dst(dst_y, dst_x).z = 255-val.z;
  }

    // X-
  if (dst_x < cols/3 && dst_y >= rows/2 && dst_y < rows)
    {
      uchar3 val = src(dst_y, dst_x);
      dst(dst_y, dst_x).x = 255-val.x;
      dst(dst_y, dst_x).y = 255-val.y;
      dst(dst_y, dst_x).z = 255-val.z;
    }
  
  // Z-
  if (dst_x >= cols/3 && dst_x < (2*cols)/3 && dst_y >= rows/2 && dst_y < rows)
  {
    uchar3 val = src(dst_y, dst_x);
    dst(dst_y, dst_x).x = 255-val.x;
    dst(dst_y, dst_x).y = 255-val.y;
    dst(dst_y, dst_x).z = 255-val.z;
  }

  // Z+
  if (dst_x >= (2*cols)/3 && dst_x < cols && dst_y >= rows/2 && dst_y < rows)
  {
    uchar3 val = src(dst_y, dst_x);
    dst(dst_y, dst_x).x = 255-val.x;
    dst(dst_y, dst_x).y = 255-val.y;
    dst(dst_y, dst_x).z = 255-val.z;
  }

}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols);

}

