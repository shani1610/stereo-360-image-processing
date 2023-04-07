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

__device__ int convertEquirectUVtoUnit2D(float theta, float phi, int len_cube)
{      
      // calculate the unit vector
      float x = cos(theta)*sin(phi);
      float y = sin(theta)*sin(phi);
      float z = cos(phi);
      
      // find the maximum value in the unit vector
      float maximum = max(max(abs(x),abs(y)),abs(z));
      float xx = x/maximum;
      float yy = y/maximum;
      float zz = z/maximum;
      
      // project ray to cube surface
      // if(xx==1 or xx==-1):
      //   (x,y,z, faceIndex) = projectX(theta,phi,xx)
      // elif(yy==1 or yy==-1):
      //   (x,y,z, faceIndex) = projectY(theta,phi,yy)
      // else:
      //   (x,y,z, faceIndex) = projectZ(theta,phi,zz)
      
      // (x,y) = unit3DToUnit2D(x,y,z,faceIndex)
      
      x*=len_cube;
      y*=len_cube;
        
      x = int(x);
      y = int(y);

      return x;
      //return {"index":faceIndex,"x":x,"y":y}   
}


__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int inputWidth, int inputHeight)
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  int U;
  int V;
  int len_cube = cols/3;
  
  //#	+----+----+----+
  //#	| Y+ | X+ | Y- |
  //#	+----+----+----+
  //#	| X- | Z- | Z+ |
  //#	+----+----+----+

  if (dst_x < cols && dst_y < rows)
    {

      // get the normalised u,v coordinates for the current pixel
      U = float(dst_x)/(cols-1);		// 0..1
      V = float(dst_y)/(rows-1);		// no need for 1-... as the image output needs to start from the top anyway.		
      
      // taking the normalised cartesian coordinates calculate the polar coordinate for the current pixel
      float theta = U*2*M_PI;
      float phi = V*M_PI;
      
      // calculate the 3D cartesian coordinate which has been projected to a cubes face

    //   cart = convertEquirectUVtoUnit2D(theta, phi, len_cube)
		
		// # 5. use this pixel to extract the colour
		
		// output.append(getColour(cart["x"],cart["y"],cart["index"]))
      int yy = 1;
      int xx = 1; 

      // final step
      uchar3 val = src(yy, xx); 
      dst(dst_y, dst_x).x = val.x;
      dst(dst_y, dst_x).y = val.y;
      dst(dst_y, dst_x).z = val.z;
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
  int inputWidth = src.cols;
  int inputHeight = src.rows;
  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, inputWidth, inputHeight);

}

