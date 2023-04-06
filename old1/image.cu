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

__device__ float getTheta(float x, float y)
{
	float rtn = 0;
	if(y<0){
		rtn =  atan2(y,x)*-1;
  }
	else {
		rtn = M_PI +(M_PI-atan2(y,x));
  }
	return rtn;
}

__device__ int polarCoordX(float x, float y, float z, int inputWidth, int inputHeight)
{
    // converting to spherical coordinates
    float rho = sqrt(x*x+y*y+z*z);
    float normTheta = getTheta(x,y)/(2*M_PI);		// /(2*math.pi) normalise theta
    float normPhi = (M_PI-acos(z/rho))/M_PI;	// /math.pi normalise phi
    
    // convert our spherical coordinates back to u,v use this for coordinates 
    int iX = normTheta*inputWidth;
    int iY = normPhi*inputHeight;

    // catch possible overflow
    if(iX>=inputWidth){
      iX=iX-(inputWidth);
    }
    if(iY>=inputHeight){
      iY=iY-(inputHeight);
    }
    return iX;
}

__device__ int polarCoordY(float x, float y, float z, int inputWidth, int inputHeight)
{
    // converting to spherical coordinates
    float rho = sqrt(x*x+y*y+z*z);
    float normTheta = getTheta(x,y)/(2*M_PI);		// /(2*math.pi) normalise theta
    float normPhi = (M_PI-acos(z/rho))/M_PI;	// /math.pi normalise phi
    
    // convert our spherical coordinates back to u,v use this for coordinates 
    int iX = normTheta*inputWidth;
    int iY = normPhi*inputHeight;

    // catch possible overflow
    if(iX>=inputWidth){
      iX=iX-(inputWidth);
    }
    if(iY>=inputHeight){
      iY=iY-(inputHeight);
    }
    return iY;
}


__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int inputWidth, int inputHeight)
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int len_cube = cols/4;
  //const int width_res = len_cube*3;
  //const int height_res = len_cube*2;
  const int sqr = len_cube;
  int tx;
  int ty;
  float x;
  float y;
  float z;
  int xx;
  int yy;

  //#	+----+----+----+
  //#	| Y+ | X+ | Y- |
  //#	+----+----+----+
  //#	| X- | Z- | Z+ |
  //#	+----+----+----+

  // Y+ Top-Left 
  if (dst_x < cols/3 && dst_y < rows/2)
    {
      // find_local_coordinates
      tx = dst_x;
      ty = dst_y;

      // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
      x = tx-0.5*sqr;
      y = 0.5*sqr;
      z = ty-0.5*sqr;

      //Converting to spherical coordinates
      xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
      yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

      // final step
      uchar3 val = src(yy, xx); 
      dst(dst_y, dst_x).x = val.x;
      dst(dst_y, dst_x).y = val.y;
      dst(dst_y, dst_x).z = val.z;
    }
  
  // X+ Top-Middle
  if (dst_x >= cols/3 && dst_x < (2*cols)/3 && dst_y < rows/2)
  {
      // find_local_coordinates
      tx = dst_x-sqr;
      ty = dst_y;

      // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
			x = 0.5*sqr;
			y = (tx-0.5*sqr)*-1;
			z = ty-0.5*sqr;

      //Converting to spherical coordinates
      xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
      yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

      // final step
      uchar3 val = src(yy, xx); 
      dst(dst_y, dst_x).x = val.x;
      dst(dst_y, dst_x).y = val.y;
      dst(dst_y, dst_x).z = val.z;
  }

  // Y- Top-Right
  if (dst_x >= (2*cols)/3 && dst_x < cols && dst_y < rows/2)
  {
    // find_local_coordinates
    tx = dst_x-sqr*2;
		ty = dst_y;

    // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
    x = (tx-0.5*sqr)*-1;
    y = -0.5*sqr;
    z = ty-0.5*sqr;

    //Converting to spherical coordinates
    xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
    yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

    // final step
    uchar3 val = src(yy, xx); 
    dst(dst_y, dst_x).x = val.x;
    dst(dst_y, dst_x).y = val.y;
    dst(dst_y, dst_x).z = val.z;
  }

    // X-
  if (dst_x < cols/3 && dst_y >= rows/2 && dst_y < rows)
    {
      // find_local_coordinates
      tx = dst_x;
      ty = dst_y - sqr;

      // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
      x = int(-0.5*sqr);
      y = int(tx-0.5*sqr);
      z = int(ty-0.5*sqr);

      //Converting to spherical coordinates
      xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
      yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

      // final step
      uchar3 val = src(yy, xx); 
      dst(dst_y, dst_x).x = val.x;
      dst(dst_y, dst_x).y = val.y;
      dst(dst_y, dst_x).z = val.z;
    }
  
  // Z-
  if (dst_x >= cols/3 && dst_x < (2*cols)/3 && dst_y >= rows/2 && dst_y < rows)
  {
      // find_local_coordinates
      tx = dst_x-sqr;
      ty = dst_y - sqr;

      // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
      x = (ty-0.5*sqr)*-1;
      y = (tx-0.5*sqr)*-1;
      z = 0.5*sqr	;

      //Converting to spherical coordinates
      xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
      yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

      // final step
      uchar3 val = src(yy, xx); 
      dst(dst_y, dst_x).x = val.x;
      dst(dst_y, dst_x).y = val.y;
      dst(dst_y, dst_x).z = val.z;
  }

  // Z+
  if (dst_x >= (2*cols)/3 && dst_x < cols && dst_y >= rows/2 && dst_y < rows)
  {
      // find_local_coordinates
      tx = dst_x-sqr*2;
      ty = dst_y - sqr;

      // normalize the coordinates and move the origin to the center of the cube 3D cartesian coordinate 
      x = ty-0.5*sqr;
      y = (tx-0.5*sqr)*-1;
      z =- 0.5*sqr	;

      //Converting to spherical coordinates
      xx = polarCoordX( x,  y,  z, inputWidth, inputHeight);
      yy = polarCoordY( x,  y,  z, inputWidth, inputHeight);

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

