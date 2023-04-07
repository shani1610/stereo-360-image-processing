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

  //#	+------+------+------+
  //#	| Y+,1 | X+,2 | Y-,3 |
  //#	+------+------+------+
  //#	| X-,4 | Z-,5 | Z+,6 |
  //#	+------+------+------+

struct ProjectionRes{
    float x;
    float y;
    float z;
    int faceIndex;
};

struct Pixels2D{
    float x2D;
    float y2D;
};

struct XYIndex{
    int x;
    int y;
    int faceIndex;
};

__device__ Pixels2D unit3DToUnit2D(ProjectionRes MyProjectionRes){
  float x = MyProjectionRes.x;
  float y = MyProjectionRes.y;
  float z = MyProjectionRes.z;
  int faceIndex = MyProjectionRes.faceIndex;
  float x2D;
  float y2D;
	if(faceIndex==2){ //"X+"
		x2D = y+0.5;
		y2D = z+0.5;
  }
	else if(faceIndex==1){ //"Y+"
		x2D = (x*-1)+0.5;
		y2D = z+0.5;
	}
  else if(faceIndex==4){ //"X-"
		x2D = (y*-1)+0.5;
		y2D = z+0.5;
	}
  else if(faceIndex==3){ //"Y-"
		x2D = x+0.5;
		y2D = z+0.5;
	}
  else if(faceIndex==6){ //"Z+"
		x2D = y+0.5;
		y2D = (x*-1)+0.5;
	}
  else{ // 5, Z-
		x2D = y+0.5;
		y2D = x+0.5;
  }
		
	// need to do this as image.getPixel takes pixels from the top left corner.
	y2D = 1-y2D;

  // return 
  Pixels2D My2DPixels;
  My2DPixels.x2D = x2D;
  My2DPixels.y2D = y2D;
	return My2DPixels;
};

__device__ ProjectionRes projectX(float theta, float phi, int sign){
  float x = sign*0.5;
  int faceIndex;
  if (sign==1){
    faceIndex = 2; // X+
  }
  else{
    faceIndex = 4; // X-
  }
  float rho = float(x)/(cos(theta)*sin(phi));
  float y = rho*sin(theta)*sin(phi);
  float z = rho*cos(phi);
  // return 
  ProjectionRes MyProjectionRes;
  MyProjectionRes.x = x;
  MyProjectionRes.y = y;
  MyProjectionRes.z = z;
  MyProjectionRes.faceIndex = faceIndex;
  return MyProjectionRes;
};

__device__ ProjectionRes projectY(float theta, float phi, int sign){
	float y = sign*0.5;
  int faceIndex;
  if (sign==1){
    faceIndex = 1; // Y+
  }
  else{
    faceIndex = 3; // Y-
  }
	float rho = float(y)/(sin(theta)*sin(phi));
	float x = rho*cos(theta)*sin(phi);
	float z = rho*cos(phi);
  // return 
  ProjectionRes MyProjectionRes;
  MyProjectionRes.x = x;
  MyProjectionRes.y = y;
  MyProjectionRes.z = z;
  MyProjectionRes.faceIndex = faceIndex;
  return MyProjectionRes;
}
	
__device__ ProjectionRes projectZ(float theta, float phi, int sign){
	float z = sign*0.5;
  int faceIndex;
  if (sign==1){
    faceIndex = 6; // Z+
  }
  else{
    faceIndex = 5; // Z-
  }
	float rho = float(z)/cos(phi);
	float x = rho*cos(theta)*sin(phi);
	float y = rho*sin(theta)*sin(phi);
  // return 
  ProjectionRes MyProjectionRes;
  MyProjectionRes.x = x;
  MyProjectionRes.y = y;
  MyProjectionRes.z = z;
  MyProjectionRes.faceIndex = faceIndex;
  return MyProjectionRes;
}

__device__ XYIndex convertEquirectUVtoUnit2D(float theta, float phi, int len_cube)
{      
      // calculate the unit vector
      float x = cos(theta)*sin(phi);
      float y = sin(theta)*sin(phi);
      float z = cos(phi);
      
      // find the maximum value in the unit vector
      float maximum = max(max(abs(x),abs(y)),abs(z));
      int xx = x/maximum;
      int yy = y/maximum;
      int zz = z/maximum;

      //project ray to cube surface
      ProjectionRes MyProjectionRes;
      int faceIndex = 0;
      if(xx==1 or xx==-1){
        MyProjectionRes = projectX(theta,phi,xx);
      }
      else if(yy==1 or yy==-1){
        MyProjectionRes = projectY(theta,phi,yy);
      }
      else{
        MyProjectionRes = projectZ(theta,phi,zz);
      }

      Pixels2D My2DPixels;
      My2DPixels = unit3DToUnit2D(MyProjectionRes);

      x = My2DPixels.x2D;
      y = My2DPixels.y2D;
      
      x*=len_cube;
      y*=len_cube;
        
      x = int(x);
      y = int(y);

      XYIndex MyXYIndex;
      MyXYIndex.x = x;
      MyXYIndex.y = y;
      MyXYIndex.faceIndex = faceIndex;
      return MyXYIndex;   
}

__device__ XYIndex getGlobalCoords(XYIndex MyXYIndex, int len_cube){
  int x_local = MyXYIndex.x;
	int y_local = MyXYIndex.y;
  int index = MyXYIndex.faceIndex;
  int x;
	int y;

	if(index==2){ //X+
    x = x_local+len_cube;
    y = y_local;
  }
	else if(index==4){ //X-
		x = x_local;
    y = y_local+len_cube;
  }
	else if(index==1){ //Y+
    x = x_local;
    y = y_local;
	}
  else if(index==3){ //Y-
    x = x_local+2*len_cube;
    y = y_local;
	}
  else if(index==6){ //Z+
    x = x_local+2*len_cube;
    y = y_local+len_cube;
	}
  else if(index==5){ //Z-
    x = x_local+len_cube;
    y = y_local+len_cube;
  }
  
  MyXYIndex.x = x;
  MyXYIndex.y = y;

  return MyXYIndex;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int inputWidth, int inputHeight)
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  float U;
  float V;
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
      XYIndex MyXYIndex;
      MyXYIndex = convertEquirectUVtoUnit2D(theta, phi, len_cube);

		  // extract the global coordinates from the local and the index
		  MyXYIndex = getGlobalCoords(MyXYIndex, len_cube);

      int yy = MyXYIndex.y;
      int xx = MyXYIndex.x; 
      //printf("%d, ", xx);
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

