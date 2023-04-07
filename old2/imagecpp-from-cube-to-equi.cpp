#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;
using namespace cv;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst );

int main( int argc, char** argv )
{
  cv::namedWindow("Equirectangular Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Cube Map Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  // let's downscale the image using new  width and height
  int down_width = h_img.cols/1;
  int down_height = h_img.rows/1;
  Mat resized_down;
  //resize down
  resize(h_img, resized_down, Size(down_width, down_height), INTER_LINEAR);
  h_img = resized_down;
  // the result
  int len_cube = h_img.cols/4;
  int out_width = len_cube * 3;
  int out_height = len_cube * 2;
  cv::Mat h_result(out_height, out_width, CV_8UC3);

  cv::cuda::GpuMat d_img, d_result;
  d_img.upload(h_img);
  d_result.upload(h_result);
  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10;
  
  for (int i=0;i<iter;i++)
    {
      startCUDA ( d_img, d_result );
    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  d_result.download(h_result);
  
  cv::imshow("Cube Map Image", h_result);
  cv::imwrite("image-cube.jpg", h_result);  
  cv::waitKey(0);//wait till user press any key
  cv::destroyWindow("MyWindow");//close the windsow and release allocate memory//
  cout << "Image is saved successfully…..";

  cout << "Time: "<< diff.count() << endl;
  cout << "Time/frame: " << diff.count()/iter << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();
  
  return 0;
}
