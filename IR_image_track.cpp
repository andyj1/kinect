// 11.20.2019

#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <k4a/k4a.hpp>
#include <k4abt.hpp>

#include <k4a/k4a.h>
#include <k4arecord/record.h>
#include <k4arecord/playback.h>

// Handle to a k4a body tracking frame.
#include <k4abttypes.h>


#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define FRAME_COUNT 10
#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    };

using namespace std;
using namespace cv;

static cv::Mat color_to_opencv(const k4a::image &im)
{
    cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void *)im.get_buffer());
    cv::Mat cv_image_no_alpha;
    cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
    return cv_image_no_alpha;
}

static cv::Mat depth_to_opencv(const k4a::image &im)
{
    return cv::Mat(im.get_height_pixels(),
                   im.get_width_pixels(),
                   CV_16U,
                   (void *)im.get_buffer(),
                   static_cast<size_t>(im.get_stride_bytes()));
}

#undef main
int main()
{
	try
	{
		std::chrono::milliseconds TIMEOUT_IN_MS = std::chrono::milliseconds(K4A_WAIT_INFINITE); // 1349400, K4A_WAIT_INFINITE
		k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		device_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
		device_config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
		device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
		device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

		k4a::device device = k4a::device::open(0);
		device.start_cameras(&device_config);
		k4a::calibration sensor_calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution);

		k4a::image ir_image;
		k4a::image color_image;
		int frame_count = 0;
		do
		{
			k4a::capture sensor_capture;
			if (device.get_capture(&sensor_capture, TIMEOUT_IN_MS))
			{
				frame_count++;
				// color_image = sensor_capture.get_color_image();
				// cv::Mat color_to_opencv_image = color_to_opencv(color_image);
				// cout << color_to_opencv_image << endl;
				// cv::imshow("actual color", color_to_opencv_image);
        		// cv::waitKey(1);
				// cout << "read color! " << endl;
				
				ir_image = sensor_capture.get_ir_image();
				cv::Mat ir_depth_to_opencv_image = depth_to_opencv(ir_image);
				cv::imshow("ir depth", ir_depth_to_opencv_image);
        		cv::waitKey(0);

				Mat thr, gray, src;
				src = ir_depth_to_opencv_image;
				std::cout << src.channels() << std::endl; // 1 --> binary already
				thr = ir_depth_to_opencv_image;

				// convert image to grayscale
				// cv::cvtColor( ir_depth_image, gray, COLOR_BGR2GRAY );
				// cout << "converted to grayscale" << endl;

				// convert grayscale to binary image
				// cv::threshold( gray, thr, 100,255,THRESH_BINARY );
				// cout << "converted to binary" << endl;

				// find moments of the image
				Moments m = cv::moments(thr,true);
				Point p(m.m10/m.m00, m.m01/m.m00);
				
				// coordinates of centroid
				cout<< cv::Mat(p)<< endl;
				
				// show the image with a point mark at the centroid
				cv::circle(src, p, 5, Scalar(128,0,0), -1);
				cv::imshow("Image with center",src);
				cv::waitKey(0);


				std::cout << "Start processing frame " << frame_count << std::endl;
			}
			else
			{
				// It should never hit time out when K4A_WAIT_INFINITE is set.
				std::cout << "Error! Get depth frame time out!" << std::endl;
				break;
			}
		} while (frame_count < FRAME_COUNT);
		std::cout << "Finished body tracking processing!" << std::endl;
		device.close();
	}
	catch (const std::exception & e)
	{
		std::cerr << "Failed with exception:" << std::endl
			<< "    " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
