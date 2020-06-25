// 06.08.2020
// Get statistics of all connected Kinect devices
#include <iostream>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

#undef main
int main()
{
	try
	{
        for (int devnum = 0; devnum < k4a::device::get_installed_count(); devnum++)
        {
            std::cerr << "Device number: " << devnum << std::endl;

            // open device in order
            k4a::device device = k4a::device::open(devnum);

            // get serial number
            std::string serialnumber = device.get_serialnum();
            std::cerr << "Serial number: "  << serialnumber << std::endl;

            // set configuration for the device
            k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
            device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
            
            // start the device
            device.start_cameras(&device_config);
            k4a::calibration sensor_calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution);

            // (pre-calibration) intrinsic matrix
            const k4a_calibration_intrinsic_parameters_t::_param &i = sensor_calibration.color_camera_calibration.intrinsics.parameters.param;
            cv::Matx33f intrinsic_matrix = cv::Matx33f::eye();
            intrinsic_matrix(0, 0) = i.fx;
            intrinsic_matrix(1, 1) = i.fy;
            intrinsic_matrix(0, 2) = i.cx;
            intrinsic_matrix(1, 2) = i.cy;
            std::cerr << "Intrinsic Matrix: " << std::endl << intrinsic_matrix << std::endl;

            // (pre-calibration) extrinsic matrix
            const k4a_calibration_extrinsics_t &ex = sensor_calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];

            cv::Matx33d R = cv::Matx33d::eye();
            cv::Vec3d t = (0., 0., 0.);
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    R(i, j) = ex.rotation[i * 3 + j];
                }
            }
            t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
            std::cerr << "Extrinsic Matrix:" << std::endl << "(rotation): " << std::endl << R << std::endl << "(translation): " << std::endl << t << std::endl;
            std::cerr << std::endl;
        }

	}
	catch (const std::exception & e)
	{
		std::cerr << "Failed with exception:" << std::endl
			<< "    " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
