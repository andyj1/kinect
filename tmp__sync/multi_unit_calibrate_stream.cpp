// 01.16.2020

#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4a/k4atypes.h>

#include <k4arecord/record.h>
#include <k4arecord/playback.h>

#include <k4abt.h>
#include <k4abt.hpp>
#include <k4abttypes.h>

#include "MultiDeviceCapturer.h"

// Handle to a k4a body tracking frame.
#define TIMEOUT_IN_MS 1000
#define FRAME_COUNT 1000
#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    };

using namespace std;

static vector<double> values;

// mapping the confidence value to string
string confidenceEnumMapping(k4abt_joint_confidence_level_t confidence_level)
{
	string resultString;
	switch (confidence_level)
	{
	case K4ABT_JOINT_CONFIDENCE_NONE: // 0
		resultString += "The joint is out of range(too far from depth camera)";
		break;
	case K4ABT_JOINT_CONFIDENCE_LOW: // 1
		resultString += "The joint is not observed(likely due to occlusion) - predicted joint pose";
		break;
	case K4ABT_JOINT_CONFIDENCE_MEDIUM:
		resultString += "Medium confidence in joint pose";
		break;
		// beyond this will be supported later
	case K4ABT_JOINT_CONFIDENCE_HIGH:
		resultString += "High confidence in joint pose";
		break;
	default:
		break;
	}
	return resultString;
}

// store position and orientation information
vector<double> print_body_information(k4abt_body_t body)
{
	for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
	{
		k4a_float3_t position = body.skeleton.joints[i].position;
		k4a_quaternion_t orientation = body.skeleton.joints[i].orientation;
		k4abt_joint_confidence_level_t confidence_level = body.skeleton.joints[i].confidence_level;

		values.push_back(i); // joint
		values.push_back(position.v[0]);
		values.push_back(position.v[1]);
		values.push_back(position.v[2]);
		values.push_back(orientation.v[0]);
		values.push_back(orientation.v[1]);
		values.push_back(orientation.v[2]);
		values.push_back(orientation.v[3]);
		values.push_back(confidence_level);
	}
	return values;
}

#undef main
int main(int argc, char **argv)
{
    int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
    int32_t powerline_freq = 2;          // default to a 60 Hz powerline
    uint16_t depth_threshold = 1000;     // default to 1 meter

    vector<uint32_t> device_indices{ 0 }; // Set up a MultiDeviceCapturer to handle getting many synchronous captures
                                          // Note that the order of indices in device_indices is not necessarily
                                          // preserved because MultiDeviceCapturer tries to find the master device based
                                          // on which one has sync out plugged in. Start with just { 0 }, and add
                                          // another if needed
	
	// take in the number of devices from argument
	size_t num_devices = 0;
	num_devices = static_cast<size_t>(atoi(argv[1]));
	if (num_devices > k4a::device::get_installed_count())
	{
		cerr << "Not enough cameras plugged in!\n";
		exit(1);
	}
	if (num_devices != 2 && num_devices != 1)
    {
        cerr << "Invalid choice for number of devices!\n";
        exit(1);
    }
    else if (num_devices == 2)
    {
        device_indices.emplace_back(1); // now device indices are { 0, 1 }
    }

	try
	{
		// open devices
		k4a_device_t device1, device2;
		if (k4a_device_open(0, &device1) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to open device 1\n");
			exit(EXIT_FAILURE);
		}
		if (k4a_device_open(1, &device2) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to open device 2\n");
			exit(EXIT_FAILURE);
		}
		
		// check the master / subordinate states of the devices
		// ** sync_in: subordinate / sync_out: master
		bool sync_in_jack_connected_0, sync_out_jack_connected_0;		
		if (k4a_device_get_sync_jack(device1, &sync_in_jack_connected_0, &sync_out_jack_connected_0) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to sync device 1 properly\n");
			exit(EXIT_FAILURE);
		}
		printf("device 1: sync in %i, sync out %i --> (1,0) subordinate\n\n", sync_in_jack_connected_0, sync_out_jack_connected_0);

		bool sync_in_jack_connected_1, sync_out_jack_connected_1;
		if (k4a_device_get_sync_jack(device2, &sync_in_jack_connected_1, &sync_out_jack_connected_1) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to sync device 2 properly\n");
			exit(EXIT_FAILURE);
		}
		printf("device 2: sync in %i, sync out %i --> (0,1) master\n\n", sync_in_jack_connected_1, sync_out_jack_connected_1);

		// sync_out: master / sync_in: subordinate
		// Kinect 1: subordinate
		k4a_device_configuration_t device_config_subordinate = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		device_config_subordinate.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
		device_config_subordinate.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
		device_config_subordinate.camera_fps = K4A_FRAMES_PER_SECOND_30;
		device_config_subordinate.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
		device_config_subordinate.color_resolution = K4A_COLOR_RESOLUTION_2160P;
		device_config_subordinate.synchronized_images_only = true;

		// Kinect 2: master
		k4a_device_configuration_t device_config_master = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		device_config_master.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
		device_config_master.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
		device_config_master.camera_fps = K4A_FRAMES_PER_SECOND_30;
		device_config_master.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
		device_config_master.color_resolution = K4A_COLOR_RESOLUTION_2160P;
		device_config_master.synchronized_images_only = true;

		// change to this sync mode for single camera
		// device_config_standalone.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;

		// start cameras based on the configurations - master / subordinate
		if (k4a_device_start_cameras(device1, &device_config_subordinate) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to start device 1\n");
			exit(EXIT_FAILURE);
		};

		if (k4a_device_start_cameras(device2, &device_config_master) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to start device 2\n");
			exit(EXIT_FAILURE);
		};

		// start IMU (inertial measurement unit)
		if (k4a_device_start_imu(device1) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to start subordinate IMU\n");
			exit(EXIT_FAILURE);
		}

		if (k4a_device_start_imu(device2) != K4A_RESULT_SUCCEEDED) {
			fprintf(stderr, "Error: Failed to start master IMU\n");
			exit(EXIT_FAILURE);
		}

		// Capture a imu sample to check
		k4a_imu_sample_t imu_sample;
		switch (k4a_device_get_imu_sample(device1, &imu_sample, TIMEOUT_IN_MS))
		{
			case K4A_WAIT_RESULT_FAILED:
				printf("Failed to read a imu sample\n");
				break;
			case K4A_WAIT_RESULT_SUCCEEDED:
				// Access the accelerometer readings
				if (sizeof(imu_sample) > 0)
				{
					printf("Accelerometer temperature:%.2f x:%.4f y:%.4f z: %.4f\n",
							imu_sample.temperature,
							imu_sample.acc_sample.xyz.x,
							imu_sample.acc_sample.xyz.y,
							imu_sample.acc_sample.xyz.z);
				}
				break;
			case K4A_WAIT_RESULT_TIMEOUT:
				printf("Timed out waiting for a imu sample\n");
				break;
		}

		cout << "Depth threshold: : " << depth_threshold << ". Color exposure time: " << color_exposure_usec
        	 << ". Powerline frequency mode: " << powerline_freq << endl;
		
		// set up devices
		for (uint32_t i : device_indices)
		{
			
		}











	} catch (exception e) {
		fprintf(stderr, "Exception: %s", e.what());
	}
	// 	k4a::capture cap;
	// 	if (device2.get_capture(&cap) == true)
	// 	{
	// 		printf("succeeded opening second sensor!\n");	
	// 	} else {
	// 		printf("failed second sensor!\n");
	// 	};

	// 	k4a::calibration sensor_calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution);
	// 	k4abt::tracker tracker = k4abt::tracker::create(sensor_calibration);

	// 	int frame_count = 0;
	// 	do
	// 	{
	// 		k4a::capture sensor_capture;
	// 		if (device.get_capture(&sensor_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
	// 		{
	// 			frame_count++;

	// 			std::cout << "Start processing frame " << frame_count << std::endl;

	// 			if (!tracker.enqueue_capture(sensor_capture))
	// 			{
	// 				// It should never hit timeout when K4A_WAIT_INFINITE is set.
	// 				std::cout << "Error! Add capture to tracker process queue timeout!" << std::endl;
	// 				break;
	// 			}

	// 			k4abt::frame body_frame = tracker.pop_result();
	// 			if (body_frame != nullptr)
	// 			{
	// 				size_t num_bodies = body_frame.get_num_bodies();
	// 				std::cout << num_bodies << " bodies detected!" << std::endl;

	// 				for (size_t i = 0; i < num_bodies; i++)
	// 				{
	// 					k4abt_body_t body = body_frame.get_body(i);

	// 					// print information
	// 					for (int i = 0; i < (int) K4ABT_JOINT_COUNT; i++)
	// 						{
	// 							k4a_float3_t position = body.skeleton.joints[i].position;
	// 							k4a_quaternion_t orientation = body.skeleton.joints[i].orientation;
	// 							k4abt_joint_confidence_level_t confidence_level = body.skeleton.joints[i].confidence_level;
								
	// 							//outfile << *(it) << "," << *(it + 1) << "," << *(it + 2) << "," << *(it + 3) << "," << *(it + 4) << "," << *(it + 5) << "," << *(it + 6) << "," << *(it + 7) << "," << confidenceEnumMapping (int(*(it + 8))) << "," << endl;
	// 							if (outfile.is_open())
	// 							{
	// 								outfile << body.id << "," << i << "," << position.v[0] << "," << position.v[1] << "," << position.v[2] << "," << orientation.v[0] << "," << orientation.v[1] << "," << orientation.v[2] << "," << orientation.v[3] << "," << confidenceEnumMapping(confidence_level) << "," << endl;
	// 							}

	// 						}
	// 				}
	// 			}
	// 			else
	// 			{
	// 				//  It should never hit timeout when K4A_WAIT_INFINITE is set.
	// 				std::cout << "Error! Pop body frame result time out!" << std::endl;
	// 				break;
	// 			}
	// 		}
	// 		else
	// 		{
	// 			// It should never hit time out when K4A_WAIT_INFINITE is set.
	// 			std::cout << "Error! Get depth frame time out!" << std::endl;
	// 			break;
	// 		}
	// 	} while (frame_count < FRAME_COUNT);
	// 	std::cout << "Finished body tracking processing!" << std::endl;

	// }
	// catch (const std::exception & e)
	// {
	// 	std::cerr << "Failed with exception:" << std::endl
	// 		<< "    " << e.what() << std::endl;
	// 	return 1;
	// }

	return 0;
}
