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

// Json
// #include <nlohmann/json.hpp>

#define FRAME_COUNT 1000
#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    };

using namespace std;

static vector<double> values;

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
int main()
{

	// output file stream
	ofstream outfile;
	outfile.open("./joints_output_single.csv", ios::out);
	outfile << ",,Position,,,Orientation,,,,Confidence Level" << endl;
	outfile << "Body ID," << "Joint #," << "x,y,z," << "x,y,z,w" << endl;

	try
	{
		k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

		k4a::device device = k4a::device::open(0);
		device.start_cameras(&device_config);
		k4a::calibration sensor_calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution);
		k4abt::tracker tracker = k4abt::tracker::create(sensor_calibration);

		int frame_count = 0;
		do
		{
			k4a::capture sensor_capture;
			if (device.get_capture(&sensor_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
			{
				frame_count++;

				std::cout << "Start processing frame " << frame_count << std::endl;

				if (!tracker.enqueue_capture(sensor_capture))
				{
					// It should never hit timeout when K4A_WAIT_INFINITE is set.
					std::cout << "Error! Add capture to tracker process queue timeout!" << std::endl;
					break;
				}

				k4abt::frame body_frame = tracker.pop_result();
				if (body_frame != nullptr)
				{
					size_t num_bodies = body_frame.get_num_bodies();
					std::cout << num_bodies << " bodies detected!" << std::endl;

					for (size_t i = 0; i < num_bodies; i++)
					{
						k4abt_body_t body = body_frame.get_body(i);

						// print information
						for (int i = 0; i < (int) K4ABT_JOINT_COUNT; i++)
							{
								k4a_float3_t position = body.skeleton.joints[i].position;
								k4a_quaternion_t orientation = body.skeleton.joints[i].orientation;
								k4abt_joint_confidence_level_t confidence_level = body.skeleton.joints[i].confidence_level;
								
								//outfile << *(it) << "," << *(it + 1) << "," << *(it + 2) << "," << *(it + 3) << "," << *(it + 4) << "," << *(it + 5) << "," << *(it + 6) << "," << *(it + 7) << "," << confidenceEnumMapping (int(*(it + 8))) << "," << endl;
								if (outfile.is_open())
								{
									outfile << body.id << "," << i << "," << position.v[0] << "," << position.v[1] << "," << position.v[2] << "," << orientation.v[0] << "," << orientation.v[1] << "," << orientation.v[2] << "," << orientation.v[3] << "," << confidenceEnumMapping(confidence_level) << "," << endl;
								}

							}
					}
				}
				else
				{
					//  It should never hit timeout when K4A_WAIT_INFINITE is set.
					std::cout << "Error! Pop body frame result time out!" << std::endl;
					break;
				}
			}
			else
			{
				// It should never hit time out when K4A_WAIT_INFINITE is set.
				std::cout << "Error! Get depth frame time out!" << std::endl;
				break;
			}
		} while (frame_count < FRAME_COUNT);
		std::cout << "Finished body tracking processing!" << std::endl;

	}
	catch (const std::exception & e)
	{
		std::cerr << "Failed with exception:" << std::endl
			<< "    " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
