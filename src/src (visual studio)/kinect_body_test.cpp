//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include <fstream>
//#include <string>
//
//// NuGet Package: Microsoft.Azure.Kinect.Sensor
//#include <k4a/k4a.h>
//#include <k4arecord/record.h>
//#include <k4arecord/playback.h>
//
//// Kinect Body Tracking SDK: "C:\Program Files\Azure Kinect Body Tracking SDK\sdk\include"
//#include <k4abt.h>
//
//// Handle to a k4a body tracking frame.
//#include <k4abttypes.h>
//
//// Json
//#include <nlohmann/json.hpp>
//
//#define FRAME_COUNT 1
//
//#define VERIFY(result, error)                                                                            \
//    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
//    {                                                                                                    \
//        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
//        exit(1);                                                                                         \
//    };
//
//using namespace std;
//
//string confidenceMap(k4abt_joint_confidence_level_t confidence_level)
//{
//	string resultString;
//	switch (confidence_level)
//	{
//	case K4ABT_JOINT_CONFIDENCE_NONE: // 0
//		resultString += "The joint is out of range(too far from depth camera)";
//		break;
//	case K4ABT_JOINT_CONFIDENCE_LOW: // 1
//		resultString += "The joint is not observed(likely due to occlusion) - predicted joint pose";
//		break;
//	case K4ABT_JOINT_CONFIDENCE_MEDIUM:
//		resultString += "Medium confidence in joint pose";
//		break;
//		// beyond this will be supported later
//	case K4ABT_JOINT_CONFIDENCE_HIGH:
//		resultString += "High confidence in joint pose";
//		break;
//	default:
//		break;
//	}
//	return resultString;
//}
//
//
//#undef main
//int main()
//{
//	try 
//	{
//		k4a_device_t device = NULL;
//		VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");
//
//		// Start camera. Make sure depth camera is enabled.
//		k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
//		deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
//		deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
//		VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");
//		
//		k4a_calibration_t sensor_calibration;
//		VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
//			"Get depth camera calibration failed!");
//
//		k4abt_tracker_t tracker = NULL;
//		k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
//		VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");
//
//		// output file stream
//		ofstream outfile;
//		outfile.open("./joint_info.csv", ios::out, ios::trunc);
//
//		int frame_count = 0;
//		do
//		{
//			printf("device getting capture...\n");
//			k4a_capture_t sensor_capture;
//			int32_t TIMEOUT_IN_MS = 1000;
//			k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE); // TIMEOUT_IN_MS
//			if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
//			{
//				std::cout << "Start processing frame " << frame_count << std::endl;
//
//				frame_count++;
//				k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
//
//				k4a_capture_release(sensor_capture); // Remember to release the sensor capture once you finish using it
//				if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT)
//				{
//					// It should never hit timeout when K4A_WAIT_INFINITE is set.
//					printf("Error! Add capture to tracker process queue timeout!\n");
//					break;
//				}
//				else if (queue_capture_result == K4A_WAIT_RESULT_FAILED)
//				{
//					printf("Error! Add capture to tracker process queue failed!\n");
//					break;
//				}
//
//				printf("body tracker being created...\n");
//				// real-time processing
//				k4abt_frame_t body_frame = NULL;
//				k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
//				if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
//				{
//					
//					// Successfully popped the body tracking result. Start your processing
//					size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
//					printf("%zu bodies are detected!\n", num_bodies);
//
//					// access body ID
//					/*for (size_t i = 0; i < num_bodies; i++)
//					{
//						k4abt_skeleton_t skeleton;
//						k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
//						uint32_t id = k4abt_frame_get_body_id(body_frame, i);
//					}*/
//
//					// access body index map
//					//k4a_image_t body_index_map = k4abt_frame_get_body_index_map(body_frame);
//					//// Do your work with the body index map
//					//k4a_image_release(body_index_map);
//
//					// access input capture
//					//k4a_capture_t input_capture = k4abt_frame_get_capture(body_frame);
//					//// Do your work with the input capture
//					//k4a_capture_release(input_capture);
//					
//
//					// loop through all body joints
//					for (size_t index = 0; index < num_bodies; index++)
//					{						
//						k4abt_skeleton_t skeleton;
//						k4a_result_t result = k4abt_frame_get_body_skeleton(body_frame, index, &skeleton);
//						if (result != K4A_RESULT_FAILED)
//						{	
//							 if (outfile.is_open())
//							 {
//							 	outfile << "[Position] Vector array," << long float(*(&skeleton)->joints->position.v) << endl;
//							 	outfile << "[Position] (x-y-z) coordinate," << long float((&skeleton)->joints->position.xyz.x) << ", " << long float((&skeleton)->joints->position.xyz.y) << ", " << long float((&skeleton)->joints->position.xyz.z) << endl;
//
//							 	outfile << "[orientation] (x-y-z-w)," << long float((&skeleton)->joints->orientation.wxyz.x) << ", " << long float((&skeleton)->joints->orientation.wxyz.y) << ", " << long float((&skeleton)->joints->orientation.wxyz.z) << ", " << long float((&skeleton)->joints->orientation.wxyz.w) << endl;
//							 	outfile << "[orientation] Vector array," << *(&skeleton)->joints->orientation.v << endl;
//
//							 	outfile << "[confidence level]," << (&skeleton)->joints->confidence_level << ", " << confidenceMap((&skeleton)->joints->confidence_level) << endl;
//
//							 }
//							 // type float
//							 printf("[Position] Vector array: %i \n", (&skeleton)->joints->position.v);
//							 printf("[Position] (x,y,z) coordinate: (%i,%i,%i) \n", (&skeleton)->joints->position.xyz.x, (&skeleton)->joints->position.xyz.y, (&skeleton)->joints->position.xyz.z);
//
//							 printf("[orientation] (x,y,z,w): (%i,%i,%i,%i) \n", (&skeleton)->joints->orientation.wxyz.x, (&skeleton)->joints->orientation.wxyz.y, (&skeleton)->joints->orientation.wxyz.z, (&skeleton)->joints->orientation.wxyz.w);
//							 printf("[orientation] Vector array: %i \n", *(&skeleton)->joints->orientation.v);
//					
//							 printf("[confidence level]: %i \n", (&skeleton)->joints->confidence_level);
//						}
//					}
//
//
//
//					// End processing 
//					k4abt_frame_release(body_frame); // Remember to release the body frame once you finish using it
//				}
//				else if (pop_frame_result == K4A_WAIT_RESULT_TIMEOUT)
//				{
//					//  It should never hit timeout when K4A_WAIT_INFINITE is set.
//					printf("Error! Pop body frame result timeout!\n");
//					break;
//				}
//				else
//				{
//					printf("Pop body frame result failed!\n");
//					break;
//				}
//			}
//			else if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT)
//			{
//				// It should never hit time out when K4A_WAIT_INFINITE is set.
//				printf("Error! Get depth frame time out!\n");
//				break;
//			}
//			else
//			{
//				printf("Get depth capture returned error: %d\n", get_capture_result);
//				break;
//			}
//	
//		} while (frame_count < FRAME_COUNT);
//	
//		printf("Finished body tracking processing!\n");
//		k4abt_tracker_shutdown(tracker);
//		k4abt_tracker_destroy(tracker);
//		k4a_device_stop_cameras(device);
//		k4a_device_close(device);
//	}
//    catch (const std::exception& e)
//    {
//        std::cerr << "Failed with exception:" << std::endl
//            << "    " << e.what() << std::endl;
//        return 1;
//    }
//	return 0;
//}