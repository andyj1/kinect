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

#include "colors.h"
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
using namespace cv;
using namespace color;

static cv::Mat color_to_opencv(const k4a::image &im);
void arun(Mat &main, Mat &secondary, Mat &R, Mat &T);
std::vector<float> computeJointAngles(k4abt_body_t body);
string confidenceEnumMapping(k4abt_joint_confidence_level_t confidence_level);
vector<double> print_body_information(k4abt_body_t body);
void crossproduct (cv::Vec3f &ans, cv::Vec3f &p1, cv::Vec3f &p2);


static int ANGLE_FRAME_ROW_COUNT = 0;
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


void crossproduct (cv::Vec3f &ans, cv::Vec3f &p1, cv::Vec3f &p2)
{
    ans[0] = p1[1]*p2[2] -p1[2]*p2[1];
    ans[1] = p1[0]*p2[2] -p1[2]*p2[0];
    ans[2] = p1[0]*p2[1] -p1[1]*p2[0];
}
std::vector<float> computeJointAngles(k4abt_body_t avg_body)
{
    // plot angles from 3D positions
    // A: 12-13-14
    cv::Vec3f a1(avg_body.skeleton.joints[12].position.v[0] - avg_body.skeleton.joints[13].position.v[0], \
                 avg_body.skeleton.joints[12].position.v[1] - avg_body.skeleton.joints[13].position.v[1], \
                 avg_body.skeleton.joints[12].position.v[2] - avg_body.skeleton.joints[13].position.v[2]);
    cv::Vec3f a2(avg_body.skeleton.joints[13].position.v[0] - avg_body.skeleton.joints[14].position.v[0], \
                 avg_body.skeleton.joints[13].position.v[1] - avg_body.skeleton.joints[14].position.v[1], \
                 avg_body.skeleton.joints[13].position.v[2] - avg_body.skeleton.joints[14].position.v[2]);
    cv::Vec3f across;
    crossproduct(across, a1, a2);
    float adot = a1.dot(a2);
    float A = atan2(norm(across), adot);

    // B: 5-6-7
    cv::Vec3f b1(avg_body.skeleton.joints[5].position.v[0] - avg_body.skeleton.joints[6].position.v[0], \
                 avg_body.skeleton.joints[5].position.v[1] - avg_body.skeleton.joints[6].position.v[1], \
                 avg_body.skeleton.joints[5].position.v[2] - avg_body.skeleton.joints[6].position.v[2]);
    cv::Vec3f b2(avg_body.skeleton.joints[6].position.v[0] - avg_body.skeleton.joints[7].position.v[0], \
                 avg_body.skeleton.joints[6].position.v[1] - avg_body.skeleton.joints[7].position.v[1], \
                 avg_body.skeleton.joints[6].position.v[2] - avg_body.skeleton.joints[7].position.v[2]);
    cv::Vec3f bcross;
    crossproduct(bcross, b1, b2);
    float bdot = b1.dot(b2);
    float B = atan2(norm(bcross), bdot);

    // C: 11-12-13
    cv::Vec3f c1(avg_body.skeleton.joints[11].position.v[0] - avg_body.skeleton.joints[12].position.v[0], \
                 avg_body.skeleton.joints[11].position.v[1] - avg_body.skeleton.joints[12].position.v[1], \
                 avg_body.skeleton.joints[11].position.v[2] - avg_body.skeleton.joints[12].position.v[2]);
    cv::Vec3f c2(avg_body.skeleton.joints[12].position.v[0] - avg_body.skeleton.joints[13].position.v[0], \
                 avg_body.skeleton.joints[12].position.v[1] - avg_body.skeleton.joints[13].position.v[1], \
                 avg_body.skeleton.joints[12].position.v[2] - avg_body.skeleton.joints[13].position.v[2]);
    cv::Vec3f ccross;
    crossproduct(ccross, c1, c2);
    float cdot = c1.dot(c2);
    float C = atan2(norm(ccross), cdot);

    // D: 4-5-6
    cv::Vec3f d1(avg_body.skeleton.joints[4].position.v[0] - avg_body.skeleton.joints[5].position.v[0], \
                 avg_body.skeleton.joints[4].position.v[1] - avg_body.skeleton.joints[5].position.v[1], \
                 avg_body.skeleton.joints[4].position.v[2] - avg_body.skeleton.joints[5].position.v[2]);
    cv::Vec3f d2(avg_body.skeleton.joints[5].position.v[0] - avg_body.skeleton.joints[6].position.v[0], \
                 avg_body.skeleton.joints[5].position.v[1] - avg_body.skeleton.joints[6].position.v[1], \
                 avg_body.skeleton.joints[5].position.v[2] - avg_body.skeleton.joints[6].position.v[2]);
    cv::Vec3f dcross;
    crossproduct(dcross, d1, d2);
    float ddot = d1.dot(d2);
    float D = atan2(norm(dcross), ddot);

    // E: 1-0-22
    cv::Vec3f e1(avg_body.skeleton.joints[1].position.v[0] - avg_body.skeleton.joints[0].position.v[0], \
                 avg_body.skeleton.joints[1].position.v[1] - avg_body.skeleton.joints[0].position.v[1], \
                 avg_body.skeleton.joints[1].position.v[2] - avg_body.skeleton.joints[0].position.v[2]);
    cv::Vec3f e2(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[22].position.v[0], \
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[22].position.v[1], \
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[22].position.v[2]);
    cv::Vec3f ecross;
    crossproduct(ecross, e1, e2);
    float edot = e1.dot(e2);
    float E = atan2(norm(ecross), edot);
    
    // F: 1-0-18
    cv::Vec3f f1(avg_body.skeleton.joints[1].position.v[0] - avg_body.skeleton.joints[0].position.v[0], \
                 avg_body.skeleton.joints[1].position.v[1] - avg_body.skeleton.joints[0].position.v[1], \
                 avg_body.skeleton.joints[1].position.v[2] - avg_body.skeleton.joints[0].position.v[2]);
    cv::Vec3f f2(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[18].position.v[0], \
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[18].position.v[1], \
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[18].position.v[2]);
    cv::Vec3f fcross;
    crossproduct(fcross, f1, f2);
    float fdot = f1.dot(f2);
    float F = atan2(norm(fcross), fdot);
    
    // G: 0-22-23
    cv::Vec3f g1(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[22].position.v[0], \
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[22].position.v[1], \
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[22].position.v[2]);
    cv::Vec3f g2(avg_body.skeleton.joints[22].position.v[0] - avg_body.skeleton.joints[23].position.v[0], \
                 avg_body.skeleton.joints[22].position.v[1] - avg_body.skeleton.joints[23].position.v[1], \
                 avg_body.skeleton.joints[22].position.v[2] - avg_body.skeleton.joints[23].position.v[2]);
    cv::Vec3f gcross;
    crossproduct(gcross, g1, g2);
    float gdot = g1.dot(g2);
    float G = atan2(norm(gcross), gdot);

    // H: 0-18-19
    cv::Vec3f h1(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[18].position.v[0], \
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[18].position.v[1], \
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[18].position.v[2]);
    cv::Vec3f h2(avg_body.skeleton.joints[18].position.v[0] - avg_body.skeleton.joints[19].position.v[0], \
                 avg_body.skeleton.joints[18].position.v[1] - avg_body.skeleton.joints[19].position.v[1], \
                 avg_body.skeleton.joints[18].position.v[2] - avg_body.skeleton.joints[19].position.v[2]);
    cv::Vec3f hcross;
    crossproduct(hcross, h1, h2);
    float hdot = h1.dot(h2);
    float H = atan2(norm(hcross), hdot);

    // I: 22-23-24
    cv::Vec3f i1(avg_body.skeleton.joints[22].position.v[0] - avg_body.skeleton.joints[23].position.v[0], \
                 avg_body.skeleton.joints[22].position.v[1] - avg_body.skeleton.joints[23].position.v[1], \
                 avg_body.skeleton.joints[22].position.v[2] - avg_body.skeleton.joints[23].position.v[2]);
    cv::Vec3f i2(avg_body.skeleton.joints[23].position.v[0] - avg_body.skeleton.joints[24].position.v[0], \
                 avg_body.skeleton.joints[23].position.v[1] - avg_body.skeleton.joints[24].position.v[1], \
                 avg_body.skeleton.joints[23].position.v[2] - avg_body.skeleton.joints[24].position.v[2]);
    cv::Vec3f icross;
    crossproduct(icross, i1, i2);
    float idot = i1.dot(i2);
    float I = atan2(norm(icross), idot);

    // J: 18-19-20
    cv::Vec3f j1(avg_body.skeleton.joints[18].position.v[0] - avg_body.skeleton.joints[19].position.v[0], \
                 avg_body.skeleton.joints[18].position.v[1] - avg_body.skeleton.joints[19].position.v[1], \
                 avg_body.skeleton.joints[18].position.v[2] - avg_body.skeleton.joints[19].position.v[2]);
    cv::Vec3f j2(avg_body.skeleton.joints[19].position.v[0] - avg_body.skeleton.joints[20].position.v[0], \
                 avg_body.skeleton.joints[19].position.v[1] - avg_body.skeleton.joints[20].position.v[1], \
                 avg_body.skeleton.joints[19].position.v[2] - avg_body.skeleton.joints[20].position.v[2]);
    cv::Vec3f jcross;
    crossproduct(jcross, j1, j2);
    float jdot = j1.dot(j2);
    float J = atan2(norm(jcross), jdot);
    
    // K: 23-24-25
    cv::Vec3f k1(avg_body.skeleton.joints[23].position.v[0] - avg_body.skeleton.joints[24].position.v[0], \
                 avg_body.skeleton.joints[23].position.v[1] - avg_body.skeleton.joints[24].position.v[1], \
                 avg_body.skeleton.joints[23].position.v[2] - avg_body.skeleton.joints[24].position.v[2]);
    cv::Vec3f k2(avg_body.skeleton.joints[24].position.v[0] - avg_body.skeleton.joints[25].position.v[0], \
                 avg_body.skeleton.joints[24].position.v[1] - avg_body.skeleton.joints[25].position.v[1], \
                 avg_body.skeleton.joints[24].position.v[2] - avg_body.skeleton.joints[25].position.v[2]);
    cv::Vec3f kcross;
    crossproduct(kcross, k1, k2);
    float kdot = k1.dot(k2);
    float K = atan2(norm(kcross), kdot);
    
    // L: 19-20-21
    cv::Vec3f l1(avg_body.skeleton.joints[19].position.v[0] - avg_body.skeleton.joints[20].position.v[0], \
                 avg_body.skeleton.joints[19].position.v[1] - avg_body.skeleton.joints[20].position.v[1], \
                 avg_body.skeleton.joints[19].position.v[2] - avg_body.skeleton.joints[20].position.v[2]);
    cv::Vec3f l2(avg_body.skeleton.joints[20].position.v[0] - avg_body.skeleton.joints[21].position.v[0], \
                 avg_body.skeleton.joints[20].position.v[1] - avg_body.skeleton.joints[21].position.v[1], \
                 avg_body.skeleton.joints[20].position.v[2] - avg_body.skeleton.joints[21].position.v[2]);
    cv::Vec3f lcross;
    crossproduct(lcross, l1, l2);
    float ldot = l1.dot(l2);
    float L = atan2(norm(lcross), ldot);

    vector<float> angles = {A,B,C,D,E,F,G,H,I,J,K,L};
    for (int ii = 0; ii <angles.size(); ++ii)
    {
        if (angles[ii] < 0)
        {
            angles[ii] += 2*M_PI;
        }
        angles[ii] = M_PI - angles[ii];
    }
    std::cerr << "finished computing joint angles"<< std::endl;
    return angles;

}

// #undef main
int main()
{
	// output file stream
	ofstream outfile;
	outfile.open("../saved_data/joints_single_master.csv", ios::out);
	outfile << ",,Position,,,Orientation,,,,Confidence Level" << endl;
	outfile << "Body ID," << "Joint #," << "x,y,z," << "x,y,z,w" << endl;

    ofstream outfile_angles;
    outfile_angles.open("../saved_data/joints_single_angles.csv", ios::out);
    outfile_angles << "Frame,A,B,C,D,E,F,G,H,I,J,K,L" << std::endl;
    outfile_angles.close();

	try
	{
		k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
		device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

		k4a::device device = k4a::device::open(0);
		device.start_cameras(&device_config);
		k4a::calibration sensor_calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution);
		k4abt::tracker tracker = k4abt::tracker::create(sensor_calibration);

		// intrinsic matrix
		const k4a_calibration_intrinsic_parameters_t::_param &main_i = device.get_calibration(device_config.depth_mode, device_config.color_resolution).color_camera_calibration.intrinsics.parameters.param;
		cv::Matx33f main_intrinsic_matrix = cv::Matx33f::eye();
		// cv::Matx33f secondary_intrinstic_matrix = cv::Matx33f::eye();
		main_intrinsic_matrix(0, 0) = main_i.fx;
		main_intrinsic_matrix(1, 1) = main_i.fy;
		main_intrinsic_matrix(0, 2) = main_i.cx;
		main_intrinsic_matrix(1, 2) = main_i.cy;

		int frame_count = 0;
		do
		{
			k4a::capture sensor_capture;
			if (device.get_capture(&sensor_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
			{
				frame_count++;
				// k4a::image main_color_image = sensor_capture.get_color_image();
				// cv::Mat cv_main_color_image = color_to_opencv(main_color_image);

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
							
							if (outfile.is_open())
							{
								outfile << body.id << "," << i << "," << position.v[0] << "," << position.v[1] << "," << position.v[2] << "," << orientation.v[0] << "," << orientation.v[1] << "," << orientation.v[2] << "," << orientation.v[3] << "," << confidenceEnumMapping(confidence_level) << "," << endl;
							}

						}
						
						cout <<"adding angles"<<endl;
						// compute joint angles
						vector<float> joint_angles = computeJointAngles(body);
						
						if (joint_angles.size() > 0)
						{
							int offset = 0;
							int index = 0;
							string angle;
							ofstream outfile_angles;
							outfile_angles.open("../saved_data/joints_single_angles.csv", ios::out|ios::app);
							outfile_angles << to_string(ANGLE_FRAME_ROW_COUNT) << ",";
							for(std::vector<float>::iterator it = joint_angles.begin(); it != joint_angles.end(); it++)
							{
								outfile_angles << *it*180/M_PI << ",";
								angle = (char)(index+65);
								
								std::cout << *it*180/M_PI << std::endl;
								
								// cv::putText(cv_main_color_image, angle+": "+to_string(*it*180/M_PI), cv::Point(cv_main_color_image.cols-200, 30+offset), FONT_HERSHEY_DUPLEX, 1, COLORS_darkorange, 1);
								offset += 30;
								index++;
							}
							ANGLE_FRAME_ROW_COUNT++;
							outfile_angles << std::endl;
							outfile_angles.close();
							
							// plot joints
							// std::vector<cv::Point> dataMain;
							// Mat mainstream(3, K4ABT_JOINT_COUNT, CV_32F);
							// std::vector<cv::Point3f> main_points;


							// // convert to point coordinates and 
							// for (int joint=0; joint < (int)K4ABT_JOINT_COUNT; joint++)
							// {
							// 	// for visualization of only using master vs. all averaged
							// 	main_points.push_back(cv::Point3f(body.skeleton.joints[joint].position.v[0],body.skeleton.joints[joint].position.v[1],body.skeleton.joints[joint].position.v[2]));
							// }
							// 	// Compute R, T from secondary to main coordinate space
							// Mat R;
							// Mat T;
							// arun(mainstream,mainstream, R, T); // R: [3x3], T: [1x3]

							// // Create zero distortion
							// cv::Mat distCoeffs(4,1,CV_32F);
							// distCoeffs.at<float>(0) = 0;
							// distCoeffs.at<float>(1) = 0;
							// distCoeffs.at<float>(2) = 0;
							// distCoeffs.at<float>(3) = 0;

							// std::vector<cv::Point2f> projectedPointsMain, projectedPointsAvg;
							// cv::Mat rvecR(3,1,cv::DataType<double>::type);//rodrigues rotation matrix
							// cv::Rodrigues(R,rvecR);
							
							// std::cerr << "projecting 3-D to 2-D..." << std::endl;
							// cv::projectPoints(main_points, rvecR, T, Mat(main_intrinsic_matrix), distCoeffs, projectedPointsMain);
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





static cv::Mat color_to_opencv(const k4a::image &im)
{
    cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void *)im.get_buffer());
    cv::Mat cv_image_no_alpha;
    cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
    return cv_image_no_alpha;
}


void arun(cv::Mat &streamfrom, cv::Mat &streamto, cv::Mat &R, cv::Mat &T)
{
    // find mean across the row (all joints for each x,y,z)
    cv::Mat avg_streamfrom, avg_streamto;                     // main, secondary: [3x32]
    cv::reduce(streamfrom, avg_streamfrom, 1, CV_REDUCE_AVG); // avg_main: [3x1]
    cv::reduce(streamto, avg_streamto, 1, CV_REDUCE_AVG);

    // find deviations from the mean
    cv::Mat rep_avg_streamfrom, rep_avg_streamto;
    cv::repeat(avg_streamfrom, 1, K4ABT_JOINT_COUNT, rep_avg_streamfrom); // rep_avg_main: [3x32]
    cv::repeat(avg_streamto, 1, K4ABT_JOINT_COUNT, rep_avg_streamto);

    cv::Mat streamfrom_sub, streamto_sub;
    cv::subtract(streamfrom, rep_avg_streamfrom, streamfrom_sub);
    cv::subtract(streamto, rep_avg_streamto, streamto_sub);

    // take singular value decomposition and compute R and T matrices
    Mat s, u, v;
    cv::SVDecomp(streamfrom_sub * streamto_sub.t(), s, u, v);
    v = v.t();
    R = v * u.t();
    double det = cv::determinant(R);

    // std::cout << "determinant: "<<det<<std::endl;
    if (det >= 0)
    {
        // T = avg_main - (R * avg_secondary); // T: [3x1]
        subtract(avg_streamto, (R * avg_streamfrom), T);
    }
    else
    {
        T = cv::Mat(1, 3, CV_32F, cv::Scalar(1, 1, 1));
    }
    // R: [3x3], T: [3x1]
    // std::cout << endl << "R "<<R.size()<<" : \n" << R << "\nT "<<T.size()<<" :\n" << T << std::endl;
}