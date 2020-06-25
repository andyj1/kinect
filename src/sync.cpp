// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <limits>
#include <list>
#include <numeric> // accumulate

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "transformation.h"
#include "MultiDeviceCapturer.h"
#include "colors.h"
#include "sync.hpp"

using namespace color;
using namespace cv;
using namespace std;

static int ANGLE_FRAME_ROW_COUNT = 0;

// Allowing at least 160 microseconds between depth cameras should ensure they do not interfere with one another.
constexpr uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;

// ideally, we could generalize this to many OpenCV types
static cv::Mat color_to_opencv(const k4a::image &im);
static cv::Mat depth_to_opencv(const k4a::image &im);
static cv::Matx33f calibration_to_color_camera_matrix(const k4a::calibration &cal);
static Transformation get_depth_to_color_transformation_from_calibration(const k4a::calibration &cal);
static k4a::calibration construct_device_to_device_calibration(const k4a::calibration &main_cal,
                                                               const k4a::calibration &secondary_cal,
                                                               const Transformation &secondary_to_main);
static vector<float> calibration_to_color_camera_dist_coeffs(const k4a::calibration &cal);
static bool find_chessboard_corners_helper(const cv::Mat &main_color_image,
                                           const cv::Mat &secondary_color_image,
                                           const cv::Size &chessboard_pattern,
                                           vector<cv::Point2f> &main_chessboard_corners,
                                           vector<cv::Point2f> &secondary_chessboard_corners);
static Transformation stereo_calibration(const k4a::calibration &main_calib,
                                         const k4a::calibration &secondary_calib,
                                         const vector<vector<cv::Point2f>> &main_chessboard_corners_list,
                                         const vector<vector<cv::Point2f>> &secondary_chessboard_corners_list,
                                         const cv::Size &image_size,
                                         const cv::Size &chessboard_pattern,
                                         float chessboard_square_length);
static k4a_device_configuration_t get_master_config();
static k4a_device_configuration_t get_subordinate_config();
static Transformation calibrate_devices(MultiDeviceCapturer &capturer,
                                        const k4a_device_configuration_t &main_config,
                                        const k4a_device_configuration_t &secondary_config,
                                        const cv::Size &chessboard_pattern,
                                        float chessboard_square_length,
                                        double calibration_timeout);
static k4a::image create_depth_image_like(const k4a::image &im);

// custom functions
void print_body_index_map_middle_line(k4a::image body_index_map);
k4a_float3_t get_average_position_xyz(k4a_float3_t main_position, k4a_float3_t secondary_position, int main_or_secondary);
k4a_quaternion_t get_average_quaternion_xyzw(k4a_quaternion_t main_quaternion, k4a_quaternion_t secondary_quaternion, int main_or_secondar);
int get_average_confidence(k4abt_joint_confidence_level_t mainCI, k4abt_joint_confidence_level_t secondaryCI);
string confidenceMap(k4abt_joint_confidence_level_t confidence_level);

void processBodyData(k4abt_body_t main_body, vector<vector<k4abt_body_t>> secondary_body_vector, cv::Mat &main, cv::Matx33f main_intrinsic_matrix, vector<int> validSubDevices);
void transformBody(k4abt_body_t &main_body, k4abt_body_t &secondary_body);
void arun(Mat &main, Mat &secondary, Mat &R, Mat &T);
void plotBody(k4abt_body_t main_body, k4abt_body_t avg_body, cv::Mat main, cv::Matx33f main_intrinsic_matrix);
std::vector<float> computeJointAngles(k4abt_body_t avg_body);

int main(int argc, char **argv)
{
    // output file stream
    ofstream outfile_orig;
    outfile_orig.open("../saved_data/joints_gen_master.csv", ios::out);
    outfile_orig << ",,Position,,,Orientation,,,,Confidence Level" << std::endl;
    outfile_orig << "Body ID,"
                 << "Joint #,"
                 << "x,y,z,"
                 << "x,y,z,w" << std::endl;

    ofstream outfile_sub1;
    outfile_sub1.open("../saved_data/joints_gen_sub1.csv", ios::out);
    outfile_sub1 << ",,Position,,,Orientation,,,,Confidence Level" << std::endl;
    outfile_sub1 << "Body ID,"
                 << "Joint #,"
                 << "x,y,z,"
                 << "x,y,z,w" << std::endl;

    ofstream outfile_sub2;
    outfile_sub2.open("../saved_data/joints_gen_sub2.csv", ios::out);
    outfile_sub2 << ",,Position,,,Orientation,,,,Confidence Level" << std::endl;
    outfile_sub2 << "Body ID,"
                 << "Joint #,"
                 << "x,y,z,"
                 << "x,y,z,w" << std::endl;

    ofstream outfile_avg;
    outfile_avg.open("../saved_data/joints_gen_sync.csv", ios::out);
    outfile_avg << ",,Position,,,Orientation,,,,Confidence Level" << std::endl;
    outfile_avg << "Body ID,"
                << "Joint #,"
                << "x,y,z,"
                << "x,y,z,w" << std::endl;
    outfile_avg.close();

    ofstream outfile_angles;
    outfile_angles.open("../saved_data/joints_gen_angles.csv", ios::out);
    outfile_angles << "Frame,A,B,C,D,E,F,G,H,I,J,K,L" << std::endl;
    outfile_angles.close();

    float chessboard_square_length = 0.; // must be included in the input params
    cv::Size chessboard_pattern(0, 0);   // height, width. Both need to be set.
    size_t num_devices = 0;
    uint16_t depth_threshold = 1000;                                  // default to 1 meter
    int32_t color_exposure_usec = 8000;                               // somewhat reasonable default exposure time
    int32_t powerline_freq = 2;                                        // default to a 60 Hz powerline
    double calibration_timeout = std::numeric_limits<double>::max();  // 60.0; // default to timing out after 60s of trying to get calibrated
    double greenscreen_duration = std::numeric_limits<double>::max(); // run forever

    vector<uint32_t> device_indices{0, 1, 2}; // Set up a MultiDeviceCapturer to handle getting many synchronous captures
                                              // Note that the order of indices in device_indices is not necessarily
                                              // preserved because MultiDeviceCapturer tries to find the master device based
                                              // on which one has sync out plugged in. Start with just { 0 }, and add
                                              // another if needed

    if (argc < 5)
    {
        std::cout << "Usage: sync_program <num-cameras> <board-height> <board-width> <board-square-length> "
                     "[depth-threshold-mm (default 1000)] [color-exposure-time-usec (default 8000)] "
                  << std::endl;
        cerr << "Not enough arguments!\n";
        std::exit(1);
    }
    else
    {
        num_devices = static_cast<size_t>(atoi(argv[1]));
        if (num_devices > k4a::device::get_installed_count())
        {
            cerr << "Not enough cameras plugged in!\n";
            std::exit(1);
        }
        chessboard_pattern.height = atoi(argv[2]);
        chessboard_pattern.width = atoi(argv[3]);
        chessboard_square_length = static_cast<float>(atof(argv[4]));

        // optional
        if (argc > 5)
        {
            depth_threshold = static_cast<uint16_t>(atoi(argv[5]));
            if (argc > 6)
            {
                color_exposure_usec = atoi(argv[6]);
                if (argc > 7)
                {
                    powerline_freq = atoi(argv[7]);
                    if (argc > 8)
                    {
                        calibration_timeout = atof(argv[8]);
                    }
                }
            }
        }
    }

    if (num_devices < 2)
    {
        cerr << "Invalid choice for number of devices!\n";
        std::exit(1);
    }
    else if (num_devices == 2)
    {
        device_indices.emplace_back(1);
    }
    if (chessboard_pattern.height == 0)
    {
        cerr << "Chessboard height is not properly set!\n";
        std::exit(1);
    }
    if (chessboard_pattern.width == 0)
    {
        cerr << "Chessboard height is not properly set!\n";
        std::exit(1);
    }
    if (chessboard_square_length == 0.)
    {
        cerr << "Chessboard square size is not properly set!\n";
        std::exit(1);
    }

    std::cout << "Chessboard height: " << chessboard_pattern.height << ". Chessboard width: " << chessboard_pattern.width
              << ". Chessboard square length: " << chessboard_square_length << std::endl;
    std::cout << "Depth threshold: : " << depth_threshold << ". Color exposure time: " << color_exposure_usec
              << ". Powerline frequency mode: " << powerline_freq << std::endl;

    // stores opened devices in 'capturer'
    // sets the color exposure, color mode, powerline freq
    // sets the first 'sync out' mode as the master
    MultiDeviceCapturer capturer(device_indices, color_exposure_usec, powerline_freq);

    // Create configurations for devices
    k4a_device_configuration_t main_config = get_master_config();
    k4a_device_configuration_t secondary_config = get_subordinate_config();

    // Get calibration info for the master device
    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,
                                                                                     main_config.color_resolution);

    // Set up a transformation
    k4a::transformation main_depth_to_main_color(main_calibration);

    // start master device by main config (MASTER mode), and the rest as secondary config iteratively (SUBORDINATE mode)
    capturer.start_devices(main_config, secondary_config);

    if (num_devices > 1)
    {
        // This wraps all the device-to-device details
        Transformation tr_secondary_color_to_main_color = calibrate_devices(capturer,
                                                                            main_config,
                                                                            secondary_config,
                                                                            chessboard_pattern,
                                                                            chessboard_square_length,
                                                                            calibration_timeout);
        // Get calibration info for the second device (same for all subordinate modes)
        k4a::calibration secondary_calibration =
            capturer.get_subordinate_device_by_index(0).get_calibration(secondary_config.depth_mode,
                                                                        secondary_config.color_resolution);
        // k4a::calibration secondary_calibration2 =
        //     capturer.get_subordinate_device_by_index(1).get_calibration(secondary_config.depth_mode,
        //                                                                 secondary_config.color_resolution);
        

        // ============ initialize body tracker (same for all since calibration config is the same; this is to speed up)
        k4abt::tracker main_tracker = k4abt::tracker::create(main_calibration);
        k4abt::tracker secondary_tracker = k4abt::tracker::create(secondary_calibration);

        k4abt_body_t main_body;
        vector<vector<k4abt_body_t>> secondary_body_vector(num_devices - 1, std::vector<k4abt_body_t>()); // per body per tracker

        k4abt::frame main_body_frame;
        vector<k4abt::frame> secondary_body_frame(num_devices - 1); // per tracker

        const char *window_title;
        string sec_title;
        k4abt_body_t secondary_body;

        // continuously run frames
        cvDestroyWindow("Chessboard view from main");
        cvDestroyWindow("Chessboard view from secondary");
        uint32_t num_bodies_min = 0;
        vector<int> validSubDevices;
        while (true)
        {
            validSubDevices.clear();

            // =================================captures for body tracker
            vector<k4a::capture> captures;
            captures = capturer.get_synchronized_captures(secondary_config, true);

            // prepare main tracker body frame
            if (!main_tracker.enqueue_capture(captures[0], std::chrono::milliseconds(K4A_WAIT_INFINITE)))
            {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                std::cout << "Error! Add capture to tracker process queue timeout!" << std::endl;
                break;
            }

            // =================================track from master camera
            try
            {
                // get result frame
                main_body_frame = main_tracker.pop_result();
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }

            uint32_t main_num_bodies = 0;
            if (main_body_frame != nullptr)
            {
                main_num_bodies = main_body_frame.get_num_bodies();
            }

            // store master joints data if body is present
            if (main_num_bodies > 0)
            {
                for (uint32_t body_idx = 0; body_idx < num_bodies_min; body_idx++)
                {
                    main_body = main_body_frame.get_body(body_idx);
                    if (main_num_bodies > 0)
                    {
                        for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
                        {
                            // master device
                            k4a_float3_t main_position = main_body.skeleton.joints[i].position;
                            k4a_quaternion_t main_orientation = main_body.skeleton.joints[i].orientation;
                            k4abt_joint_confidence_level_t main_confidence_level = main_body.skeleton.joints[i].confidence_level;
                            if (outfile_orig.is_open())
                            {
                                outfile_orig << main_body.id << "," << i << "," << main_position.v[0] << "," << main_position.v[1] << "," << main_position.v[2] << "," << main_orientation.v[0] << "," << main_orientation.v[1] << "," << main_orientation.v[2] << "," << main_orientation.v[3] << "," << confidenceMap(main_confidence_level) << "," << std::endl;
                            }
                        }
                    }
                }
            }

            // =================================display synced time of each camera on bottom right of view
            // these color and depth images in CV are for constantly running visualization purposes
            k4a::image main_color_image = captures[0].get_color_image();
            cv::Mat cv_main_color_image = color_to_opencv(main_color_image);
            // timestamps of the captures
            std::chrono::microseconds master_color_time = captures[0].get_color_image().get_device_timestamp();
            std::chrono::microseconds sub1_color_time = captures[1].get_color_image().get_device_timestamp();
            std::chrono::microseconds sub2_color_time = captures[2].get_color_image().get_device_timestamp();
            cv::putText(cv_main_color_image, "Time (master): " + to_string(master_color_time.count() / 1000000) + "." + to_string(master_color_time.count() / 1000 % 1000) + " s", cv::Point(cv_main_color_image.cols * 2 / 3 + 20, cv_main_color_image.rows - 110), FONT_HERSHEY_DUPLEX, 1, COLORS_black, 1);
            cv::putText(cv_main_color_image, "Time (sub1): " + to_string(sub1_color_time.count() / 1000000) + "." + to_string(sub1_color_time.count() / 1000 % 1000) + " s", cv::Point(cv_main_color_image.cols * 2 / 3 + 20, cv_main_color_image.rows - 70), FONT_HERSHEY_DUPLEX, 1, COLORS_black, 1);
            cv::putText(cv_main_color_image, "Time (sub2): " + to_string(sub2_color_time.count() / 1000000) + "." + to_string(sub1_color_time.count() / 1000 % 1000) + " s", cv::Point(cv_main_color_image.cols * 2 / 3 + 20, cv_main_color_image.rows - 30), FONT_HERSHEY_DUPLEX, 1, COLORS_black, 1);

            // =================================intrinsics of the unit and extrinsics between color and depth camera in each device
            const k4a_calibration_intrinsic_parameters_t::_param &main_i = main_calibration.color_camera_calibration.intrinsics.parameters.param;
            // const k4a_calibration_intrinsic_parameters_t::_param &secondary_i = secondary_calibration.color_camera_calibration.intrinsics.parameters.param;
            // intrinsics: main_intrinsic_matrix, secondary_intrinsic_matrix
            cv::Matx33f main_intrinsic_matrix = cv::Matx33f::eye();
            main_intrinsic_matrix(0, 0) = main_i.fx;
            main_intrinsic_matrix(1, 1) = main_i.fy;
            main_intrinsic_matrix(0, 2) = main_i.cx;
            main_intrinsic_matrix(1, 2) = main_i.cy;

            // extrinsics: main_ext_tr, secondary_ext_tr
            const k4a_calibration_extrinsics_t &main_ext = main_calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
            const k4a_calibration_extrinsics_t &secondary_ext = secondary_calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
            Transformation main_ext_tr, secondary_ext_tr;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    main_ext_tr.R(i, j) = main_ext.rotation[i * 3 + j];
                    secondary_ext_tr.R(i, j) = secondary_ext.rotation[i * 3 + j];
                }
            }
            main_ext_tr.t = cv::Vec3d(main_ext.translation[0], main_ext.translation[1], main_ext.translation[2]);
            secondary_ext_tr.t = cv::Vec3d(secondary_ext.translation[0], secondary_ext.translation[1], secondary_ext.translation[2]);

            // =================================loop through subordinate mode devices
            cv::Mat cv_secondary_color_image;
            for (size_t sdev = 0; sdev < (int)(num_devices - 1); sdev++)
            {
                // reset
                secondary_body_frame.clear();

                k4a::image secondary_color_image = captures[sdev + 1].get_color_image();
                cv_secondary_color_image = color_to_opencv(secondary_color_image);

                try
                {
                    // feed in capture to the tracker
                    secondary_tracker.enqueue_capture(captures[sdev + 1], std::chrono::milliseconds(K4A_WAIT_INFINITE));
                    try
                    {
                        // get result frame
                        secondary_body_frame[sdev] = secondary_tracker.pop_result();
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << '\n';
                    }

                    // read from main and secondary trackers
                    uint32_t secondary_num_bodies = 0;
                    if (secondary_body_frame[sdev] != nullptr)
                    {
                        secondary_num_bodies = secondary_body_frame[sdev].get_num_bodies();
                        std::cerr << "secondary bodies(" << sdev << "):\t" << secondary_num_bodies << " bodies" << endl;
                    }

                    // proceed if at least master detected bodies
                    if (main_num_bodies > 0)
                    {
                        // possible scenarios
                        // 1. if no subordinate devices detected bodies
                        // 2. if master and 1 sub detected bodies
                        // 3. if master and 2 sub detected bodies

                        // for now, consider only for 1 body in the view
                        num_bodies_min = 1;
                        // num_bodies_min = std::min(main_num_bodies, secondary_num_bodies);
                        // std::cerr << num_bodies_min << " bodies detected!" << std::endl;

                        for (uint32_t body_idx = 0; body_idx < num_bodies_min; body_idx++)
                        {
                            // extract main body from main frame
                            main_body = main_body_frame.get_body(body_idx);

                            // skip if this device doesn't detect any bodies
                            if (secondary_num_bodies <= 0)
                            {
                                std::cerr << "sub device:\t" << sdev << " didn't find any devices --> " << secondary_num_bodies << std::endl;
                            }
                            else
                            {
                                // store sub device body
                                secondary_body_vector[sdev].push_back(secondary_body_frame[sdev].get_body(body_idx));

                                // append this device to valid subordinate devices
                                std::cerr << "adding valid sub device:\t" << sdev << std::endl;
                                validSubDevices.push_back(sdev);

                                // transform sub to master and store directly
                                transformBody(main_body, secondary_body_vector[sdev][body_idx]);

                                // for each body, store joints data in csv
                                for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
                                {
                                    // store first sub device joints data
                                    if (sdev == 0)
                                    {
                                        k4a_float3_t sub1_position = secondary_body_vector[0][body_idx].skeleton.joints[i].position;
                                        k4a_quaternion_t sub1_orientation = secondary_body_vector[0][body_idx].skeleton.joints[i].orientation;
                                        k4abt_joint_confidence_level_t sub1_confidence_level = secondary_body_vector[0][body_idx].skeleton.joints[i].confidence_level;
                                        if (outfile_sub1.is_open())
                                        {
                                            outfile_sub1 << secondary_body_vector[0][body_idx].id << "," << i << "," << sub1_position.v[0] << "," << sub1_position.v[1] << "," << sub1_position.v[2] << "," << sub1_orientation.v[0] << "," << sub1_orientation.v[1] << "," << sub1_orientation.v[2] << "," << sub1_orientation.v[3] << "," << confidenceMap(sub1_confidence_level) << "," << std::endl;
                                        }
                                    }
                                    else if (sdev == 1)
                                    // store subsequent sub device joints data
                                    {
                                        k4a_float3_t sub2_position = secondary_body_vector[1][body_idx].skeleton.joints[i].position;
                                        k4a_quaternion_t sub2_orientation = secondary_body_vector[1][body_idx].skeleton.joints[i].orientation;
                                        k4abt_joint_confidence_level_t sub2_confidence_level = secondary_body_vector[1][body_idx].skeleton.joints[i].confidence_level;
                                        if (outfile_sub2.is_open())
                                        {
                                            outfile_sub2 << secondary_body_vector[1][body_idx].id << "," << i << "," << sub2_position.v[0] << "," << sub2_position.v[1] << "," << sub2_position.v[2] << "," << sub2_orientation.v[0] << "," << sub2_orientation.v[1] << "," << sub2_orientation.v[2] << "," << sub2_orientation.v[3] << "," << confidenceMap(sub2_confidence_level) << "," << std::endl;
                                        }
                                    }
                                } // end for loop
                            }     // end (if sub device found bodies)
                        }         // end loop through min number of bodies
                    }             // end (if master device found bodies)
                    else
                    {
                        std::cerr << "Error! No bodies found in master device:  " << std::endl;
                        continue;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << e.what() << '\n';
                    continue;
                }
            } // END loop through subordinate mode devices

            // CHECK:
            if (main_num_bodies > 0)
            {
                std::cerr << "[DEBUG] master found body:\t\t\t\t" << main_num_bodies << " (master)" << std::endl;
            }
            else
            {
                std::cerr << "[DEBUG] master did NOT find any bodies!" << std::endl;
            }

            // std::cerr << "Secondary body vector size (at init): "<< secondary_body_vector.size() << std::endl;
            std::cerr << "[DEBUG] Valid secondary body devices(total: " << validSubDevices.size() << ")\t";
            for (std::vector<int>::iterator it = validSubDevices.begin(); it != validSubDevices.end(); it++)
                std::cerr << ' ' << *it;
            std::cerr << " (valid sub devices)\n";

            // move on to take average and project 3-D homogeneous points onto 2-D plane
            if (main_num_bodies > 0)
                processBodyData(main_body, secondary_body_vector, cv_main_color_image, main_intrinsic_matrix, validSubDevices);

            // // transformation check: print out positions from main, subordinate1, subordinate2
            // std::cerr << "TRANSFORMED: " << std::endl;
            // std::cerr << main_body.skeleton.joints[0].position.v[0] << " / " << main_body.skeleton.joints[0].position.v[1] << " / " << main_body.skeleton.joints[0].position.v[2] << std::endl;
            // for (int dev_count = 0; dev_count < secondary_body_vector.size(); dev_count++)
            // {
            //     for (int body_count = 0; body_count < secondary_body_vector[dev_count].size(); body_count++)
            //     {
            //         k4a_float3_t pos = secondary_body_vector[dev_count][body_count].skeleton.joints[0].position;
            //         std::cerr << pos.v[0] << " / " << pos.v[1] << " / " << pos.v[2] << std::endl;
            //     }
            // }

            // clear entries for the frame
            // std:cerr << "size of secondary body vector: " << secondary_body_vector.size() << std::endl;
            int clearcount = 0;
            for (auto it = secondary_body_vector.begin(); it != secondary_body_vector.end(); ++it)
            {
                (*it).clear();
                clearcount++;
            }
            std::cerr << "cleared:\t" << clearcount << std::endl;

        } // ========== END while loop
    }     // END main function
    else
    {
        std::cerr << "Not enough devices!" << std::endl;
        std::exit(1);
    }
    outfile_orig.close();
    outfile_sub1.close();
    outfile_sub2.close();

    return 0;
}

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

static cv::Matx33f calibration_to_color_camera_matrix(const k4a::calibration &cal)
{
    const k4a_calibration_intrinsic_parameters_t::_param &i = cal.color_camera_calibration.intrinsics.parameters.param;
    cv::Matx33f camera_matrix = cv::Matx33f::eye();
    camera_matrix(0, 0) = i.fx;
    camera_matrix(1, 1) = i.fy;
    camera_matrix(0, 2) = i.cx;
    camera_matrix(1, 2) = i.cy;
    return camera_matrix;
}

static Transformation get_depth_to_color_transformation_from_calibration(const k4a::calibration &cal)
{
    const k4a_calibration_extrinsics_t &ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    Transformation tr;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            tr.R(i, j) = ex.rotation[i * 3 + j];
        }
    }
    tr.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
    return tr;
}

// This function constructs a calibration that operates as a transformation between the secondary device's depth camera
// and the main camera's color camera. IT WILL NOT GENERALIZE TO OTHER TRANSFORMS. Under the hood, the transformation
// depth_image_to_color_camera method can be thought of as converting each depth pixel to a 3d point using the
// intrinsics of the depth camera, then using the calibration's extrinsics to convert between depth and color, then
// using the color intrinsics to produce the point in the color camera perspective.
static k4a::calibration construct_device_to_device_calibration(const k4a::calibration &main_cal,
                                                               const k4a::calibration &secondary_cal,
                                                               const Transformation &secondary_to_main)
{
    k4a::calibration cal = secondary_cal;
    k4a_calibration_extrinsics_t &ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            ex.rotation[i * 3 + j] = static_cast<float>(secondary_to_main.R(i, j));
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        ex.translation[i] = static_cast<float>(secondary_to_main.t[i]);
    }
    cal.color_camera_calibration = main_cal.color_camera_calibration;
    return cal;
}

static vector<float> calibration_to_color_camera_dist_coeffs(const k4a::calibration &cal)
{
    const k4a_calibration_intrinsic_parameters_t::_param &i = cal.color_camera_calibration.intrinsics.parameters.param;
    return {i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6};
}

bool find_chessboard_corners_helper(const cv::Mat &main_color_image,
                                    const cv::Mat &secondary_color_image,
                                    const cv::Size &chessboard_pattern,
                                    vector<cv::Point2f> &main_chessboard_corners,
                                    vector<cv::Point2f> &secondary_chessboard_corners)
{
    bool found_chessboard_main = cv::findChessboardCorners(main_color_image,
                                                           chessboard_pattern,
                                                           main_chessboard_corners);
    bool found_chessboard_secondary = cv::findChessboardCorners(secondary_color_image,
                                                                chessboard_pattern,
                                                                secondary_chessboard_corners);

    // Cover the failure cases where chessboards were not found in one or both images.
    if (!found_chessboard_main || !found_chessboard_secondary)
    {
        if (found_chessboard_main)
        {
            std::cout << "Could not find the chessboard corners in the secondary image. Trying again...\n";
        }
        // Likewise, if the chessboard was found in the secondary image, it was not found in the main image.
        else if (found_chessboard_secondary)
        {
            std::cout << "Could not find the chessboard corners in the main image. Trying again...\n";
        }
        // The only remaining case is the corners were in neither image.
        else
        {
            std::cout << "Could not find the chessboard corners in either image. Trying again...\n";
        }
        return false;
    }
    // Before we go on, there's a quick problem with calibration to address.  Because the chessboard looks the same when
    // rotated 180 degrees, it is possible that the chessboard corner finder may find the correct points, but in the
    // wrong order.

    // A visual:
    //        Image 1                  Image 2
    // .....................    .....................
    // .....................    .....................
    // .........xxxxx2......    .....xxxxx1..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........1xxxxx......    .....2xxxxx..........
    // .....................    .....................
    // .....................    .....................

    // The problem occurs when this case happens: the find_chessboard() function correctly identifies the points on the
    // chessboard (shown as 'x's) but the order of those points differs between images taken by the two cameras.
    // Specifically, the first point in the list of points found for the first image (1) is the *last* point in the list
    // of points found for the second image (2), though they correspond to the same physical point on the chessboard.

    // To avoid this problem, we can make the assumption that both of the cameras will be oriented in a similar manner
    // (e.g. turning one of the cameras upside down will break this assumption) and enforce that the vector between the
    // first and last points found in pixel space (which will be at opposite ends of the chessboard) are pointing the
    // same direction- so, the dot product of the two vectors is positive.

    cv::Vec2f main_image_corners_vec = main_chessboard_corners.back() - main_chessboard_corners.front();
    cv::Vec2f secondary_image_corners_vec = secondary_chessboard_corners.back() - secondary_chessboard_corners.front();
    if (main_image_corners_vec.dot(secondary_image_corners_vec) <= 0.0)
    {
        std::reverse(secondary_chessboard_corners.begin(), secondary_chessboard_corners.end());
    }
    return true;
}

Transformation stereo_calibration(const k4a::calibration &main_calib,
                                  const k4a::calibration &secondary_calib,
                                  const vector<vector<cv::Point2f>> &main_chessboard_corners_list,
                                  const vector<vector<cv::Point2f>> &secondary_chessboard_corners_list,
                                  const cv::Size &image_size,
                                  const cv::Size &chessboard_pattern,
                                  float chessboard_square_length)
{
    // We have points in each image that correspond to the corners that the findChessboardCorners function found.
    // However, we still need the points in 3 dimensions that these points correspond to. Because we are ultimately only
    // interested in find a transformation between two cameras, these points don't have to correspond to an external
    // "origin" point. The only important thing is that the relative distances between points are accurate. As a result,
    // we can simply make the first corresponding point (0, 0) and construct the remaining points based on that one. The
    // order of points inserted into the vector here matches the ordering of findChessboardCorners. The units of these
    // points are in millimeters, mostly because the depth provided by the depth cameras is also provided in
    // millimeters, which makes for easy comparison.
    vector<cv::Point3f> chessboard_corners_world;
    for (int h = 0; h < chessboard_pattern.height; ++h)
    {
        for (int w = 0; w < chessboard_pattern.width; ++w)
        {
            chessboard_corners_world.emplace_back(
                cv::Point3f{w * chessboard_square_length, h * chessboard_square_length, 0.0});
        }
    }

    // Calibrating the cameras requires a lot of data. OpenCV's stereoCalibrate function requires:
    // - a list of points in reatr_secondary_color_to_main_colorl 3d space that will be used to calibrate*
    // - a corresponding list of pixel coordinates as seen by the first camera*
    // - a corresponding list of pixel coordinates as seen by the second camera*
    // - the camera matrix of the first camera
    // - the distortion coefficients of the first camera
    // - the camera matrix of the second camera
    // - the distortion coefficients of the second camera
    // - the size (in pixels) of the images
    // - R: stereoCalibrate stores the rotation matrix from the first camera to the second here
    // - t: stereoCalibrate stores the translation vector from the first camera to the second here
    // - E: stereoCalibrate stores the essential matrix here (we don't use this)
    // - F: stereoCalibrate stores the fundamental matrix here (we don't use this)
    //
    // * note: OpenCV's stereoCalibrate actually requires as input an array of arrays of points for these arguments,
    // allowing a caller to provide multiple frames from the same camera with corresponding points. For example, if
    // extremely high precision was required, many images could be taken with each camera, and findChessboardCorners
    // applied to each of those images, and OpenCV can jointly solve for all of the pairs of corresponding images.
    // However, to keep things simple, we use only one image from each device to calibrate.  This is also why each of
    // the vectors of corners is placed into another vector.
    //
    // A function in OpenCV's calibration function also requires that these points be F32 types, so we use those.
    // However, OpenCV still provides doubles as output, strangely enough.
    vector<vector<cv::Point3f>> chessboard_corners_world_nested_for_cv(main_chessboard_corners_list.size(),
                                                                       chessboard_corners_world);

    cv::Matx33f main_camera_matrix = calibration_to_color_camera_matrix(main_calib);
    cv::Matx33f secondary_camera_matrix = calibration_to_color_camera_matrix(secondary_calib);
    vector<float> main_dist_coeff = calibration_to_color_camera_dist_coeffs(main_calib);
    vector<float> secondary_dist_coeff = calibration_to_color_camera_dist_coeffs(secondary_calib);

    // Finally, we'll actually calibrate the cameras.
    // Pass secondary first, then main, because we want a transform from secondary to main.
    Transformation tr;
    double error = cv::stereoCalibrate(chessboard_corners_world_nested_for_cv,
                                       secondary_chessboard_corners_list,
                                       main_chessboard_corners_list,
                                       secondary_camera_matrix,
                                       secondary_dist_coeff,
                                       main_camera_matrix,
                                       main_dist_coeff,
                                       image_size,
                                       tr.R, // output
                                       tr.t, // output
                                       cv::noArray(),
                                       cv::noArray(),
                                       cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL | cv::CALIB_CB_FAST_CHECK);
    std::cout << "Finished calibrating!\n";
    std::cout << "Calibration Error: " << error << "\n";
    return tr;
}

// The following functions provide the configurations that should be used for each camera.
// NOTE: For best results both cameras should have the same configuration (framerate, resolution, color and depth
// modes). Additionally the both master and subordinate should have the same exposure and power line settings. Exposure
// settings can be different but the subordinate must have a longer exposure from master. To synchronize a master and
// subordinate with different exposures the user should set `subordinate_delay_off_master_usec = ((subordinate exposure
// time) - (master exposure time))/2`.
//
static k4a_device_configuration_t get_default_config()
{
    k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    // WFOV, 15
    camera_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // No need for depth during calibration
    camera_config.camera_fps = K4A_FRAMES_PER_SECOND_30;     // Don't use all USB bandwidth
    camera_config.subordinate_delay_off_master_usec = 0;     // Must be zero for master
    camera_config.synchronized_images_only = true;
    return camera_config;
}

// Master customizable settings
static k4a_device_configuration_t get_master_config()
{
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

    // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
    // sensor doesn't interfere with the other. To accomplish this the master depth image captures
    // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
    // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
    // images centered around the color image as closely as possible.
    camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
    camera_config.synchronized_images_only = true;
    return camera_config;
}

// Subordinate customizable settings
static k4a_device_configuration_t get_subordinate_config()
{
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

    // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
    // sensor doesn't interfere with the other. To accomplish this the master depth image captures
    // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
    // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
    // images centered around the color image as closely as possible.
    camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
    return camera_config;
}

static Transformation calibrate_devices(MultiDeviceCapturer &capturer,
                                        const k4a_device_configuration_t &main_config,
                                        const k4a_device_configuration_t &secondary_config,
                                        const cv::Size &chessboard_pattern,
                                        float chessboard_square_length,
                                        double calibration_timeout)
{
    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,
                                                                                     main_config.color_resolution);

    k4a::calibration secondary_calibration =
        capturer.get_subordinate_device_by_index(0).get_calibration(secondary_config.depth_mode,
                                                                    secondary_config.color_resolution);
    vector<vector<cv::Point2f>> main_chessboard_corners_list;
    vector<vector<cv::Point2f>> secondary_chessboard_corners_list;
    std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
    while (std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count() < calibration_timeout)
    {
        vector<k4a::capture> captures = capturer.get_synchronized_captures(secondary_config);
        k4a::capture &main_capture = captures[0];
        k4a::capture &secondary_capture = captures[1];
        // get_color_image is guaranteed to be non-null because we use get_synchronized_captures for color
        // (get_synchronized_captures also offers a flag to use depth for the secondary camera instead of color).
        k4a::image main_color_image = main_capture.get_color_image();
        k4a::image secondary_color_image = secondary_capture.get_color_image();
        cv::Mat cv_main_color_image = color_to_opencv(main_color_image);
        cv::Mat cv_secondary_color_image = color_to_opencv(secondary_color_image);

        vector<cv::Point2f> main_chessboard_corners;
        vector<cv::Point2f> secondary_chessboard_corners;
        bool got_corners = find_chessboard_corners_helper(cv_main_color_image,
                                                          cv_secondary_color_image,
                                                          chessboard_pattern,
                                                          main_chessboard_corners,
                                                          secondary_chessboard_corners);
        if (got_corners)
        {
            main_chessboard_corners_list.emplace_back(main_chessboard_corners);
            secondary_chessboard_corners_list.emplace_back(secondary_chessboard_corners);
            cv::drawChessboardCorners(cv_main_color_image, chessboard_pattern, main_chessboard_corners, true);
            cv::drawChessboardCorners(cv_secondary_color_image, chessboard_pattern, secondary_chessboard_corners, true);
        }

        cv::namedWindow("Chessboard view from main", CV_WINDOW_AUTOSIZE);
        cv::imshow("Chessboard view from main", cv_main_color_image);
        cv::waitKey(1);

        // cv::namedWindow("Chessboard view from secondary", CV_WINDOW_AUTOSIZE);
        // cv::imshow("Chessboard view from secondary", cv_secondary_color_image);
        // cv::waitKey(1);

        // save image
        cv::imwrite("./calibration_master_color_image.jpg", cv_main_color_image);
        cv::imwrite("./calibration_sub_color_image.jpg", cv_secondary_color_image);

        // Get 20 frames before doing calibration.
        if (main_chessboard_corners_list.size() >= 20)
        {
            std::cout << "Calculating calibration..." << std::endl;
            return stereo_calibration(main_calibration,
                                      secondary_calibration,
                                      main_chessboard_corners_list,
                                      secondary_chessboard_corners_list,
                                      cv_main_color_image.size(),
                                      chessboard_pattern,
                                      chessboard_square_length);
        }
    }
    std::cerr << "Calibration timed out !\n ";
    std::exit(1);
}

static k4a::image create_depth_image_like(const k4a::image &im)
{
    return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
                              im.get_width_pixels(),
                              im.get_height_pixels(),
                              im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

void processBodyData(k4abt_body_t main_body, vector<vector<k4abt_body_t>> secondary_body_vector, cv::Mat &main, cv::Matx33f main_intrinsic_matrix, vector<int> validSubDevices)
{

    // take joint stream with higher confidence interval among master and subordinate devices

    // find minimum number of bodies to scan through (number of overlapping bodies across devices)
    int min_nbody = 1; // for now, restrict this to 1 body
    // int min_nbody = 1000;
    // for (int sdev=0; sdev < num_subordinates; sdev++)
    // {
    //     if (min_nbody > secondary_body_vector[sdev].size())
    //     {
    //         min_nbody = secondary_body_vector[sdev].size();
    //     }
    // }

    // reformat the secondary body vector into valid secondary body vector
    std::vector<vector<k4abt_body_t>> validSubBodyVector((int)validSubDevices.size(), std::vector<k4abt_body_t>());
    int idx = 0;
    for (std::vector<int>::iterator it = validSubDevices.begin(); it != validSubDevices.end(); ++it)
    {
        std::cerr << "adding body for devices: " << *it << std::endl;
        validSubBodyVector[idx] = std::move(secondary_body_vector[*it]);
        idx++;
    }

    std::cerr << "moving to validSubBodyVector SUCCESS: size = " << idx << endl;

    int totalValidSubCount = validSubDevices.size();
    for (int num_sdev = 0; num_sdev < totalValidSubCount; num_sdev++)
    {
        std::cerr << "sample position at joint 0, x, body 0 at each device[" << num_sdev << "]\t" << validSubBodyVector.at(num_sdev)[0].skeleton.joints[0].position.v[0] << "\t";
    }
    std::cerr << std::endl;

    // initialize structures
    vector<k4abt_body_t> body_dev(validSubDevices.size() + 1); // add 1 for master
    body_dev.clear();

    // append master device body at 0
    body_dev.push_back(main_body);

    k4abt_body_t avg_body;
    k4a_float3_t avg_pos;
    k4a_quaternion_t avg_orient;
    k4abt_joint_confidence_level_t avg_confidence;
    std::vector<std::vector<int>> highest_conf_dev(K4ABT_JOINT_COUNT);

    // iterate through bodies
    for (int bodyid = 0; bodyid < min_nbody; bodyid++)
    {
        avg_body.id = bodyid;

        // reset
        for (auto it = highest_conf_dev.begin(); it != highest_conf_dev.end(); ++it)
        {
            (*it).clear();
        }

        // append sub device bodies
        for (int num_sdev = 0; num_sdev < totalValidSubCount; num_sdev++)
        {
            body_dev.push_back(validSubBodyVector[num_sdev][bodyid]);
        }

        // sort master and sub device bodies by confidence levels: MEDIUM --> LOW --> EVERYTHING ELSE
        for (int curr_dev = 0; curr_dev < (int)body_dev.size(); curr_dev++)
        {
            for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
            {
                if (body_dev[curr_dev].skeleton.joints[joint].confidence_level == K4ABT_JOINT_CONFIDENCE_MEDIUM)
                {
                    highest_conf_dev[joint].push_back(curr_dev);
                }
            }
        }
        // get device numbers that have LOW confidence level for each joint, if the joint is not found with MEDIUM confidence level
        for (int curr_dev = 0; curr_dev < (int)body_dev.size(); curr_dev++)
        {
            for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
            {
                if (body_dev[curr_dev].skeleton.joints[joint].confidence_level == K4ABT_JOINT_CONFIDENCE_LOW && highest_conf_dev[joint].empty())
                {
                    highest_conf_dev[joint].push_back(curr_dev);
                }
            }
        }
        // if empty (nor MEDIUM or LOW confident), then put all devices in
        for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
        {
            if (highest_conf_dev[joint].size() == 0)
            {
                for (int num_dev = 1; num_dev < totalValidSubCount; num_dev++)
                {
                    highest_conf_dev[joint].push_back(num_dev);
                }
            }
        }

        // check which devices each joint is pulling data from
        int jointCount = 0;
        for (auto &row : highest_conf_dev)
        {
            jointCount++;
            std::cerr << "[DEBUG] joint: " << jointCount << " / "
                      << "confident device: ";
            for (std::vector<int>::iterator it = row.begin(); it != row.end(); ++it)
            {
                std::cerr << *it << " ";
            }
        std:
            cerr << std::endl;
        }

        // pull data / average data from the found devices
        // initialize average body object to store(replace joint data)

        int joint_count = 0;
        k4a_float3_t conf_joint_pos;
        k4a_quaternion_t conf_joint_quat;

        // iterate through joints
        for (auto &row : highest_conf_dev)
        {
            // reset position for all subs
            avg_pos.xyz.x = 0;
            avg_pos.xyz.y = 0;
            avg_pos.xyz.z = 0;
            avg_pos.v[0] = 0;
            avg_pos.v[1] = 0;
            avg_pos.v[2] = 0;
            // reset quaternion for all subs
            avg_orient.wxyz.w = 0;
            avg_orient.wxyz.x = 0;
            avg_orient.wxyz.y = 0;
            avg_orient.wxyz.z = 0;
            avg_orient.v[0] = 0;
            avg_orient.v[1] = 0;
            avg_orient.v[2] = 0;
            avg_orient.v[3] = 0;
            if (row.size() > 1) // if there is at least one subordinate device that is confident
            {
                std::cerr << "# of confident sub devices for joint(" << joint_count + 1 << "): "; // (" << highest_conf_dev[joint_count].size()<<" dev total)";
                for (std::vector<int>::iterator it = row.begin(); it != row.end(); ++it)
                {
                    std::cerr << "\t" << *it << " ";
                }
                std::cerr << std::endl;
                // reset confidence level for all sub
                avg_confidence = K4ABT_JOINT_CONFIDENCE_LOW;
                int rowsize = 0;
                int iter = 0;
                // iterate through all "confident" devices
                for (int idx = 0; idx < row.size(); idx++)
                {
                    iter = row.at(idx);
                    std::cerr << "going through device:\t" << iter << std::endl;

                    // set main body to be initial for avg body coordinates
                    if (iter == 0) // master
                    {
                        avg_pos.xyz.x += main_body.skeleton.joints[joint_count].position.xyz.x;
                        avg_pos.xyz.y += main_body.skeleton.joints[joint_count].position.xyz.y;
                        avg_pos.xyz.z += main_body.skeleton.joints[joint_count].position.xyz.z;

                        avg_orient.wxyz.w += main_body.skeleton.joints[joint_count].orientation.wxyz.w;
                        avg_orient.wxyz.x += main_body.skeleton.joints[joint_count].orientation.wxyz.x;
                        avg_orient.wxyz.y += main_body.skeleton.joints[joint_count].orientation.wxyz.y;
                        avg_orient.wxyz.z += main_body.skeleton.joints[joint_count].orientation.wxyz.z;

                        if (avg_confidence < main_body.skeleton.joints[joint_count].confidence_level)
                        {
                            avg_confidence = main_body.skeleton.joints[joint_count].confidence_level;
                        };
                        rowsize++;
                        std::cerr << "master joint data added" << std::endl;
                    }
                    else if (iter > 0)
                    {

                        conf_joint_pos = validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].position;
                        avg_pos.xyz.x += conf_joint_pos.xyz.x;
                        avg_pos.xyz.y += conf_joint_pos.xyz.y;
                        avg_pos.xyz.z += conf_joint_pos.xyz.z;

                        conf_joint_quat = validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].orientation;
                        avg_orient.wxyz.w += conf_joint_quat.wxyz.w;
                        avg_orient.wxyz.x += conf_joint_quat.wxyz.x;
                        avg_orient.wxyz.y += conf_joint_quat.wxyz.y;
                        avg_orient.wxyz.z += conf_joint_quat.wxyz.z;

                        // confidence level
                        k4abt_joint_confidence_level_t secondary_dev_conf_level = validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].confidence_level;
                        if (avg_confidence < secondary_dev_conf_level)
                        {
                            avg_confidence = secondary_dev_conf_level;
                        };
                        rowsize++;
                        std::cerr << "joint data added for up to " << iter << " subordinate devices" << std::endl;
                    }
                    //  verify main, sub 0, sub 1
                    std::cerr << "Master: (" << main_body.skeleton.joints[joint_count].position.xyz.x << ", " << main_body.skeleton.joints[joint_count].position.xyz.y << ", " << main_body.skeleton.joints[joint_count].position.xyz.z << ")" << endl;
                    if (iter > 0) // print for subordinate devices
                    {
                        std::cerr << "Sub[" << iter << "]: (" << validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].position.xyz.x << ", " << validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].position.xyz.y << ", " << validSubBodyVector[iter - 1][bodyid].skeleton.joints[joint_count].position.xyz.z << ")" << std::endl;
                    }
                }

                // divide position, quaternion by the number of all devices added for average
                std::cerr << "averaging over (how many devices):\t" << rowsize << std::endl;

                avg_pos.xyz.x = avg_pos.v[0] = avg_pos.xyz.x / rowsize;
                avg_pos.xyz.y = avg_pos.v[1] = avg_pos.xyz.y / rowsize;
                avg_pos.xyz.z = avg_pos.v[2] = avg_pos.xyz.z / rowsize;

                avg_orient.wxyz.w = avg_orient.v[0] = avg_orient.wxyz.w / rowsize;
                avg_orient.wxyz.x = avg_orient.v[1] = avg_orient.wxyz.x / rowsize;
                avg_orient.wxyz.y = avg_orient.v[2] = avg_orient.wxyz.y / rowsize;
                avg_orient.wxyz.z = avg_orient.v[3] = avg_orient.wxyz.z / rowsize;

                // create a joint object with the averaged position and quaternion
                // update body: keep first device's body ID, update each joint data

                avg_body.skeleton.joints[joint_count].position = avg_pos;
                avg_body.skeleton.joints[joint_count].orientation = avg_orient;
                avg_body.skeleton.joints[joint_count].confidence_level = avg_confidence;

                joint_count++;
                std::cerr << "AVG for joint[" << joint_count << "]: (" << avg_body.skeleton.joints[joint_count].position.xyz.x << ", " << avg_body.skeleton.joints[joint_count].position.xyz.y << ", " << avg_body.skeleton.joints[joint_count].position.xyz.z << "), confidence: " << avg_body.skeleton.joints[joint_count].confidence_level << std::endl;

                std::cerr << std::endl
                          << std::endl;
            } // end joints if there is at least one subordinate device
            else if (row.size() == 1)
            {
                std::cerr << "[NO SUB DETECTED] going through device:\t" << 0 << std::endl;
                // if only master is present
                avg_pos.xyz.x += main_body.skeleton.joints[joint_count].position.xyz.x;
                avg_pos.xyz.y += main_body.skeleton.joints[joint_count].position.xyz.y;
                avg_pos.xyz.z += main_body.skeleton.joints[joint_count].position.xyz.z;

                avg_orient.wxyz.w += main_body.skeleton.joints[joint_count].orientation.wxyz.w;
                avg_orient.wxyz.x += main_body.skeleton.joints[joint_count].orientation.wxyz.x;
                avg_orient.wxyz.y += main_body.skeleton.joints[joint_count].orientation.wxyz.y;
                avg_orient.wxyz.z += main_body.skeleton.joints[joint_count].orientation.wxyz.z;

                avg_confidence = main_body.skeleton.joints[joint_count].confidence_level;
                if (avg_confidence < 0 || avg_confidence > 5)
                    avg_confidence = K4ABT_JOINT_CONFIDENCE_LOW; // exceptional cases

                avg_body.skeleton.joints[joint_count].position = avg_pos;
                avg_body.skeleton.joints[joint_count].orientation = avg_orient;
                avg_body.skeleton.joints[joint_count].confidence_level = avg_confidence;

                joint_count++;
                std::cerr << "Master: [" << joint_count << "]: (" << avg_body.skeleton.joints[joint_count].position.xyz.x << ", " << avg_body.skeleton.joints[joint_count].position.xyz.y << ", " << avg_body.skeleton.joints[joint_count].position.xyz.z << "), confidence: " << avg_body.skeleton.joints[joint_count].confidence_level << std::endl;

                std::cerr << std::endl
                          << std::endl;
            }
        }

        // store averaged data information
        ofstream outfile_avg;
        outfile_avg.open("../saved_data/joints_gen_sync.csv", ios::out | ios::app);
        for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
        {
            k4a_float3_t avg_position = avg_body.skeleton.joints[i].position;
            k4a_quaternion_t avg_orientation = avg_body.skeleton.joints[i].orientation;
            k4abt_joint_confidence_level_t avg_confidence_level = avg_body.skeleton.joints[i].confidence_level;

            if (outfile_avg.is_open())
            {
                outfile_avg << avg_body.id << "," << i << "," << avg_position.v[0] << "," << avg_position.v[1] << "," << avg_position.v[2] << "," << avg_orientation.v[0] << "," << avg_orientation.v[1] << "," << avg_orientation.v[2] << "," << avg_orientation.v[3] << "," << confidenceMap(avg_confidence_level) << "," << std::endl;
            }
        }
        outfile_avg.close();
        plotBody(main_body, avg_body, main, main_intrinsic_matrix);
    } // end this body ID
    std::cerr << "finished going through all master and subordinate devices" << std::endl;
}

/*
    given two 3-D vectors, computes the internal joint angle
 */
void getJointAngle(float &ans, cv::Vec3f &p1, cv::Vec3f &p2)
{
    cv::Vec3f fcross;
    // cross product of p1 and p2
    fcross[0] = p1[1] * p2[2] - p1[2] * p2[1];
    fcross[1] = p1[2] * p2[0] - p1[0] * p2[2];
    fcross[2] = p1[0] * p2[1] - p1[1] * p2[0];
    // take norm of cross product: sqrt(fcrossx * fcrossx + fcrossy * fcrossy + fcrossz * fcrossz);
    float fcross_norm = norm(fcross);
    // take dot product: p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2];
    float fdot = p1.dot(p2);
    // take atan2 so it falls within -pi/2 and pi/2 radians (+pi if in 2nd/4th quadrant (negative))
    ans = atan2(fcross_norm, fdot);
}

/*
    given a body structure, computes the angles A-L;
 */
std::vector<float> computeJointAngles(k4abt_body_t avg_body)
{
    // plot angles from 3D positions
    // A: 12-13-14
    cv::Vec3f a1(avg_body.skeleton.joints[13].position.v[0] - avg_body.skeleton.joints[12].position.v[0],
                 avg_body.skeleton.joints[13].position.v[1] - avg_body.skeleton.joints[12].position.v[1],
                 avg_body.skeleton.joints[13].position.v[2] - avg_body.skeleton.joints[12].position.v[2]);
    cv::Vec3f a2(avg_body.skeleton.joints[13].position.v[0] - avg_body.skeleton.joints[14].position.v[0],
                 avg_body.skeleton.joints[13].position.v[1] - avg_body.skeleton.joints[14].position.v[1],
                 avg_body.skeleton.joints[13].position.v[2] - avg_body.skeleton.joints[14].position.v[2]);

    float A;
    getJointAngle(A, a1, a2);

    // B: 5-6-7
    cv::Vec3f b1(avg_body.skeleton.joints[6].position.v[0] - avg_body.skeleton.joints[5].position.v[0],
                 avg_body.skeleton.joints[6].position.v[1] - avg_body.skeleton.joints[5].position.v[1],
                 avg_body.skeleton.joints[6].position.v[2] - avg_body.skeleton.joints[5].position.v[2]);
    cv::Vec3f b2(avg_body.skeleton.joints[6].position.v[0] - avg_body.skeleton.joints[7].position.v[0],
                 avg_body.skeleton.joints[6].position.v[1] - avg_body.skeleton.joints[7].position.v[1],
                 avg_body.skeleton.joints[6].position.v[2] - avg_body.skeleton.joints[7].position.v[2]);

    float B;
    getJointAngle(B, b1, b2);

    // C: 11-12-13
    cv::Vec3f c1(avg_body.skeleton.joints[12].position.v[0] - avg_body.skeleton.joints[11].position.v[0],
                 avg_body.skeleton.joints[12].position.v[1] - avg_body.skeleton.joints[11].position.v[1],
                 avg_body.skeleton.joints[12].position.v[2] - avg_body.skeleton.joints[11].position.v[2]);
    cv::Vec3f c2(avg_body.skeleton.joints[12].position.v[0] - avg_body.skeleton.joints[13].position.v[0],
                 avg_body.skeleton.joints[12].position.v[1] - avg_body.skeleton.joints[13].position.v[1],
                 avg_body.skeleton.joints[12].position.v[2] - avg_body.skeleton.joints[13].position.v[2]);

    float C;
    getJointAngle(C, c1, c2);

    // D: 4-5-6
    cv::Vec3f d1(avg_body.skeleton.joints[5].position.v[0] - avg_body.skeleton.joints[4].position.v[0],
                 avg_body.skeleton.joints[5].position.v[1] - avg_body.skeleton.joints[4].position.v[1],
                 avg_body.skeleton.joints[5].position.v[2] - avg_body.skeleton.joints[4].position.v[2]);
    cv::Vec3f d2(avg_body.skeleton.joints[5].position.v[0] - avg_body.skeleton.joints[6].position.v[0],
                 avg_body.skeleton.joints[5].position.v[1] - avg_body.skeleton.joints[6].position.v[1],
                 avg_body.skeleton.joints[5].position.v[2] - avg_body.skeleton.joints[6].position.v[2]);

    float D;
    getJointAngle(D, d1, d2);

    // E: 1-0-22
    cv::Vec3f e1(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[1].position.v[0],
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[1].position.v[1],
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[1].position.v[2]);
    cv::Vec3f e2(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[22].position.v[0],
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[22].position.v[1],
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[22].position.v[2]);

    float E;
    getJointAngle(E, e1, e2);

    // F: 1-0-18
    cv::Vec3f f1(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[1].position.v[0],
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[1].position.v[1],
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[1].position.v[2]);
    cv::Vec3f f2(avg_body.skeleton.joints[0].position.v[0] - avg_body.skeleton.joints[18].position.v[0],
                 avg_body.skeleton.joints[0].position.v[1] - avg_body.skeleton.joints[18].position.v[1],
                 avg_body.skeleton.joints[0].position.v[2] - avg_body.skeleton.joints[18].position.v[2]);

    float F;
    getJointAngle(F, f1, f2);

    // G: 0-22-23
    cv::Vec3f g1(avg_body.skeleton.joints[22].position.v[0] - avg_body.skeleton.joints[0].position.v[0],
                 avg_body.skeleton.joints[22].position.v[1] - avg_body.skeleton.joints[0].position.v[1],
                 avg_body.skeleton.joints[22].position.v[2] - avg_body.skeleton.joints[0].position.v[2]);
    cv::Vec3f g2(avg_body.skeleton.joints[22].position.v[0] - avg_body.skeleton.joints[23].position.v[0],
                 avg_body.skeleton.joints[22].position.v[1] - avg_body.skeleton.joints[23].position.v[1],
                 avg_body.skeleton.joints[22].position.v[2] - avg_body.skeleton.joints[23].position.v[2]);

    float G;
    getJointAngle(G, g1, g2);

    // H: 0-18-19
    cv::Vec3f h1(avg_body.skeleton.joints[18].position.v[0] - avg_body.skeleton.joints[0].position.v[0],
                 avg_body.skeleton.joints[18].position.v[1] - avg_body.skeleton.joints[0].position.v[1],
                 avg_body.skeleton.joints[18].position.v[2] - avg_body.skeleton.joints[0].position.v[2]);
    cv::Vec3f h2(avg_body.skeleton.joints[18].position.v[0] - avg_body.skeleton.joints[19].position.v[0],
                 avg_body.skeleton.joints[18].position.v[1] - avg_body.skeleton.joints[19].position.v[1],
                 avg_body.skeleton.joints[18].position.v[2] - avg_body.skeleton.joints[19].position.v[2]);

    float H;
    getJointAngle(H, h1, h2);

    // I: 22-23-24
    cv::Vec3f i1(avg_body.skeleton.joints[23].position.v[0] - avg_body.skeleton.joints[22].position.v[0],
                 avg_body.skeleton.joints[23].position.v[1] - avg_body.skeleton.joints[22].position.v[1],
                 avg_body.skeleton.joints[23].position.v[2] - avg_body.skeleton.joints[22].position.v[2]);
    cv::Vec3f i2(avg_body.skeleton.joints[23].position.v[0] - avg_body.skeleton.joints[24].position.v[0],
                 avg_body.skeleton.joints[23].position.v[1] - avg_body.skeleton.joints[24].position.v[1],
                 avg_body.skeleton.joints[23].position.v[2] - avg_body.skeleton.joints[24].position.v[2]);

    float I;
    getJointAngle(I, i1, i2);

    // J: 18-19-20
    cv::Vec3f j1(avg_body.skeleton.joints[19].position.v[0] - avg_body.skeleton.joints[18].position.v[0],
                 avg_body.skeleton.joints[19].position.v[1] - avg_body.skeleton.joints[18].position.v[1],
                 avg_body.skeleton.joints[19].position.v[2] - avg_body.skeleton.joints[18].position.v[2]);
    cv::Vec3f j2(avg_body.skeleton.joints[19].position.v[0] - avg_body.skeleton.joints[20].position.v[0],
                 avg_body.skeleton.joints[19].position.v[1] - avg_body.skeleton.joints[20].position.v[1],
                 avg_body.skeleton.joints[19].position.v[2] - avg_body.skeleton.joints[20].position.v[2]);

    float J;
    getJointAngle(J, j1, j2);

    // K: 23-24-25
    cv::Vec3f k1(avg_body.skeleton.joints[24].position.v[0] - avg_body.skeleton.joints[23].position.v[0],
                 avg_body.skeleton.joints[24].position.v[1] - avg_body.skeleton.joints[23].position.v[1],
                 avg_body.skeleton.joints[24].position.v[2] - avg_body.skeleton.joints[23].position.v[2]);
    cv::Vec3f k2(avg_body.skeleton.joints[24].position.v[0] - avg_body.skeleton.joints[25].position.v[0],
                 avg_body.skeleton.joints[24].position.v[1] - avg_body.skeleton.joints[25].position.v[1],
                 avg_body.skeleton.joints[24].position.v[2] - avg_body.skeleton.joints[25].position.v[2]);

    float K;
    getJointAngle(K, k1, k2);

    // L: 19-20-21
    cv::Vec3f l1(avg_body.skeleton.joints[20].position.v[0] - avg_body.skeleton.joints[19].position.v[0],
                 avg_body.skeleton.joints[20].position.v[1] - avg_body.skeleton.joints[19].position.v[1],
                 avg_body.skeleton.joints[20].position.v[2] - avg_body.skeleton.joints[19].position.v[2]);
    cv::Vec3f l2(avg_body.skeleton.joints[20].position.v[0] - avg_body.skeleton.joints[21].position.v[0],
                 avg_body.skeleton.joints[20].position.v[1] - avg_body.skeleton.joints[21].position.v[1],
                 avg_body.skeleton.joints[20].position.v[2] - avg_body.skeleton.joints[21].position.v[2]);

    float L;
    getJointAngle(L, l1, l2);

    vector<float> angles = {A, B, C, D, E, F, G, H, I, J, K, L};
    // verify in terminal output
    for (int ii = 0; ii < angles.size(); ++ii)
    {
        std::cerr << "Angle [" << ii << "]: " << angles[ii] * 180 / M_PI << " (deg)" << std::endl;
    }
    std::cerr << "finished computing joint angles" << std::endl;
    return angles;
}

// TODO: calculate joint angles from quaternions
std::vector<float> computeJointAnglesQuat(cv::Mat main, k4abt_body_t avg_body)
{
    std::vector<float> angles;
    // A: 12-13-14
    // B: 5-6-7
    // C: 11-12-13
    // D: 4-5-6
    // E: 1-0-222
    // F: 1-0-18
    // G: 0-22-23
    // H: 0-18-19
    // I: 22-23-24
    // J: 18-19-20
    // K: 23-24-25
    // L: 19-20-21
    return angles;
}

/*
    projects 3-D to 2-D and plots the master and averaged bodies
 */
void plotBody(k4abt_body_t main_body, k4abt_body_t avg_body, cv::Mat main, cv::Matx33f main_intrinsic_matrix)
{
    std::vector<cv::Point> dataMain, dataSecondary, dataAvg;
    Mat mainstream(3, K4ABT_JOINT_COUNT, CV_32F), avgstream(3, K4ABT_JOINT_COUNT, CV_32F);
    std::vector<cv::Point3f> main_points, avg_points; // input to projectPoints

    // joint angles START
    vector<float> joint_angles = computeJointAngles(avg_body);
    // vector<float> joint_angles_quat = computeJointAnglesQuat(main, avg_body);

    int offset = 0;
    int index = 0;
    string angle;
    ofstream outfile_angles;
    outfile_angles.open("../saved_data/joints_gen_angles.csv", ios::out | ios::app);
    outfile_angles << to_string(ANGLE_FRAME_ROW_COUNT) << ",";
    for (std::vector<float>::iterator it = joint_angles.begin(); it != joint_angles.end(); it++)
    {
        outfile_angles << *it * 180 / M_PI << ",";

        // display the angles
        angle = (char)(index + 65);
        cv::putText(main, angle + ": " + to_string(*it * 180 / M_PI), cv::Point(main.cols - 200, 30 + offset), FONT_HERSHEY_DUPLEX, 1, COLORS_black, 1);

        offset += 30;
        index++;
    }
    ANGLE_FRAME_ROW_COUNT++;
    outfile_angles << std::endl;
    outfile_angles.close();
    // joint angles END

    // projection START
    for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
    {
        // for R,T calculation (for 2-D projection from 3-D)
        mainstream.at<float>(0, joint) = main_body.skeleton.joints[joint].position.v[0];
        mainstream.at<float>(1, joint) = main_body.skeleton.joints[joint].position.v[1];
        mainstream.at<float>(2, joint) = main_body.skeleton.joints[joint].position.v[2];
        avgstream.at<float>(0, joint) = avg_body.skeleton.joints[joint].position.v[0];
        avgstream.at<float>(1, joint) = avg_body.skeleton.joints[joint].position.v[1];
        avgstream.at<float>(2, joint) = avg_body.skeleton.joints[joint].position.v[2];

        // points to project from (3-D)
        main_points.push_back(cv::Point3f(main_body.skeleton.joints[joint].position.v[0], main_body.skeleton.joints[joint].position.v[1], main_body.skeleton.joints[joint].position.v[2]));
        avg_points.push_back(cv::Point3d(avg_body.skeleton.joints[joint].position.v[0], avg_body.skeleton.joints[joint].position.v[1], avg_body.skeleton.joints[joint].position.v[2]));
    }

    // Compute R, T from secondary to main coordinate space
    Mat R;
    Mat T;
    arun(avgstream, mainstream, R, T); // R: [3x3], T: [1x3]

    // Create zero distortion
    cv::Mat distCoeffs(4, 1, CV_32F);
    distCoeffs.at<float>(0) = 0;
    distCoeffs.at<float>(1) = 0;
    distCoeffs.at<float>(2) = 0;
    distCoeffs.at<float>(3) = 0;

    std::vector<cv::Point2f> projectedPointsMain, projectedPointsAvg;
    cv::Mat rvecR(3, 1, cv::DataType<double>::type); //rodrigues rotation matrix [1x3]
    cv::Rodrigues(R, rvecR);

    // project 3-D coordinates onto 2-D plane
    std::cerr << "projecting 3-D to 2-D..." << std::endl;
    cv::projectPoints(main_points, rvecR, T, Mat(main_intrinsic_matrix), distCoeffs, projectedPointsMain);
    cv::projectPoints(avg_points, rvecR, T, Mat(main_intrinsic_matrix), distCoeffs, projectedPointsAvg);
    // projection END

    // plotting joints START
    std::cerr << "plotting joints..." << std::endl;
    for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
    {
        // printf("[Synced] Joint[%d]: Position[mm] ( %f, %f, %f ); Orientation ( %f, %f, %f, %f); Confidence Level (%d)  \n", i, avgPos.v[0], avgPos.v[1], avgPos.v[2], main_orientation.v[0], avgQuaternion.v[1], avgQuaternion.v[2], avgQuaternion.v[3], avgCI);

        // ============ Display both joint streams in one RGB image ==========
        int offsetx = 0;
        int offsety = 0;
        // plot 2D points for main camera
        int radius = 3;
        // cv::Point center1 = cv::Point(main_position.xyz.x, main_position.xyz.y);
        cv::Point center1 = projectedPointsMain[i];
        center1 = cv::Point(center1.x + offsetx, center1.y + offsety);
        dataMain.push_back(center1);
        cv::Scalar color = COLORS_red;
        cv::circle(main, center1, radius, color, CV_FILLED);
        cv::putText(main, to_string(i), center1, FONT_HERSHEY_DUPLEX, 1, Scalar(0, 143, 143), 2);

        // cv::namedWindow("cv_main_color_image", CV_WINDOW_AUTOSIZE);
        // cv::imshow("cv_main_color_image", main);
        // cv::waitKey(1);

        // plot 2D points for averaged
        radius = 5;
        cv::Point center0 = projectedPointsAvg[i];
        center0 = cv::Point(center0.x + offsetx, center0.y + offsety);
        dataAvg.push_back(center0);
        color = COLORS_green;
        cv::circle(main, center0, radius, color, CV_FILLED);
        cv::putText(main, to_string(i), center0, FONT_HERSHEY_DUPLEX, 1, Scalar(0, 143, 143), 2);

        // cv::namedWindow("cv_main_color_image", CV_WINDOW_AUTOSIZE);
        // cv::imshow("cv_main_color_image", main);
        // cv::waitKey(1);
    }

    // draws lines between joints for both main and secondary streams
    std::cerr << "connecting joints with lines..." << std::endl;
    cv::Scalar color;
    int counter = 0, thickness = 0;
    std::list<std::vector<cv::Point>> datalist{dataMain, dataAvg};
    std::vector<cv::Mat> imgList{main};
    for (auto stream : datalist)
    {
        switch (counter)
        {
        case 0: // main
            color = COLORS_red;
            thickness = 5;
            // std::cerr<<std::endl<<"plotting for master"<<std::endl;
            break;
        case 1: // avg
            color = COLORS_green;
            thickness = 3;
            // std::cerr<<std::endl<<"plotting for averaged"<<std::endl;
            break;
        }

        // [child]: child joint to parent joint
        // imgList: main, secondary views
        // stream: joint positions

        // connect by adjacent joints
        // 1: spine naval to pelvis
        cv::line(imgList.at(0), stream[1], stream[0], color, thickness, 8); // 8: cv::line
        // 2: spine chest to spine naval
        cv::line(imgList.at(0), stream[2], stream[1], color, thickness, 8);
        // 3: neck to spine chest
        cv::line(imgList.at(0), stream[3], stream[2], color, thickness, 8);
        // 4: clavicle left to spine chest
        cv::line(imgList.at(0), stream[4], stream[2], color, thickness, 8);
        // 5: shoulder left to clavicle left
        cv::line(imgList.at(0), stream[5], stream[4], color, thickness, 8);
        // 6: elbow left to shoulder left
        cv::line(imgList.at(0), stream[6], stream[5], color, thickness, 8);
        // 7: wrist left to elbow left
        cv::line(imgList.at(0), stream[7], stream[6], color, thickness, 8);
        // 8: hand left to wrist left
        cv::line(imgList.at(0), stream[8], stream[7], color, thickness, 8);
        // 9: handtip left to hand left
        cv::line(imgList.at(0), stream[9], stream[8], color, thickness, 8);
        // 10: thumb left to writst left
        cv::line(imgList.at(0), stream[10], stream[7], color, thickness, 8);
        // 11: clavicle right to spine chest
        cv::line(imgList.at(0), stream[11], stream[2], color, thickness, 8);
        // 12: shoulder right to clavicle right
        cv::line(imgList.at(0), stream[12], stream[11], color, thickness, 8);
        // 13: elbow rith to shoulder right
        cv::line(imgList.at(0), stream[13], stream[12], color, thickness, 8);
        // 14: wrist right to elbow right
        cv::line(imgList.at(0), stream[14], stream[13], color, thickness, 8);
        // 15: hand right to wrist right
        cv::line(imgList.at(0), stream[15], stream[14], color, thickness, 8);
        // 16: handtip right to hand right
        cv::line(imgList.at(0), stream[16], stream[15], color, thickness, 8);
        // 17: thumb right to writst right
        cv::line(imgList.at(0), stream[17], stream[14], color, thickness, 8);
        // 18: hip left to pelvis
        cv::line(imgList.at(0), stream[18], stream[0], color, thickness, 8);
        // 19: knee left to hip left
        cv::line(imgList.at(0), stream[19], stream[18], color, thickness, 8);
        // 20: ankle left to knee left
        cv::line(imgList.at(0), stream[20], stream[19], color, thickness, 8);
        // 21: foot left to ankle left
        cv::line(imgList.at(0), stream[21], stream[20], color, thickness, 8);
        // 22: hip right to plevis
        cv::line(imgList.at(0), stream[22], stream[0], color, thickness, 8);
        // 23: knee right to hip right
        cv::line(imgList.at(0), stream[23], stream[22], color, thickness, 8);
        // 24: ankle right to hip right
        cv::line(imgList.at(0), stream[24], stream[23], color, thickness, 8);
        // 25: foot right to ankle right
        cv::line(imgList.at(0), stream[25], stream[24], color, thickness, 8);
        // 26: head to neck
        cv::line(imgList.at(0), stream[26], stream[3], color, thickness, 8);
        // 27: nose to head
        cv::line(imgList.at(0), stream[27], stream[26], color, thickness, 8);
        // 28: eye left to head
        cv::line(imgList.at(0), stream[28], stream[26], color, thickness, 8);
        // 29: ear left to head
        cv::line(imgList.at(0), stream[29], stream[26], color, thickness, 8);
        // 30: eye right to head
        cv::line(imgList.at(0), stream[30], stream[26], color, thickness, 8);
        // 31: ear right to head
        cv::line(imgList.at(0), stream[31], stream[26], color, thickness, 8);

        counter += 1;
    }
    // plotting joints END

    // display
    cv::namedWindow("main color camera", CV_WINDOW_AUTOSIZE);
    cv::putText(imgList.at(0), "[master device] red", cv::Point(30, 30), CV_FONT_HERSHEY_PLAIN, 1.5, COLORS_red, 2, 8, false);
    cv::putText(imgList.at(0), "[synced] green", cv::Point(30, 70), CV_FONT_HERSHEY_PLAIN, 1.5, COLORS_green, 2, 8, false);
    cv::imshow("main color camera", imgList.at(0));
    cv::waitKey(1);
}

/*
    transforms one body stream to another
 */
void transformBody(k4abt_body_t &main_body, k4abt_body_t &secondary_body)
{
    // called per frame, for data stream containing body objects with 32 positions, orientations
    // transform secondary to main body space coordinates
    // using Arun's method for computing R, T matrices from positions (for 3dof)

    // main, secondary: [3 x #_joints]
    Mat main(3, K4ABT_JOINT_COUNT, CV_32F), secondary(3, K4ABT_JOINT_COUNT, CV_32F);
    for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
    {
        // // print current positions
        // std::cout << "main x,y,z: " << main_body.skeleton.joints[joint].position.xyz.x << "\t" << main_body.skeleton.joints[joint].position.xyz.y << "\t" << main_body.skeleton.joints[joint].position.xyz.z << std::endl;
        // std::cout << "secondary x,y,z: " << secondary_body.skeleton.joints[joint].position.xyz.x << "\t" << secondary_body.skeleton.joints[joint].position.xyz.y << "\t" << secondary_body.skeleton.joints[joint].position.xyz.z << std::endl;

        main.at<float>(0, joint) = main_body.skeleton.joints[joint].position.v[0];
        main.at<float>(1, joint) = main_body.skeleton.joints[joint].position.v[1];
        main.at<float>(2, joint) = main_body.skeleton.joints[joint].position.v[2];
        secondary.at<float>(0, joint) = secondary_body.skeleton.joints[joint].position.v[0];
        secondary.at<float>(1, joint) = secondary_body.skeleton.joints[joint].position.v[1];
        secondary.at<float>(2, joint) = secondary_body.skeleton.joints[joint].position.v[2];
    }

    // compute R, T that give the transformation matrix from secondary to main coordinate space
    Mat R, T;
    arun(secondary, main, R, T); // R: [3x3], T: [1x3]

    // main, secondary: [3 x 32]
    Mat T_rep;
    cv::repeat(T, 1, K4ABT_JOINT_COUNT, T_rep); // T_rep: [3x32]

    Mat secondary_tf;
    cv::add(R * secondary, T_rep, secondary_tf);

    // update secondary body object with transformed positions
    for (int joint = 0; joint < (int)K4ABT_JOINT_COUNT; joint++)
    {
        secondary_body.skeleton.joints[joint].position.v[0] = secondary_tf.at<float>(0, joint);
        secondary_body.skeleton.joints[joint].position.v[1] = secondary_tf.at<float>(1, joint);
        secondary_body.skeleton.joints[joint].position.v[2] = secondary_tf.at<float>(2, joint);
    }
}

/*
    compute R and T extrinsic matrix components using least squares fitting of two 3-D point sets
 */
void arun(Mat &streamfrom, Mat &streamto, Mat &R, Mat &T)
{
    // find mean across the row (all joints for each x,y,z)
    Mat avg_streamfrom, avg_streamto;                     // main, secondary: [3x32]
    reduce(streamfrom, avg_streamfrom, 1, CV_REDUCE_AVG); // avg_main: [3x1]
    reduce(streamto, avg_streamto, 1, CV_REDUCE_AVG);

    // find deviations from the mean
    Mat rep_avg_streamfrom, rep_avg_streamto;
    cv::repeat(avg_streamfrom, 1, K4ABT_JOINT_COUNT, rep_avg_streamfrom); // rep_avg_main: [3x32]
    cv::repeat(avg_streamto, 1, K4ABT_JOINT_COUNT, rep_avg_streamto);

    Mat streamfrom_sub, streamto_sub;
    subtract(streamfrom, rep_avg_streamfrom, streamfrom_sub);
    subtract(streamto, rep_avg_streamto, streamto_sub);

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
        T = Mat(1, 3, CV_32F, Scalar(1, 1, 1));
    }
    // final sizes:: R: [3x3], T: [3x1]
}

/* 
    take average positions from 2 position (x,y,z) tuples
*/
k4a_float3_t get_average_position_xyz(k4a_float3_t main_position, k4a_float3_t secondary_position, int main_or_secondary)
{
    k4a_float3_t avgPosition;
    switch (main_or_secondary)
    {
    case 0:
        avgPosition = main_position;
        break;
    case 1:
        avgPosition = secondary_position;
        break;
    case 2: // take average
        for (int i = 0; i < 3; i++)
        {
            avgPosition.v[0] = (float)(main_position.v[0] + secondary_position.v[0]) / 2;
            avgPosition.v[1] = (float)(main_position.v[1] + secondary_position.v[1]) / 2;
            avgPosition.v[2] = (float)(main_position.v[2] + secondary_position.v[2]) / 2;
        }
        break;
    }

    return avgPosition;
}

/*
    take average orientations from 2 quaternions
 */
k4a_quaternion_t get_average_quaternion_xyzw(k4a_quaternion_t main_quaternion, k4a_quaternion_t secondary_quaternion, int main_or_secondary)
{
    k4a_quaternion_t avgQuaternion;
    switch (main_or_secondary)
    {
    case 0:
        avgQuaternion = main_quaternion;
        break;
    case 1:
        avgQuaternion = secondary_quaternion;
        break;
    case 2: // take average
        for (int i = 0; i < 4; i++)
        {
            avgQuaternion.v[0] = (float)(main_quaternion.v[0] + secondary_quaternion.v[0]) / 2;
            avgQuaternion.v[1] = (float)(main_quaternion.v[1] + secondary_quaternion.v[1]) / 2;
            avgQuaternion.v[2] = (float)(main_quaternion.v[2] + secondary_quaternion.v[2]) / 2;
            avgQuaternion.v[3] = (float)(main_quaternion.v[3] + secondary_quaternion.v[3]) / 2;
        }
        break;
    }

    return avgQuaternion;
}

/*
    translate confidence level enum to string representation
 */
string confidenceMap(k4abt_joint_confidence_level_t confidence_level)
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