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

#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "transformation.h"
#include "MultiDeviceCapturer.h"
#include "colors.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using namespace color;

// cv::Point prevPt1 = cv::Point(0,0), prevPt2 = cv::Point(0,0);
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
void print_body_information(k4abt_body_t main_body, k4abt_body_t secondary_body, ofstream &outfile, cv::Mat& main, cv::Mat& secondary);
void print_body_index_map_middle_line(k4a::image body_index_map);
k4a_float3_t get_average_position_xyz(k4a_float3_t main_position, k4a_float3_t secondary_position, int main_or_secondary);
k4a_quaternion_t get_average_quaternion_xyzw(k4a_quaternion_t main_quaternion, k4a_quaternion_t secondary_quaternion, int main_or_secondar);
int get_average_confidence(k4abt_joint_confidence_level_t mainCI, k4abt_joint_confidence_level_t secondaryCI);
string confidenceEnumMapping(k4abt_joint_confidence_level_t confidence_level);

void plotBody(std::vector<cv::Point> dataMain, std::vector<cv::Point> dataSecondary, cv::Mat main, cv::Mat secondary);

int main(int argc, char **argv)
{
    // output file stream
    ofstream outfile;
    outfile.open("./joints_output.csv", ios::out);
    outfile << ",,Position,,,Orientation,,,,Confidence Level" << endl;
    outfile << "Body ID," << "Joint #," << "x,y,z," << "x,y,z,w" << endl;

    float chessboard_square_length = 0.; // must be included in the input params
    int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
    int32_t powerline_freq = 2;          // default to a 60 Hz powerline
    cv::Size chessboard_pattern(0, 0);   // height, width. Both need to be set.
    uint16_t depth_threshold = 1000;     // default to 1 meter
    size_t num_devices = 0;
    double calibration_timeout = 60.0; // default to timing out after 60s of trying to get calibrated
    double greenscreen_duration = std::numeric_limits<double>::max(); // run forever

    vector<uint32_t> device_indices{ 0 }; // Set up a MultiDeviceCapturer to handle getting many synchronous captures
                                          // Note that the order of indices in device_indices is not necessarily
                                          // preserved because MultiDeviceCapturer tries to find the master device based
                                          // on which one has sync out plugged in. Start with just { 0 }, and add
                                          // another if needed

    if (argc < 5)
    {
        cout << "Usage: green_screen <num-cameras> <board-height> <board-width> <board-square-length> "
                "[depth-threshold-mm (default 1000)] [color-exposure-time-usec (default 8000)] "
                "[powerline-frequency-mode (default 2 for 60 Hz)] [calibration-timeout-sec (default 60)]"
                "[greenscreen-duration-sec (default infinity- run forever)]"
             << endl;

        cerr << "Not enough arguments!\n";
        exit(1);
    }
    else
    {
        num_devices = static_cast<size_t>(atoi(argv[1]));
        if (num_devices > k4a::device::get_installed_count())
        {
            cerr << "Not enough cameras plugged in!\n";
            exit(1);
        }
        chessboard_pattern.height = atoi(argv[2]);
        chessboard_pattern.width = atoi(argv[3]);
        chessboard_square_length = static_cast<float>(atof(argv[4]));

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
                        if (argc > 9)
                        {
                            greenscreen_duration = atof(argv[9]);
                        }
                    }
                }
            }
        }
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
    if (chessboard_pattern.height == 0)
    {
        cerr << "Chessboard height is not properly set!\n";
        exit(1);
    }
    if (chessboard_pattern.width == 0)
    {
        cerr << "Chessboard height is not properly set!\n";
        exit(1);
    }
    if (chessboard_square_length == 0.)
    {
        cerr << "Chessboard square size is not properly set!\n";
        exit(1);
    }

    cout << "Chessboard height: " << chessboard_pattern.height << ". Chessboard width: " << chessboard_pattern.width
         << ". Chessboard square length: " << chessboard_square_length << endl;
    cout << "Depth threshold: : " << depth_threshold << ". Color exposure time: " << color_exposure_usec
         << ". Powerline frequency mode: " << powerline_freq << endl;

    MultiDeviceCapturer capturer(device_indices, color_exposure_usec, powerline_freq);

    // Create configurations for devices
    k4a_device_configuration_t main_config = get_master_config();
    if (num_devices == 1) // no need to have a master cable if it's standalone
    {
        main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    }
    k4a_device_configuration_t secondary_config = get_subordinate_config();

    // Construct all the things that we'll need whether or not we are running with 1 or 2 cameras
    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,
                                                                                     main_config.color_resolution);

    // Set up a transformation. DO THIS OUTSIDE OF YOUR MAIN LOOP! Constructing transformations involves time-intensive
    // hardware setup and should not change once you have a rigid setup, so only call it once or it will run very
    // slowly.
    k4a::transformation main_depth_to_main_color(main_calibration);

    capturer.start_devices(main_config, secondary_config);
    // get an image to be the background
    vector<k4a::capture> background_captures = capturer.get_synchronized_captures(secondary_config);
    cv::Mat background_image = color_to_opencv(background_captures[0].get_color_image());
    cv::Mat output_image = background_image.clone(); // allocated outside the loop to avoid re-creating every time

    if (num_devices == 1)
    {
        std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
        while (std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count() <
               greenscreen_duration)
        {
            vector<k4a::capture> captures;
            // secondary_config isn't actually used here because there's no secondary device but the function needs it
            captures = capturer.get_synchronized_captures(secondary_config, true);
            k4a::image main_color_image = captures[0].get_color_image();
            k4a::image main_depth_image = captures[0].get_depth_image();

            // let's green screen out things that are far away.
            // first: let's get the main depth image into the color camera space
            k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
            main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
            cv::Mat cv_main_depth_in_main_color = depth_to_opencv(main_depth_in_main_color);
            cv::Mat cv_main_color_image = color_to_opencv(main_color_image);

            // single-camera case
            cv::Mat within_threshold_range = (cv_main_depth_in_main_color != 0) &
                                             (cv_main_depth_in_main_color < depth_threshold);
            // show the close details
            cv_main_color_image.copyTo(output_image, within_threshold_range);
            // hide the rest with the background image
            background_image.copyTo(output_image, ~within_threshold_range);
            cv::namedWindow("Green Screen");
            cv::imshow("Green Screen", output_image);
            cv::waitKey(1);
        }
    }
    else if (num_devices == 2)
    {
        // This wraps all the device-to-device details
        Transformation tr_secondary_color_to_main_color = calibrate_devices(capturer,
                                                                            main_config,
                                                                            secondary_config,
                                                                            chessboard_pattern,
                                                                            chessboard_square_length,
                                                                            calibration_timeout);
        // save matrix
        // const char* filename1 = "./RotationMatrix.txt";
        // ofstream file1(filename1);
        // file1 <<"" << cv::format(tr_secondary_color_to_main_color.R, cv::Formatter::FMT_CSV);
        // file1.close();

        // const char* filename2 = "./TranslationMatrix.txt";
        // ofstream file2(filename2);
        // file2 <<"" << cv::format(tr_secondary_color_to_main_color.t, cv::Formatter::FMT_CSV);
        // file2.close();
        // exit(1);


        k4a::calibration secondary_calibration =
            capturer.get_subordinate_device_by_index(0).get_calibration(secondary_config.depth_mode,
                                                                        secondary_config.color_resolution);
        // Get the transformation from secondary depth to secondary color using its calibration object
        Transformation tr_secondary_depth_to_secondary_color = get_depth_to_color_transformation_from_calibration(
            secondary_calibration);

        // We now have the secondary depth to secondary color transform. We also have the transformation from the
        // secondary color perspective to the main color perspective from the calibration earlier. Now let's compose the
        // depth secondary -> color secondary, color secondary -> color main into depth secondary -> color main
        Transformation tr_secondary_depth_to_main_color = tr_secondary_depth_to_secondary_color.compose_with(
            tr_secondary_color_to_main_color);

        // Construct a new calibration object to transform from the secondary depth camera to the main color camera
        k4a::calibration secondary_depth_to_main_color_cal =
            construct_device_to_device_calibration(main_calibration,
                                                   secondary_calibration,
                                                   tr_secondary_depth_to_main_color);
        k4a::transformation secondary_depth_to_main_color(secondary_depth_to_main_color_cal);

        // ============ START initialize body tracker
        k4abt::tracker main_tracker = k4abt::tracker::create(main_calibration);
        k4abt::tracker secondary_tracker = k4abt::tracker::create(secondary_calibration);
        // ============ END initialize body tracker


        k4abt_body_t main_body;
        k4abt_body_t secondary_body;
        k4abt::frame main_body_frame;
        k4abt::frame secondary_body_frame;

        std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
        while (std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count() <
               greenscreen_duration)
        {

            vector<k4a::capture> captures;
            captures = capturer.get_synchronized_captures(secondary_config, true);
            // get main color and depth image
            k4a::image main_color_image = captures[0].get_color_image();
            k4a::image main_depth_image = captures[0].get_depth_image();
            // get secondary color and depth image
            k4a::image secondary_color_image = captures[1].get_color_image();
            k4a::image secondary_depth_image = captures[1].get_depth_image();

            // let's green screen out things that are far away.
            // first: let's get the main depth image into the color camera space
            k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
            main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
            cv::Mat cv_main_depth_in_main_color = depth_to_opencv(main_depth_in_main_color);
            cv::Mat cv_main_color_image = color_to_opencv(main_color_image);


            // Get the depth image in the main color perspective
            k4a::image secondary_depth_in_main_color = create_depth_image_like(main_color_image);
            secondary_depth_to_main_color.depth_image_to_color_camera(secondary_depth_image,
                                                                      &secondary_depth_in_main_color);
            cv::Mat cv_secondary_depth_in_main_color = depth_to_opencv(secondary_depth_in_main_color);

            k4a::image secondary_depth_in_secondary_color = create_depth_image_like(secondary_color_image);
            main_depth_to_main_color.depth_image_to_color_camera(secondary_depth_image, &secondary_depth_in_secondary_color);
            cv::Mat cv_secondary_depth_in_secondary_color = depth_to_opencv(secondary_depth_in_secondary_color);
            cv::Mat cv_secondary_color_image = color_to_opencv(secondary_color_image);

            // opencv secondary color Mat / opencv main color Mat
            cv::namedWindow("cv_secondary_color_image", CV_WINDOW_AUTOSIZE);
            cv::imshow("cv_secondary_color_image", cv_secondary_color_image);
            cv::waitKey(1);
            cv::namedWindow("cv_main_color_image", CV_WINDOW_AUTOSIZE);
            cv::imshow("cv_main_color_image", cv_main_color_image);
            cv::waitKey(1);

            // captures[0].set_color_image(main_color_image);
            // captures[0].set_depth_image(main_depth_in_main_color);
            // captures[1].set_color_image(secondary_color_image);
            // captures[1].set_depth_image(secondary_depth_in_secondary_color);

            // ========== START read skeletal data

            // read from main and secondary trackers
            if (!main_tracker.enqueue_capture(captures[0]))
            {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                std::cout << "Error! Add capture to tracker process queue timeout!" << std::endl;
                break;
            }
            // std::cout << "checkpoint - 1\n" << std::endl;
            uint32_t main_num_bodies;
            uint32_t secondary_num_bodies;
            try
            {
                try
                {
                    main_body_frame = main_tracker.pop_result();
                    // std::cout << "checkpoint - 2a\n" << std::endl;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }

                secondary_tracker.enqueue_capture(captures[1]);
                try
                {
                    secondary_body_frame = secondary_tracker.pop_result();
                    // std::cout << "checkpoint - 2b\n" << std::endl;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                uint32_t main_num_bodies;
                uint32_t secondary_num_bodies;
                if ((main_body_frame != nullptr) && (secondary_body_frame != nullptr))
                {
                    main_num_bodies = main_body_frame.get_num_bodies();
                    secondary_num_bodies = secondary_body_frame.get_num_bodies();
                    // std::cout << "checkpoint - 3\n" << std::endl;
                }

                // std::cout << "main num bodies found: " << main_num_bodies << std::endl;
                // std::cout << "secondary num bodies found: " << secondary_num_bodies << std::endl;

                if (main_num_bodies > 0 && secondary_num_bodies > 0)
                {
                    uint32_t num_bodies_min = std::min(main_num_bodies, secondary_num_bodies);
                    std::cout << num_bodies_min << " bodies are detected!" << std::endl;

                    // std::cout << "checkpoint - 4\n" << std::endl;
                    for (uint32_t i = 0; i < num_bodies_min; i++)
                    {
                        main_body = main_body_frame.get_body(i);
                        secondary_body = secondary_body_frame.get_body(i);

                        // print body information
                        std::cout << main_body.id << " / " << secondary_body.id << std::endl;
                        if (main_body.id == secondary_body.id)
                        {
                            print_body_information(main_body, secondary_body, outfile, cv_main_color_image, cv_secondary_color_image);
                        }
                        else
                        {
                            std::cerr << "Body ID's don't match...\n" << std::endl;
                        }

                    }
                }
                else
                {
                    //  It should never hit timeout when K4A_WAIT_INFINITE is set.
                    std::cout << "Error! Pop body frame result time out!" << std::endl;
                    continue;
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            // ========== END read skeletal data


            // cv::imshow("cv_secondary_depth_image", cv_secondary_depth_in_main_color);
            // cv::waitKey(1);
            // cv::imshow("cv_main_depth_image", cv_main_depth_in_main_color);
            // cv::waitKey(1);

            // exit(0);
        }
    }
    else
    {
        cerr << "Invalid number of devices!" << endl;
        exit(1);
    }
    outfile.close();
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
    return { i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6 };
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
            cout << "Could not find the chessboard corners in the secondary image. Trying again...\n";
        }
        // Likewise, if the chessboard was found in the secondary image, it was not found in the main image.
        else if (found_chessboard_secondary)
        {
            cout << "Could not find the chessboard corners in the main image. Trying again...\n";
        }
        // The only remaining case is the corners were in neither image.
        else
        {
            cout << "Could not find the chessboard corners in either image. Trying again...\n";
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
                cv::Point3f{ w * chessboard_square_length, h * chessboard_square_length, 0.0 });
        }
    }

    // Calibrating the cameras requires a lot of data. OpenCV's stereoCalibrate function requires:
    // - a list of points in real 3d space that will be used to calibrate*
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
    cout << "Finished calibrating!\n";
    cout << "Got error of " << error << "\n";
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

        cv::namedWindow("Chessboard view from secondary", CV_WINDOW_AUTOSIZE);
        cv::imshow("Chessboard view from secondary", cv_secondary_color_image);
        cv::waitKey(1);

        // save image
        // cv::imwrite( "./cv_main_color_image.jpg", cv_main_color_image);
        // cv::imwrite( "./cv_secondary_color_image.jpg", cv_secondary_color_image);



        // Get 20 frames before doing calibration.
        if (main_chessboard_corners_list.size() >= 20)
        {
            cout << "Calculating calibration..." << endl;
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
    exit(1);
}

static k4a::image create_depth_image_like(const k4a::image &im)
{
    return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
                              im.get_width_pixels(),
                              im.get_height_pixels(),
                              im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

// =============== START body tracking functions

void print_body_information(k4abt_body_t main_body, k4abt_body_t secondary_body, ofstream& outfile, cv::Mat& main, cv::Mat& secondary)
{
    std::cout << "Main Body ID: " << main_body.id << std::endl;
    std::cout << "Secondary Body ID: " << secondary_body.id << std::endl;
    std::vector<cv::Point> dataMain, dataSecondary;
    for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
    {
        k4a_float3_t main_position = main_body.skeleton.joints[i].position;
        k4a_quaternion_t main_orientation = main_body.skeleton.joints[i].orientation;
        k4abt_joint_confidence_level_t main_confidence_level = main_body.skeleton.joints[i].confidence_level;

        k4a_float3_t secondary_position = secondary_body.skeleton.joints[i].position;
        k4a_quaternion_t secondary_orientation = secondary_body.skeleton.joints[i].orientation;
        k4abt_joint_confidence_level_t secondary_confidence_level = secondary_body.skeleton.joints[i].confidence_level;

        int main_or_secondary = get_average_confidence(main_confidence_level, secondary_confidence_level);
        k4a_float3_t avgPos = get_average_position_xyz(main_position, secondary_position, main_or_secondary);
        k4a_quaternion_t avgQuaternion = get_average_quaternion_xyzw(main_orientation, secondary_orientation, main_or_secondary);
        k4abt_joint_confidence_level_t avgCI;
        if (main_or_secondary == 0) { avgCI = main_confidence_level; }
        else if (main_or_secondary == 1) { avgCI = secondary_confidence_level; }
        else { avgCI = main_confidence_level; }
        // printf("[Synced] Joint[%d]: Position[mm] ( %f, %f, %f ); Orientation ( %f, %f, %f, %f); Confidence Level (%d)  \n", i, avgPos.v[0], avgPos.v[1], avgPos.v[2], main_orientation.v[0], avgQuaternion.v[1], avgQuaternion.v[2], avgQuaternion.v[3], avgCI);


        // ============ Display both joint streams in one RGB image ==========
        // plot 2D points for main camera
        int radius = 10;
        cv::Point center1 = cv::Point(main_position.xyz.x, main_position.xyz.y);
        dataMain.push_back(center1);
        // if (prevPt1 == cv::Point(0,0))
        // {
        //     prevPt1 = center1;
        // }
        cv::Scalar color = COLORS_red;
        circle(main, center1, radius, color, CV_FILLED);
        putText(main, to_string(i), center1, FONT_HERSHEY_DUPLEX, 1, Scalar(0,143,143), 2);
        // line( main, prevPt1, center1, Scalar( 110, 220, 0 ),  thickness, 8 );
        // prevPt1 = center1;
        cv::namedWindow("cv_main_color_image", CV_WINDOW_AUTOSIZE);
        cv::imshow("cv_main_color_image", main);
        cv::waitKey(1);

        // plot 2D points for subordinate camera - 1
        radius = 5;
        cv::Point center2 = cv::Point(main_position.xyz.x, main_position.xyz.y);
        dataSecondary.push_back(center2);
        // if (prevPt2 == cv::Point(0,0))
        // {
        //     prevPt2 = center2;
        // }
        color =  COLORS_blue;
        circle(secondary, center2, radius, color, CV_FILLED);
        putText(secondary, to_string(i), center2, FONT_HERSHEY_DUPLEX, 1, Scalar(0,143,143), 2);
        // line( secondary, prevPt2, center2, Scalar( 110, 220, 0 ),  thickness, 8 );
        // prevPt2 = center2;

        cv::namedWindow("cv_secondary_color_image", CV_WINDOW_AUTOSIZE);
        cv::imshow("cv_secondary_color_image", secondary);
        cv::waitKey(1);

        if (outfile.is_open())
        {
            outfile << main_body.id << "," << i << "," << avgPos.v[0] << "," << avgPos.v[1] << "," << avgPos.v[2] << "," << avgQuaternion.v[0] << "," << avgQuaternion.v[1] << "," << avgQuaternion.v[2] << "," << avgQuaternion.v[3] << "," << confidenceEnumMapping(avgCI) << "," << endl;
        }
    }
    plotBody(dataMain, dataSecondary, main, secondary);
}

void plotBody(std::vector<cv::Point> dataMain, std::vector<cv::Point> dataSecondary, cv::Mat main, cv::Mat secondary)
{
    cv::Scalar color; // green
    int counter = 0, thickness = 0;
    std::list<std::vector<cv::Point>> datalist {dataMain, dataSecondary};
    std::vector<cv::Mat> imgList {main, secondary};
    for (auto stream : datalist)
    {
        switch (counter)
        {
            case 0:
                color = COLORS_red;
                thickness = 10;
                break;
            case 1:
                color = COLORS_blue;
                thickness = 3;
                break;
        }
        // [child]: child joint to parent joint
        // 1: spine naval to pelvis
        line(imgList.at(0), stream[1], stream[0], color,  thickness, 8 );
        // 2: spine chest to spine naval
        line(imgList.at(0), stream[2], stream[1], color,  thickness, 8 );
        // 3: neck to spine chest
        line(imgList.at(0), stream[3], stream[2], color,  thickness, 8 );
        // 4: clavicle left to spine chest
        line(imgList.at(0), stream[4], stream[2], color,  thickness, 8 );
        // 5: shoulder left to clavicle left
        line(imgList.at(0), stream[5], stream[4], color,  thickness, 8 );
        // 6: elbow left to shoulder left
        line(imgList.at(0), stream[6], stream[5], color,  thickness, 8 );
        // 7: wrist left to elbow left
        line(imgList.at(0), stream[7], stream[6], color,  thickness, 8 );
        // 8: hand left to wrist left
        line(imgList.at(0), stream[8], stream[7], color,  thickness, 8 );
        // 9: handtip left to hand left
        line(imgList.at(0), stream[9], stream[8], color,  thickness, 8 );
        // 10: thumb left to writst left
        line(imgList.at(0), stream[10], stream[7], color,  thickness, 8 );
        // 11: clavicle right to spine chest
        line(imgList.at(0), stream[11], stream[2], color,  thickness, 8 );
        // 12: shoulder right to clavicle right
        line(imgList.at(0), stream[12], stream[11], color,  thickness, 8 );
        // 13: elbow rith to shoulder right
        line(imgList.at(0), stream[13], stream[12], color,  thickness, 8 );
        // 14: wrist right to elbow right
        line(imgList.at(0), stream[14], stream[13], color,  thickness, 8 );
        // 15: hand right to wrist right
        line(imgList.at(0), stream[15], stream[14], color,  thickness, 8 );
        // 16: handtip right to hand right
        line(imgList.at(0), stream[16], stream[15], color,  thickness, 8 );
        // 17: thumb right to writst right
        line(imgList.at(0), stream[17], stream[14], color,  thickness, 8 );
        // 18: hip left to pelvis
        line(imgList.at(0), stream[18], stream[0], color,  thickness, 8 );
        // 19: knee left to hip left
        line(imgList.at(0), stream[19], stream[18], color,  thickness, 8 );
        // 20: ankle left to knee left
        line(imgList.at(0), stream[20], stream[19], color,  thickness, 8 );
        // 21: foot left to ankle left
        line(imgList.at(0), stream[21], stream[20], color,  thickness, 8 );
        // 22: hip right to plevis
        line(imgList.at(0), stream[22], stream[0], color,  thickness, 8 );
        // 23: knee right to hip right
        line(imgList.at(0), stream[23], stream[22], color,  thickness, 8 );
        // 24: ankle right to hip right
        line(imgList.at(0), stream[24], stream[22], color,  thickness, 8 );
        // 25: foot right to ankle right
        line(imgList.at(0), stream[25], stream[24], color,  thickness, 8 );
        // 26: head to neck
        line(imgList.at(0), stream[26], stream[3], color,  thickness, 8 );
        // 27: nose to head
        line(imgList.at(0), stream[27], stream[26], color,  thickness, 8 );
        // 28: eye left to head
        line(imgList.at(0), stream[28], stream[26], color,  thickness, 8 );
        // 29: ear left to head
        line(imgList.at(0), stream[29], stream[26], color,  thickness, 8 );
        // 30: eye right to head
        line(imgList.at(0), stream[30], stream[26], color,  thickness, 8 );
        // 31: ear right to head
        line(imgList.at(0), stream[31], stream[26], color,  thickness, 8 );
        counter += 1;
    }

    // std::vector<cv::Point> stream;
    // stream = dataMain;
    // color = COLORS_red;
    // int thickness = 5;
    // // [child]: child joint to parent joint
    // // 1: spine naval to pelvis
    // line(imgList.at(0), stream[1], stream[0], color,  thickness, 8 );
    // // 2: spine chest to spine naval
    // line(imgList.at(0), stream[2], stream[1], color,  thickness, 8 );
    // // 3: neck to spine chest
    // line(imgList.at(0), stream[3], stream[2], color,  thickness, 8 );
    // // 4: clavicle left to spine chest
    // line(imgList.at(0), stream[4], stream[2], color,  thickness, 8 );
    // // 5: shoulder left to clavicle left
    // line(imgList.at(0), stream[5], stream[4], color,  thickness, 8 );
    // // 6: elbow left to shoulder left
    // line(imgList.at(0), stream[6], stream[5], color,  thickness, 8 );
    // // 7: wrist left to elbow left
    // line(imgList.at(0), stream[7], stream[6], color,  thickness, 8 );
    // // 8: hand left to wrist left
    // line(imgList.at(0), stream[8], stream[7], color,  thickness, 8 );
    // // 9: handtip left to hand left
    // line(imgList.at(0), stream[9], stream[8], color,  thickness, 8 );
    // // 10: thumb left to writst left
    // line(imgList.at(0), stream[10], stream[7], color,  thickness, 8 );
    // // 11: clavicle right to spine chest
    // line(imgList.at(0), stream[11], stream[2], color,  thickness, 8 );
    // // 12: shoulder right to clavicle right
    // line(imgList.at(0), stream[12], stream[11], color,  thickness, 8 );
    // // 13: elbow rith to shoulder right
    // line(imgList.at(0), stream[13], stream[12], color,  thickness, 8 );
    // // 14: wrist right to elbow right
    // line(imgList.at(0), stream[14], stream[13], color,  thickness, 8 );
    // // 15: hand right to wrist right
    // line(imgList.at(0), stream[15], stream[14], color,  thickness, 8 );
    // // 16: handtip right to hand right
    // line(imgList.at(0), stream[16], stream[15], color,  thickness, 8 );
    // // 17: thumb right to writst right
    // line(imgList.at(0), stream[17], stream[14], color,  thickness, 8 );
    // // 18: hip left to pelvis
    // line(imgList.at(0), stream[18], stream[0], color,  thickness, 8 );
    // // 19: knee left to hip left
    // line(imgList.at(0), stream[19], stream[18], color,  thickness, 8 );
    // // 20: ankle left to knee left
    // line(imgList.at(0), stream[20], stream[19], color,  thickness, 8 );
    // // 21: foot left to ankle left
    // line(imgList.at(0), stream[21], stream[20], color,  thickness, 8 );
    // // 22: hip right to plevis
    // line(imgList.at(0), stream[22], stream[0], color,  thickness, 8 );
    // // 23: knee right to hip right
    // line(imgList.at(0), stream[23], stream[22], color,  thickness, 8 );
    // // 24: ankle right to hip right
    // line(imgList.at(0), stream[24], stream[22], color,  thickness, 8 );
    // // 25: foot right to ankle right
    // line(imgList.at(0), stream[25], stream[24], color,  thickness, 8 );
    // // 26: head to neck
    // line(imgList.at(0), stream[26], stream[3], color,  thickness, 8 );
    // // 27: nose to head
    // line(imgList.at(0), stream[27], stream[26], color,  thickness, 8 );
    // // 28: eye left to head
    // line(imgList.at(0), stream[28], stream[26], color,  thickness, 8 );
    // // 29: ear left to head
    // line(imgList.at(0), stream[29], stream[26], color,  thickness, 8 );
    // // 30: eye right to head
    // line(imgList.at(0), stream[30], stream[26], color,  thickness, 8 );
    // // 31: ear right to head
    // line(imgList.at(0), stream[31], stream[26], color,  thickness, 8 );

    // stream = dataSecondary;
    // color = COLORS_blue;
    // thickness = 1;
    // // [child]: child joint to parent joint
    // // 1: spine naval to pelvis
    // line(imgList.at(0), stream[1], stream[0], color,  thickness, 8 );
    // // 2: spine chest to spine naval
    // line(imgList.at(0), stream[2], stream[1], color,  thickness, 8 );
    // // 3: neck to spine chest
    // line(imgList.at(0), stream[3], stream[2], color,  thickness, 8 );
    // // 4: clavicle left to spine chest
    // line(imgList.at(0), stream[4], stream[2], color,  thickness, 8 );
    // // 5: shoulder left to clavicle left
    // line(imgList.at(0), stream[5], stream[4], color,  thickness, 8 );
    // // 6: elbow left to shoulder left
    // line(imgList.at(0), stream[6], stream[5], color,  thickness, 8 );
    // // 7: wrist left to elbow left
    // line(imgList.at(0), stream[7], stream[6], color,  thickness, 8 );
    // // 8: hand left to wrist left
    // line(imgList.at(0), stream[8], stream[7], color,  thickness, 8 );
    // // 9: handtip left to hand left
    // line(imgList.at(0), stream[9], stream[8], color,  thickness, 8 );
    // // 10: thumb left to writst left
    // line(imgList.at(0), stream[10], stream[7], color,  thickness, 8 );
    // // 11: clavicle right to spine chest
    // line(imgList.at(0), stream[11], stream[2], color,  thickness, 8 );
    // // 12: shoulder right to clavicle right
    // line(imgList.at(0), stream[12], stream[11], color,  thickness, 8 );
    // // 13: elbow rith to shoulder right
    // line(imgList.at(0), stream[13], stream[12], color,  thickness, 8 );
    // // 14: wrist right to elbow right
    // line(imgList.at(0), stream[14], stream[13], color,  thickness, 8 );
    // // 15: hand right to wrist right
    // line(imgList.at(0), stream[15], stream[14], color,  thickness, 8 );
    // // 16: handtip right to hand right
    // line(imgList.at(0), stream[16], stream[15], color,  thickness, 8 );
    // // 17: thumb right to writst right
    // line(imgList.at(0), stream[17], stream[14], color,  thickness, 8 );
    // // 18: hip left to pelvis
    // line(imgList.at(0), stream[18], stream[0], color,  thickness, 8 );
    // // 19: knee left to hip left
    // line(imgList.at(0), stream[19], stream[18], color,  thickness, 8 );
    // // 20: ankle left to knee left
    // line(imgList.at(0), stream[20], stream[19], color,  thickness, 8 );
    // // 21: foot left to ankle left
    // line(imgList.at(0), stream[21], stream[20], color,  thickness, 8 );
    // // 22: hip right to plevis
    // line(imgList.at(0), stream[22], stream[0], color,  thickness, 8 );
    // // 23: knee right to hip right
    // line(imgList.at(0), stream[23], stream[22], color,  thickness, 8 );
    // // 24: ankle right to hip right
    // line(imgList.at(0), stream[24], stream[22], color,  thickness, 8 );
    // // 25: foot right to ankle right
    // line(imgList.at(0), stream[25], stream[24], color,  thickness, 8 );
    // // 26: head to neck
    // line(imgList.at(0), stream[26], stream[3], color,  thickness, 8 );
    // // 27: nose to head
    // line(imgList.at(0), stream[27], stream[26], color,  thickness, 8 );
    // // 28: eye left to head
    // line(imgList.at(0), stream[28], stream[26], color,  thickness, 8 );
    // // 29: ear left to head
    // line(imgList.at(0), stream[29], stream[26], color,  thickness, 8 );
    // // 30: eye right to head
    // line(imgList.at(0), stream[30], stream[26], color,  thickness, 8 );
    // // 31: ear right to head
    // line(imgList.at(0), stream[31], stream[26], color,  thickness, 8 );

    cv::namedWindow("MAIN", CV_WINDOW_AUTOSIZE);
    cv::imshow("MAIN", imgList.at(0));
    cv::waitKey(1);
    // cv::imshow("MAIN", secondary);
    // cv::imshow("SECONDARY", secondary);
    // cv::waitKey(1);

}

void print_body_index_map_middle_line(k4a::image body_index_map)
{
    uint8_t* body_index_map_buffer = body_index_map.get_buffer();

    // Given body_index_map pixel type should be uint8, the stride_byte should be the same as width
    // TODO: Since there is no API to query the byte-per-pixel information, we have to compare the width and stride to
    // know the information. We should replace this assert with proper byte-per-pixel query once the API is provided by
    // K4A SDK.
    assert(body_index_map.get_stride_bytes() == body_index_map.get_width_pixels());

    int middle_line_num = body_index_map.get_height_pixels() / 2;
    body_index_map_buffer = body_index_map_buffer + middle_line_num * body_index_map.get_width_pixels();

    std::cout << "BodyIndexMap at Line " << middle_line_num << ":" << std::endl;
    for (int i = 0; i < body_index_map.get_width_pixels(); i++)
    {
        std::cout << (int)*body_index_map_buffer << ", ";
        body_index_map_buffer++;
    }
    std::cout << std::endl;
}
// ================ END body tracking functions

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
        case 2:  // take average
            for(int i = 0; i < 3; i++)
            {
                avgPosition.v[0] = (float) (main_position.v[0] + secondary_position.v[0]) / 2;
                avgPosition.v[1] = (float) (main_position.v[1] + secondary_position.v[1]) / 2;
                avgPosition.v[2] = (float) (main_position.v[2] + secondary_position.v[2]) / 2;
            }
            break;
    }

    return avgPosition;
}

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
        case 2:  // take average
            for(int i = 0; i < 4; i++)
            {
                avgQuaternion.v[0] = (float) (main_quaternion.v[0] + secondary_quaternion.v[0]) / 2;
                avgQuaternion.v[1] = (float) (main_quaternion.v[1] + secondary_quaternion.v[1]) / 2;
                avgQuaternion.v[2] = (float) (main_quaternion.v[2] + secondary_quaternion.v[2]) / 2;
                avgQuaternion.v[3] = (float) (main_quaternion.v[3] + secondary_quaternion.v[3]) / 2;
            }
            break;
    }

    return avgQuaternion;
}

int get_average_confidence(k4abt_joint_confidence_level_t main_confidence_level, k4abt_joint_confidence_level_t secondary_confidence_level)
{
    int main_or_secondary = 0; // main
    if (main_confidence_level < secondary_confidence_level)
    {
        main_or_secondary = 1; // secondary
    } else if (main_confidence_level == secondary_confidence_level)
    {
        main_or_secondary = 2; // equal confidence
    }
    return main_or_secondary;
}

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
