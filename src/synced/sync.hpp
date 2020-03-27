#include <k4a/k4a.hpp>
#include <k4abt.hpp>

#include "transformation.h"
#include "MultiDeviceCapturer.h"
#include "colors.h"

using namespace color;
using namespace cv;
using namespace std;

class sync{

    public:

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
        string confidenceEnumMapping(k4abt_joint_confidence_level_t confidence_level);

        void print_body_information(k4abt_body_t main_body, k4abt_body_t secondary_body, cv::Mat& main, cv::Mat& secondary, cv::Matx33f main_intrinsic_matrix);
        void plotBody(std::vector<cv::Point> dataMain, std::vector<cv::Point> dataSecondary, cv::Mat main, cv::Mat secondary);
        void transform_body(k4abt_body_t& main_body, k4abt_body_t& secondary_body);
        void arun(Mat& main, Mat& secondary, Mat& R, Mat& T);

    private:

};