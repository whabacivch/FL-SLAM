/**
 * camera_rgbd_node: Single node for RGB-D I/O.
 *
 * Subscribes to compressed RGB and raw depth, decodes/scales, pairs by timestamp,
 * publishes single RGBDImage. Replaces image_decompress_cpp + depth_passthrough.
 *
 * Single path: no fallbacks, no dual outputs. Fail-fast if depth missing.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float64.hpp"
#include "fl_slam_poc/msg/rgbd_image.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace
{
constexpr int kQueueDepth = 10;
constexpr double kDepthMmToM = 1000.0;

std::string to_lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

double stamp_to_sec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

}  // namespace

class CameraRgbdNode final : public rclcpp::Node
{
 public:
  CameraRgbdNode() : rclcpp::Node("camera_rgbd_node")
  {
    declare_parameter<std::string>("rgb_compressed_topic", "/camera/color/image_raw/compressed");
    declare_parameter<std::string>("depth_raw_topic", "/camera/aligned_depth_to_color/image_raw");
    declare_parameter<std::string>("output_topic", "/gc/sensors/camera_rgbd");
    declare_parameter<bool>("depth_scale_mm_to_m", true);
    declare_parameter<double>("pair_max_dt_sec", 0.05);
    declare_parameter<std::string>("qos_reliability", "best_effort");
    declare_parameter<bool>("enable_time_alignment", false);
    declare_parameter<std::string>("time_reference_topic", "/gc/sensors/time_reference");
    declare_parameter<double>("max_drift_sec", 0.5);

    rgb_in_ = get_parameter("rgb_compressed_topic").as_string();
    depth_in_ = get_parameter("depth_raw_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    depth_scale_mm_to_m_ = get_parameter("depth_scale_mm_to_m").as_bool();
    pair_max_dt_sec_ = get_parameter("pair_max_dt_sec").as_double();
    qos_reliability_ = to_lower(get_parameter("qos_reliability").as_string());
    enable_time_alignment_ = get_parameter("enable_time_alignment").as_bool();
    time_reference_topic_ = get_parameter("time_reference_topic").as_string();
    max_drift_sec_ = get_parameter("max_drift_sec").as_double();

    if (rgb_in_.empty() || depth_in_.empty()) {
      RCLCPP_ERROR(get_logger(),
                   "camera_rgbd_node requires both rgb_compressed_topic and depth_raw_topic");
      throw std::runtime_error("Missing required topic parameters");
    }

    auto qos = build_qos(qos_reliability_);
    pub_ = create_publisher<fl_slam_poc::msg::RGBDImage>(output_topic_, kQueueDepth);
    if (enable_time_alignment_) {
      if (time_reference_topic_.empty()) {
        throw std::runtime_error("camera_rgbd_node: enable_time_alignment requires time_reference_topic");
      }
      time_sub_ = create_subscription<std_msgs::msg::Float64>(
        time_reference_topic_, qos,
        [this](const std_msgs::msg::Float64::SharedPtr msg) {
          last_ref_stamp_ = msg->data;
          if (offset_ready_ && last_local_stamp_.has_value()) {
            const double aligned = last_local_stamp_.value() + offset_sec_;
            const double drift = std::abs(aligned - last_ref_stamp_.value());
            if (drift > max_drift_sec_) {
              throw std::runtime_error("camera_rgbd_node: timestamp drift exceeds max_drift_sec");
            }
          }
        });
    }

    sub_rgb_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      rgb_in_, qos,
      std::bind(&CameraRgbdNode::on_rgb_compressed, this, std::placeholders::_1));

    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(
      depth_in_, qos,
      std::bind(&CameraRgbdNode::on_depth_raw, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "camera_rgbd_node started:\n"
                "  RGB (compressed): %s\n"
                "  Depth (raw):      %s\n"
                "  Output:           %s\n"
                "  pair_max_dt_sec:  %.3f\n"
                "  depth_scale_mm_to_m: %s",
                rgb_in_.c_str(), depth_in_.c_str(), output_topic_.c_str(),
                pair_max_dt_sec_, depth_scale_mm_to_m_ ? "true" : "false");
    if (enable_time_alignment_) {
      RCLCPP_INFO(get_logger(),
                  "  time alignment: enabled (ref=%s, max_drift_sec=%.3f)",
                  time_reference_topic_.c_str(), max_drift_sec_);
    }
  }

 private:
  struct TimestampedImage
  {
    builtin_interfaces::msg::Time stamp;
    std::string frame_id;
    cv::Mat image;
    bool valid{false};
  };

  rclcpp::QoS build_qos(const std::string & reliability)
  {
    if (reliability == "reliable") {
      return rclcpp::QoS(kQueueDepth).reliable();
    }
    if (reliability == "system_default") {
      return rclcpp::QoS(kQueueDepth).reliability(rclcpp::ReliabilityPolicy::SystemDefault);
    }
    // Default: best_effort (common for sensor bags)
    return rclcpp::QoS(kQueueDepth).best_effort();
  }

  void on_rgb_compressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    try {
      cv::Mat buffer(1, static_cast<int>(msg->data.size()), CV_8UC1,
                     const_cast<unsigned char *>(msg->data.data()));
      cv::Mat bgr = cv::imdecode(buffer, cv::IMREAD_COLOR);
      if (bgr.empty()) {
        log_error(rgb_errors_, "RGB decode returned empty image");
        return;
      }

      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

      current_rgb_.stamp = msg->header.stamp;
      current_rgb_.frame_id = msg->header.frame_id;
      current_rgb_.image = rgb;
      current_rgb_.valid = true;

      rgb_count_ += 1;
      if (rgb_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First RGB decoded: %dx%d", rgb.cols, rgb.rows);
      }

      try_publish_pair();
    } catch (const std::exception & exc) {
      log_error(rgb_errors_, std::string("RGB decode failed: ") + exc.what());
    }
  }

  void on_depth_raw(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      const int height = static_cast<int>(msg->height);
      const int width = static_cast<int>(msg->width);
      if (height <= 0 || width <= 0) {
        log_error(depth_errors_, "Depth: invalid size");
        return;
      }

      cv::Mat depth_m;

      if (msg->encoding == "16UC1" || msg->encoding == "16SC1") {
        if (!depth_scale_mm_to_m_) {
          log_error(depth_errors_, "Received 16-bit depth but scale_mm_to_m=false");
          return;
        }
        // Handle potential row padding (step may exceed width*2)
        const int row_bytes = static_cast<int>(msg->step);
        const int row_elems = row_bytes / 2;
        cv::Mat raw(height, row_elems, CV_16UC1,
                    const_cast<unsigned char *>(msg->data.data()));
        cv::Mat cropped = raw(cv::Rect(0, 0, width, height));
        cropped.convertTo(depth_m, CV_32F, 1.0 / kDepthMmToM);
      } else if (msg->encoding == "32FC1") {
        const int row_bytes = static_cast<int>(msg->step);
        const int row_elems = row_bytes / 4;
        cv::Mat raw(height, row_elems, CV_32FC1,
                    const_cast<unsigned char *>(msg->data.data()));
        depth_m = raw(cv::Rect(0, 0, width, height)).clone();
      } else {
        log_error(depth_errors_, "Unsupported depth encoding: " + msg->encoding);
        return;
      }

      current_depth_.stamp = msg->header.stamp;
      current_depth_.frame_id = msg->header.frame_id;
      current_depth_.image = depth_m;
      current_depth_.valid = true;

      depth_count_ += 1;
      if (depth_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First depth received: %dx%d", depth_m.cols, depth_m.rows);
      }

      try_publish_pair();
    } catch (const std::exception & exc) {
      log_error(depth_errors_, std::string("Depth processing failed: ") + exc.what());
    }
  }

  void try_publish_pair()
  {
    if (!current_rgb_.valid || !current_depth_.valid) {
      return;
    }

    const double t_rgb = stamp_to_sec(current_rgb_.stamp);
    const double t_depth = stamp_to_sec(current_depth_.stamp);
    const double dt = std::abs(t_rgb - t_depth);

    if (dt > pair_max_dt_sec_) {
      // Timestamps too far apart; wait for closer match
      return;
    }

    // Build and publish RGBDImage
    fl_slam_poc::msg::RGBDImage out;

    // Use RGB timestamp for the pair header
    out.header.stamp = current_rgb_.stamp;
    out.header.frame_id = !current_rgb_.frame_id.empty() ? current_rgb_.frame_id
                        : (!current_depth_.frame_id.empty() ? current_depth_.frame_id : "camera");

    // Optional time alignment: apply single offset from reference timeline.
    if (enable_time_alignment_) {
      const double stamp_sec = stamp_to_sec(out.header.stamp);
      last_local_stamp_ = stamp_sec;
      if (!offset_ready_ && last_ref_stamp_.has_value()) {
        offset_sec_ = last_ref_stamp_.value() - stamp_sec;
        offset_ready_ = true;
      }
      if (offset_ready_) {
        const double aligned = stamp_sec + offset_sec_;
        if (last_out_stamp_.has_value() && aligned < last_out_stamp_.value()) {
          throw std::runtime_error("camera_rgbd_node: non-monotonic aligned timestamps");
        }
        last_out_stamp_ = aligned;
        out.header.stamp.sec = static_cast<int32_t>(aligned);
        out.header.stamp.nanosec = static_cast<uint32_t>((aligned - std::floor(aligned)) * 1e9);
      }
    }

    // RGB image
    cv_bridge::CvImage rgb_bridge;
    rgb_bridge.header.stamp = current_rgb_.stamp;
    rgb_bridge.header.frame_id = out.header.frame_id;
    rgb_bridge.encoding = "rgb8";
    rgb_bridge.image = current_rgb_.image;
    out.rgb = *rgb_bridge.toImageMsg();

    // Depth image
    cv_bridge::CvImage depth_bridge;
    depth_bridge.header.stamp = out.header.stamp;  // keep coherent pair stamp
    depth_bridge.header.frame_id = out.header.frame_id;
    depth_bridge.encoding = "32FC1";
    depth_bridge.image = current_depth_.image;
    out.depth = *depth_bridge.toImageMsg();

    pub_->publish(out);

    pair_count_ += 1;
    if (pair_count_ == 1) {
      RCLCPP_INFO(get_logger(),
                  "First RGB-D pair published (dt=%.4f sec, RGB: %dx%d, Depth: %dx%d)",
                  dt,
                  current_rgb_.image.cols, current_rgb_.image.rows,
                  current_depth_.image.cols, current_depth_.image.rows);
    }

    // Fail-fast: if time alignment enabled but offset never initialized, something is wrong
    // (likely QoS mismatch preventing time_reference messages from being received)
    if (enable_time_alignment_ && !offset_ready_) {
      unaligned_publish_count_++;
      if (unaligned_publish_count_ == 1) {
        RCLCPP_WARN(get_logger(),
          "Time alignment enabled but no time_reference received yet - publishing unaligned");
      }
      if (unaligned_publish_count_ > 10) {
        throw std::runtime_error(
          "camera_rgbd_node: enable_time_alignment=true but no time_reference received after 10 frames. "
          "Check QoS compatibility (publisher uses BEST_EFFORT) or set enable_time_alignment=false.");
      }
    }

    // Clear current to require fresh data for next pair
    current_rgb_.valid = false;
    current_depth_.valid = false;
  }

  void log_error(int & counter, const std::string & message)
  {
    counter += 1;
    if (counter <= 5) {
      RCLCPP_WARN(get_logger(), "%s", message.c_str());
    }
  }

  std::string rgb_in_;
  std::string depth_in_;
  std::string output_topic_;
  bool depth_scale_mm_to_m_{true};
  double pair_max_dt_sec_{0.05};
  std::string qos_reliability_;
  bool enable_time_alignment_{false};
  std::string time_reference_topic_;
  double max_drift_sec_{0.5};
  std::optional<double> last_ref_stamp_{};
  std::optional<double> last_out_stamp_{};
  std::optional<double> last_local_stamp_{};
  double offset_sec_{0.0};
  bool offset_ready_{false};

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_rgb_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr time_sub_;
  rclcpp::Publisher<fl_slam_poc::msg::RGBDImage>::SharedPtr pub_;

  TimestampedImage current_rgb_;
  TimestampedImage current_depth_;

  int rgb_count_{0};
  int depth_count_{0};
  int pair_count_{0};
  int rgb_errors_{0};
  int depth_errors_{0};
  int unaligned_publish_count_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraRgbdNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
