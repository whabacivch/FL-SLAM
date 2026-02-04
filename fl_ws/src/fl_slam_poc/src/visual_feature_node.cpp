/**
 * visual_feature_node: C++ visual preprocessing for GC SLAM.
 *
 * Subscribes to RGBDImage (rgb8 + 32FC1 depth), extracts ORB keypoints,
 * samples depth with robust statistics, computes closed-form backprojection
 * covariance, and publishes a fixed-budget VisualFeatureBatch.
 *
 * Single path: no fallbacks. Fail-fast if required params are missing.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "fl_slam_poc/msg/rgbd_image.hpp"
#include "fl_slam_poc/msg/visual_feature.hpp"
#include "fl_slam_poc/msg/visual_feature_batch.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

namespace
{
constexpr int kQueueDepth = 10;
constexpr double kEps = 1e-12;
constexpr double kLog2Pi = 1.8378770664093453;  // log(2*pi)

inline double safe_sigmoid(double x)
{
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

inline double clamp(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

struct QuadFitResult
{
  Eigen::Vector2d grad_z;
  Eigen::Matrix2d H;
  Eigen::Vector3d normal;
  double K;
  double ma;
  double lam_min;
};

}  // namespace

class VisualFeatureNode final : public rclcpp::Node
{
 public:
  VisualFeatureNode() : rclcpp::Node("visual_feature_node")
  {
    declare_parameter<std::string>("input_topic", "/gc/sensors/camera_rgbd");
    declare_parameter<std::string>("output_topic", "/gc/sensors/visual_features");
    declare_parameter<int>("max_features", 512);
    declare_parameter<int>("orb_nlevels", 8);
    declare_parameter<double>("orb_scale_factor", 1.2);

    declare_parameter<std::string>("depth_sample_mode", "median3");
    declare_parameter<double>("pixel_sigma", 1.0);
    declare_parameter<std::string>("depth_model", "linear");
    declare_parameter<double>("depth_sigma0", 0.01);
    declare_parameter<double>("depth_sigma_slope", 0.01);
    declare_parameter<double>("min_depth_m", 0.05);
    declare_parameter<double>("max_depth_m", 80.0);
    declare_parameter<double>("depth_validity_slope", 5.0);
    declare_parameter<double>("response_soft_scale", 50.0);
    declare_parameter<double>("depth_scale", 1.0);
    declare_parameter<double>("cov_reg_eps", 1e-9);
    declare_parameter<double>("invalid_cov_inflate", 1e6);

    declare_parameter<int>("hex_radius", 2);
    declare_parameter<int>("quad_fit_radius", 2);
    declare_parameter<int>("quad_fit_min_points", 6);
    declare_parameter<double>("quad_fit_lstsq_eps", 1e-8);

    declare_parameter<double>("student_t_nu", 3.0);
    declare_parameter<double>("student_t_w_min", 0.1);

    declare_parameter<double>("ma_tau", 10.0);
    declare_parameter<double>("ma_delta_inflate", 1e-4);

    declare_parameter<double>("kappa0", 1.0);
    declare_parameter<double>("kappa_alpha", 10.0);
    declare_parameter<double>("kappa_max", 100.0);
    declare_parameter<double>("kappa_min", 0.1);

    declare_parameter<std::vector<double>>("camera_K", {500.0, 500.0, 320.0, 240.0});

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    max_features_ = get_parameter("max_features").as_int();
    orb_nlevels_ = get_parameter("orb_nlevels").as_int();
    orb_scale_factor_ = get_parameter("orb_scale_factor").as_double();

    depth_sample_mode_ = get_parameter("depth_sample_mode").as_string();
    pixel_sigma_ = get_parameter("pixel_sigma").as_double();
    depth_model_ = get_parameter("depth_model").as_string();
    depth_sigma0_ = get_parameter("depth_sigma0").as_double();
    depth_sigma_slope_ = get_parameter("depth_sigma_slope").as_double();
    min_depth_m_ = get_parameter("min_depth_m").as_double();
    max_depth_m_ = get_parameter("max_depth_m").as_double();
    depth_validity_slope_ = get_parameter("depth_validity_slope").as_double();
    response_soft_scale_ = get_parameter("response_soft_scale").as_double();
    depth_scale_ = get_parameter("depth_scale").as_double();
    cov_reg_eps_ = get_parameter("cov_reg_eps").as_double();
    invalid_cov_inflate_ = get_parameter("invalid_cov_inflate").as_double();

    hex_radius_ = get_parameter("hex_radius").as_int();
    quad_fit_radius_ = get_parameter("quad_fit_radius").as_int();
    quad_fit_min_points_ = get_parameter("quad_fit_min_points").as_int();
    quad_fit_lstsq_eps_ = get_parameter("quad_fit_lstsq_eps").as_double();

    student_t_nu_ = get_parameter("student_t_nu").as_double();
    student_t_w_min_ = get_parameter("student_t_w_min").as_double();

    ma_tau_ = get_parameter("ma_tau").as_double();
    ma_delta_inflate_ = get_parameter("ma_delta_inflate").as_double();

    kappa0_ = get_parameter("kappa0").as_double();
    kappa_alpha_ = get_parameter("kappa_alpha").as_double();
    kappa_max_ = get_parameter("kappa_max").as_double();
    kappa_min_ = get_parameter("kappa_min").as_double();

    auto K = get_parameter("camera_K").as_double_array();
    if (K.size() != 4) {
      throw std::runtime_error("camera_K must be [fx, fy, cx, cy]");
    }
    fx_ = K[0];
    fy_ = K[1];
    cx_ = K[2];
    cy_ = K[3];

    orb_ = cv::ORB::create(
      max_features_,
      static_cast<float>(orb_scale_factor_),
      orb_nlevels_,
      31,
      0,
      2,
      cv::ORB::HARRIS_SCORE,
      31,
      20);

    pub_ = create_publisher<fl_slam_poc::msg::VisualFeatureBatch>(output_topic_, kQueueDepth);
    sub_ = create_subscription<fl_slam_poc::msg::RGBDImage>(
      input_topic_, kQueueDepth,
      std::bind(&VisualFeatureNode::on_rgbd, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "visual_feature_node started: input=%s output=%s max_features=%d",
      input_topic_.c_str(), output_topic_.c_str(), max_features_);
  }

 private:
  struct DepthSample
  {
    double z_m;
    double z_var_m2;
    bool valid;
    std::vector<double> zs;
  };

  double depth_to_meters(double z_raw) const
  {
    return z_raw * depth_scale_;
  }

  DepthSample validate_depth(double z_m, std::optional<double> var_override = std::nullopt) const
  {
    DepthSample out{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), false, {}};
    if (!std::isfinite(z_m) || z_m <= 0.0) {
      return out;
    }
    out.z_m = z_m;
    out.z_var_m2 = (var_override && std::isfinite(*var_override)) ? *var_override : std::numeric_limits<double>::quiet_NaN();
    out.valid = true;
    return out;
  }

  double depth_weight(double z_m) const
  {
    if (!std::isfinite(z_m)) {
      return 0.0;
    }
    const double a = depth_validity_slope_;
    const double w_min = 1.0 / (1.0 + std::exp(-a * (z_m - min_depth_m_)));
    const double w_max = 1.0 / (1.0 + std::exp(+a * (z_m - max_depth_m_)));
    return clamp(w_min * w_max, 0.0, 1.0);
  }

  double response_weight(double response) const
  {
    if (response <= 0.0) {
      return 0.0;
    }
    const double s = response_soft_scale_;
    return response / (response + s);
  }

  double depth_sigma(double z_m) const
  {
    const double z = std::abs(z_m);
    if (depth_model_ == "linear") {
      return depth_sigma0_ + depth_sigma_slope_ * z;
    }
    if (depth_model_ == "quadratic") {
      return depth_sigma0_ + depth_sigma_slope_ * (z * z);
    }
    throw std::runtime_error("Unknown depth_model: " + depth_model_);
  }

  DepthSample depth_sample(const cv::Mat & depth, double u, double v)
  {
    const int x = static_cast<int>(std::round(u));
    const int y = static_cast<int>(std::round(v));
    const int H = depth.rows;
    const int W = depth.cols;

    DepthSample out{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), false, {}};
    if (x < 0 || y < 0 || x >= W || y >= H) {
      return out;
    }

    if (depth_sample_mode_ == "nearest") {
      const double z = depth_to_meters(depth.at<float>(y, x));
      auto vres = validate_depth(z);
      return vres;
    }

    if (depth_sample_mode_ == "hex") {
      return depth_sample_hex(depth, u, v, x, y);
    }

    int r = 1;
    if (depth_sample_mode_ == "median3") {
      r = 1;
    } else if (depth_sample_mode_ == "median5") {
      r = 2;
    } else {
      throw std::runtime_error("Unknown depth_sample_mode: " + depth_sample_mode_);
    }

    const int x0 = std::max(0, x - r);
    const int x1 = std::min(W - 1, x + r);
    const int y0 = std::max(0, y - r);
    const int y1 = std::min(H - 1, y + r);

    std::vector<double> zs;
    zs.reserve((x1 - x0 + 1) * (y1 - y0 + 1));
    for (int yy = y0; yy <= y1; ++yy) {
      for (int xx = x0; xx <= x1; ++xx) {
        double z = depth_to_meters(depth.at<float>(yy, xx));
        if (std::isfinite(z) && z > 0.0) {
          zs.push_back(z);
        }
      }
    }

    if (zs.empty()) {
      return out;
    }

    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
    const double z_med = zs[zs.size() / 2];
    double z_var = std::numeric_limits<double>::quiet_NaN();
    if (zs.size() >= 4) {
      double mean = 0.0;
      for (double z : zs) { mean += z; }
      mean /= static_cast<double>(zs.size());
      double var = 0.0;
      for (double z : zs) {
        double d = z - mean;
        var += d * d;
      }
      z_var = var / static_cast<double>(zs.size());
    }

    DepthSample res = validate_depth(z_med, z_var);
    res.zs = zs;
    return res;
  }

  DepthSample depth_sample_hex(const cv::Mat & depth, double u, double v, int x, int y)
  {
    const int H = depth.rows;
    const int W = depth.cols;
    const int r = std::max(1, hex_radius_);
    std::vector<std::pair<int, int>> offsets;
    offsets.reserve(7);
    offsets.emplace_back(0, 0);
    for (int k = 0; k < 6; ++k) {
      const double ang = static_cast<double>(k) * M_PI / 3.0;
      const int dx = static_cast<int>(std::round(r * std::cos(ang)));
      const int dy = static_cast<int>(std::round(r * std::sin(ang)));
      offsets.emplace_back(dx, dy);
    }

    std::vector<double> zs;
    for (const auto & off : offsets) {
      const int xi = x + off.first;
      const int yi = y + off.second;
      if (xi >= 0 && xi < W && yi >= 0 && yi < H) {
        double z = depth_to_meters(depth.at<float>(yi, xi));
        if (std::isfinite(z) && z > 0.0) {
          zs.push_back(z);
        }
      }
    }

    DepthSample out{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), false, {}};
    if (zs.size() < 4) {
      return out;
    }

    std::vector<double> zs_copy = zs;
    std::nth_element(zs_copy.begin(), zs_copy.begin() + zs_copy.size() / 2, zs_copy.end());
    const double z_hat = zs_copy[zs_copy.size() / 2];

    std::vector<double> abs_dev;
    abs_dev.reserve(zs.size());
    for (double z : zs) {
      abs_dev.push_back(std::abs(z - z_hat));
    }
    std::nth_element(abs_dev.begin(), abs_dev.begin() + abs_dev.size() / 2, abs_dev.end());
    const double mad = abs_dev[abs_dev.size() / 2];
    const double sigma_z = 1.4826 * mad;
    const double sigma_z2 = sigma_z * sigma_z;

    DepthSample res = validate_depth(z_hat, sigma_z2);
    res.zs = zs;
    return res;
  }

  double student_t_effective_var(double z_hat, double sigma_z2, const std::vector<double> & zs) const
  {
    if (zs.size() < 2 || !std::isfinite(sigma_z2) || sigma_z2 <= 0.0) {
      return sigma_z2;
    }
    const double nu = student_t_nu_;
    const double w_min = student_t_w_min_;
    const double sigma2 = std::max(sigma_z2, kEps);
    double q = 0.0;
    for (double zi : zs) {
      const double r = zi - z_hat;
      q += (r * r);
    }
    q /= (static_cast<double>(zs.size()) * sigma2 + kEps);
    double w = (nu + 1.0) / (nu + q);
    if (w < w_min) {
      w = w_min;
    }
    return sigma_z2 / w;
  }

  Eigen::Vector3d backproject(double u, double v, double z) const
  {
    const double x = (u - cx_) * z / fx_;
    const double y = (v - cy_) * z / fy_;
    return Eigen::Vector3d(x, y, z);
  }

  Eigen::Matrix3d backprojection_cov(double u, double v, double z, double var_u, double var_v, double var_z) const
  {
    const double du = u - cx_;
    const double dv = v - cy_;
    const double vu = std::max(var_u, 0.0);
    const double vv = std::max(var_v, 0.0);
    const double vz = std::max(var_z, 0.0);

    const double var_x = (z * z * vu + du * du * vz + vu * vz) / (fx_ * fx_);
    const double var_y = (z * z * vv + dv * dv * vz + vv * vz) / (fy_ * fy_);
    const double var_z_local = vz;

    const double cov_xy = (du * dv * vz) / (fx_ * fy_);
    const double cov_xz = (du * vz) / fx_;
    const double cov_yz = (dv * vz) / fy_;

    Eigen::Matrix3d cov;
    cov << var_x, cov_xy, cov_xz,
           cov_xy, var_y, cov_yz,
           cov_xz, cov_yz, var_z_local;
    return cov;
  }

  std::optional<QuadFitResult> quadratic_fit(const cv::Mat & depth, double u, double v, double z_hat)
  {
    const int x0 = static_cast<int>(std::round(u));
    const int y0 = static_cast<int>(std::round(v));
    const int H_rows = depth.rows;
    const int W_cols = depth.cols;
    const int r = std::max(1, quad_fit_radius_);

    std::vector<Eigen::Vector3d> pts;
    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        const int xi = x0 + dx;
        const int yi = y0 + dy;
        if (xi >= 0 && xi < W_cols && yi >= 0 && yi < H_rows) {
          const double zi = depth_to_meters(depth.at<float>(yi, xi));
          if (std::isfinite(zi) && zi > 0.0) {
            pts.emplace_back(static_cast<double>(xi), static_cast<double>(yi), zi);
          }
        }
      }
    }

    if (static_cast<int>(pts.size()) < quad_fit_min_points_) {
      return std::nullopt;
    }

    const int n_pts = static_cast<int>(pts.size());
    Eigen::MatrixXd A(n_pts, 6);
    Eigen::VectorXd b(n_pts);
    for (int i = 0; i < n_pts; ++i) {
      const double u_t = pts[i](0) - u;
      const double v_t = pts[i](1) - v;
      A(i, 0) = u_t * u_t;
      A(i, 1) = u_t * v_t;
      A(i, 2) = v_t * v_t;
      A(i, 3) = u_t;
      A(i, 4) = v_t;
      A(i, 5) = 1.0;
      b(i) = pts[i](2);
    }

    Eigen::MatrixXd AtA = A.transpose() * A;
    AtA += quad_fit_lstsq_eps_ * Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd Atb = A.transpose() * b;
    Eigen::VectorXd beta = AtA.ldlt().solve(Atb);

    const double a = beta(0);
    const double b_coef = beta(1);
    const double c = beta(2);
    const double d = beta(3);
    const double e = beta(4);

    const double zu_pix = d;
    const double zv_pix = e;

    Eigen::Matrix2d H_pix;
    H_pix << 2.0 * a, b_coef,
             b_coef, 2.0 * c;

    const double z = std::max(z_hat, 1e-6);
    const double sx = fx_ / z;
    const double sy = fy_ / z;
    const double zu = sx * zu_pix;
    const double zv = sy * zv_pix;

    Eigen::Matrix2d H_uv;
    H_uv << sx * sx * H_pix(0, 0), sx * sy * H_pix(0, 1),
            sx * sy * H_pix(1, 0), sy * sy * H_pix(1, 1);

    const double det_H = H_uv.determinant();
    const double grad_sq = zu * zu + zv * zv;
    const double denom = (1.0 + grad_sq);
    const double K = (denom > 0.0) ? det_H / (denom * denom) : 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(H_uv);
    const double lam_min = eig.eigenvalues().minCoeff();

    Eigen::Vector3d normal(-zu, -zv, 1.0);
    const double nnorm = normal.norm();
    if (nnorm > 0.0) {
      normal /= nnorm;
    }

    QuadFitResult out;
    out.grad_z = Eigen::Vector2d(zu, zv);
    out.H = H_uv;
    out.normal = normal;
    out.K = K;
    out.ma = det_H;
    out.lam_min = lam_min;
    return out;
  }

  void on_rgbd(const fl_slam_poc::msg::RGBDImage::SharedPtr msg)
  {
    cv_bridge::CvImageConstPtr rgb_ptr;
    cv_bridge::CvImageConstPtr depth_ptr;
    try {
      rgb_ptr = cv_bridge::toCvCopy(msg->rgb, "rgb8");
      depth_ptr = cv_bridge::toCvCopy(msg->depth, "32FC1");
    } catch (const std::exception & exc) {
      RCLCPP_ERROR(get_logger(), "cv_bridge conversion failed: %s", exc.what());
      return;
    }

    const cv::Mat & rgb = rgb_ptr->image;
    const cv::Mat & depth = depth_ptr->image;
    if (rgb.empty() || depth.empty()) {
      RCLCPP_WARN(get_logger(), "RGBDImage has empty rgb or depth");
      return;
    }

    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    orb_->detectAndCompute(gray, cv::noArray(), kps, desc);
    if (kps.empty()) {
      publish_empty(msg->header);
      return;
    }

    if (static_cast<int>(kps.size()) > max_features_) {
      std::vector<int> idx(kps.size());
      for (size_t i = 0; i < kps.size(); ++i) idx[i] = static_cast<int>(i);
      std::nth_element(idx.begin(), idx.begin() + max_features_, idx.end(),
        [&](int a, int b) { return kps[a].response > kps[b].response; });
      idx.resize(max_features_);
      std::vector<cv::KeyPoint> kps_sel;
      kps_sel.reserve(idx.size());
      for (int i : idx) {
        kps_sel.push_back(kps[i]);
      }
      kps.swap(kps_sel);
    }

    fl_slam_poc::msg::VisualFeatureBatch out;
    out.header = msg->header;
    out.capacity = static_cast<uint32_t>(max_features_);
    out.stamp_sec = static_cast<double>(msg->header.stamp.sec) +
                    static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    out.features.clear();
    out.features.reserve(max_features_);

    int count = 0;
    for (const auto & kp : kps) {
      const double u = kp.pt.x;
      const double v = kp.pt.y;

      DepthSample ds = depth_sample(depth, u, v);
      const bool z_valid = ds.valid && std::isfinite(ds.z_m) && ds.z_m > 0.0;

      const double w_depth = z_valid ? depth_weight(ds.z_m) : 0.0;
      const double w_resp = response_weight(kp.response);
      const double weight = clamp(w_depth * w_resp, 0.0, 1.0);

      std::optional<QuadFitResult> quad_fit;
      if (z_valid) {
        quad_fit = quadratic_fit(depth, u, v, ds.z_m);
      }

      double var_z_eff = std::numeric_limits<double>::quiet_NaN();
      if (z_valid && std::isfinite(ds.z_var_m2)) {
        const double base_var = std::max(ds.z_var_m2, depth_sigma(ds.z_m) * depth_sigma(ds.z_m));
        var_z_eff = student_t_effective_var(ds.z_m, base_var, ds.zs);
      } else if (z_valid) {
        var_z_eff = depth_sigma(ds.z_m) * depth_sigma(ds.z_m);
      }
      if (z_valid && !std::isfinite(var_z_eff)) {
        var_z_eff = depth_sigma(ds.z_m) * depth_sigma(ds.z_m);
      }

      Eigen::Vector3d xyz(0.0, 0.0, 0.0);
      Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * invalid_cov_inflate_;

      if (z_valid) {
        xyz = backproject(u, v, ds.z_m);
        const double var_z_use = std::max(var_z_eff, depth_sigma(ds.z_m) * depth_sigma(ds.z_m));
        cov = backprojection_cov(u, v, ds.z_m, pixel_sigma_ * pixel_sigma_, pixel_sigma_ * pixel_sigma_, var_z_use);
        if (quad_fit.has_value()) {
          const double w_ma = safe_sigmoid(ma_tau_ * quad_fit->lam_min);
          cov += (1.0 - w_ma) * ma_delta_inflate_ * Eigen::Matrix3d::Identity();
        }
      }

      Eigen::Vector3d mu_app(0.0, 0.0, 0.0);
      double kappa_app = 0.0;
      if (quad_fit.has_value()) {
        const double w_ma = safe_sigmoid(ma_tau_ * quad_fit->lam_min);
        mu_app = quad_fit->normal;
        const double rel_noise = (z_valid && std::isfinite(var_z_eff))
          ? std::sqrt(var_z_eff) / (ds.z_m + kEps)
          : 1.0;
        const double rho = 1.0 / (rel_noise + kEps);
        const double mean_curv_mag = std::sqrt(std::abs(quad_fit->K));
        kappa_app = kappa0_ + kappa_alpha_ * mean_curv_mag * rho;
        kappa_app = clamp(kappa_app, kappa_min_, kappa_max_);
        kappa_app = kappa_app * w_ma;
      }

      double depth_sigma_c_sq = std::numeric_limits<double>::quiet_NaN();
      double depth_lambda_c = 0.0;
      double depth_theta_c = 0.0;
      if (z_valid && std::isfinite(var_z_eff) && var_z_eff > 0.0) {
        depth_sigma_c_sq = var_z_eff;
        depth_lambda_c = 1.0 / depth_sigma_c_sq;
        depth_theta_c = depth_lambda_c * ds.z_m;
      }

      const int ix = static_cast<int>(std::round(clamp(u, 0.0, static_cast<double>(rgb.cols - 1))));
      const int iy = static_cast<int>(std::round(clamp(v, 0.0, static_cast<double>(rgb.rows - 1))));
      cv::Vec3b pix = rgb.at<cv::Vec3b>(iy, ix);
      const double r = static_cast<double>(pix[0]) / 255.0;
      const double g = static_cast<double>(pix[1]) / 255.0;
      const double b = static_cast<double>(pix[2]) / 255.0;

      fl_slam_poc::msg::VisualFeature feat;
      feat.u = u;
      feat.v = v;
      feat.xyz = {xyz(0), xyz(1), xyz(2)};
      feat.cov_xyz = {
        cov(0, 0), cov(0, 1), cov(0, 2),
        cov(1, 0), cov(1, 1), cov(1, 2),
        cov(2, 0), cov(2, 1), cov(2, 2)
      };
      feat.weight = weight;
      feat.mu_app = {mu_app(0), mu_app(1), mu_app(2)};
      feat.kappa_app = kappa_app;
      feat.color = {r, g, b};
      feat.depth_lambda_c = depth_lambda_c;
      feat.depth_theta_c = depth_theta_c;
      feat.depth_sigma_c_sq = depth_sigma_c_sq;
      feat.valid = true;

      out.features.push_back(feat);
      count += 1;
      if (count >= max_features_) {
        break;
      }
    }

    out.count = static_cast<uint32_t>(count);

    while (static_cast<int>(out.features.size()) < max_features_) {
      fl_slam_poc::msg::VisualFeature pad;
      pad.valid = false;
      out.features.push_back(pad);
    }

    pub_->publish(out);
  }

  void publish_empty(const std_msgs::msg::Header & header)
  {
    fl_slam_poc::msg::VisualFeatureBatch out;
    out.header = header;
    out.capacity = static_cast<uint32_t>(max_features_);
    out.stamp_sec = static_cast<double>(header.stamp.sec) +
                    static_cast<double>(header.stamp.nanosec) * 1e-9;
    out.count = 0;
    out.features.clear();
    out.features.reserve(max_features_);
    for (int i = 0; i < max_features_; ++i) {
      fl_slam_poc::msg::VisualFeature pad;
      pad.valid = false;
      out.features.push_back(pad);
    }
    pub_->publish(out);
  }

  std::string input_topic_;
  std::string output_topic_;
  int max_features_ = 512;
  int orb_nlevels_ = 8;
  double orb_scale_factor_ = 1.2;

  std::string depth_sample_mode_;
  double pixel_sigma_ = 1.0;
  std::string depth_model_;
  double depth_sigma0_ = 0.01;
  double depth_sigma_slope_ = 0.01;
  double min_depth_m_ = 0.05;
  double max_depth_m_ = 80.0;
  double depth_validity_slope_ = 5.0;
  double response_soft_scale_ = 50.0;
  double depth_scale_ = 1.0;
  double cov_reg_eps_ = 1e-9;
  double invalid_cov_inflate_ = 1e6;

  int hex_radius_ = 2;
  int quad_fit_radius_ = 2;
  int quad_fit_min_points_ = 6;
  double quad_fit_lstsq_eps_ = 1e-8;

  double student_t_nu_ = 3.0;
  double student_t_w_min_ = 0.1;

  double ma_tau_ = 10.0;
  double ma_delta_inflate_ = 1e-4;

  double kappa0_ = 1.0;
  double kappa_alpha_ = 10.0;
  double kappa_max_ = 100.0;
  double kappa_min_ = 0.1;

  double fx_ = 500.0;
  double fy_ = 500.0;
  double cx_ = 320.0;
  double cy_ = 240.0;

  cv::Ptr<cv::ORB> orb_;

  rclcpp::Publisher<fl_slam_poc::msg::VisualFeatureBatch>::SharedPtr pub_;
  rclcpp::Subscription<fl_slam_poc::msg::RGBDImage>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VisualFeatureNode>());
  rclcpp::shutdown();
  return 0;
}
