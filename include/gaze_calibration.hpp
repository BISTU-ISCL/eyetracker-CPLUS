#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace eyetracker {

struct EyeFeatures {
    // 瞳孔中心坐标（像素）
    std::array<double, 2> pupil_center{0.0, 0.0};
    // 瞳孔半径（像素）
    double pupil_radius{0.0};
    // 虹膜上耀点列表（像素）
    std::vector<std::array<double, 2>> glints{};
    // 上下眼睑之间的像素距离，用于判断眨眼
    double eyelid_opening{0.0};
};

struct CalibrationResult {
    bool ready{false};
    std::array<double, 3> coeff_x{0.0, 0.0, 0.0};
    std::array<double, 3> coeff_y{0.0, 0.0, 0.0};
    std::array<double, 2> residual_error{0.0, 0.0};
};

class BrightPupilCalibrator {
public:
    // screen_point 以屏幕像素坐标给出
    void add_calibration_sample(const EyeFeatures& eye, std::array<double, 2> screen_point);

    // 计算线性标定矩阵：屏幕点 = W * (瞳孔中心 - 耀点重心) + b
    CalibrationResult compute();

    // 给定一帧特征，估算视线落点
    std::optional<std::array<double, 2>> estimate_gaze(const EyeFeatures& eye) const;

    // 基于多帧眼睑开合情况估算眨眼频率（次/分钟）
    static double estimate_blink_frequency(const std::vector<EyeFeatures>& frames,
                                           double fps,
                                           double blink_threshold);

    // 将瞳孔半径转换为直径，支持像素到毫米的换算
    static double pupil_diameter_mm(const EyeFeatures& eye, double pixel_to_mm = 1.0);

private:
    std::vector<EyeFeatures> eye_samples_{};
    std::vector<std::array<double, 2>> screen_samples_{};
    CalibrationResult result_{};

    static std::array<double, 2> pupil_glint_vector(const EyeFeatures& eye);
};

// 用于演示的序列化输出
std::string to_string(const CalibrationResult& result);

}  // namespace eyetracker

