#include "gaze_calibration.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace eyetracker {
namespace {

struct Matrix3x3 {
    double m[3][3]{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
};

Matrix3x3 multiply_at_a(const std::vector<std::array<double, 3>>& rows) {
    Matrix3x3 result;
    for (const auto& r : rows) {
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                result.m[i][j] += r[i] * r[j];
            }
        }
    }
    return result;
}

std::array<double, 3> multiply_at_b(const std::vector<std::array<double, 3>>& rows,
                                    const std::vector<double>& targets) {
    std::array<double, 3> result{0.0, 0.0, 0.0};
    for (std::size_t idx = 0; idx < rows.size(); ++idx) {
        for (std::size_t i = 0; i < 3; ++i) {
            result[i] += rows[idx][i] * targets[idx];
        }
    }
    return result;
}

std::optional<Matrix3x3> invert(const Matrix3x3& m) {
    const double det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
                       m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
                       m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);

    if (std::fabs(det) < 1e-9) {
        return std::nullopt;
    }

    Matrix3x3 inv;
    inv.m[0][0] = (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) / det;
    inv.m[0][1] = (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) / det;
    inv.m[0][2] = (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) / det;

    inv.m[1][0] = (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) / det;
    inv.m[1][1] = (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) / det;
    inv.m[1][2] = (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]) / det;

    inv.m[2][0] = (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) / det;
    inv.m[2][1] = (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) / det;
    inv.m[2][2] = (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) / det;
    return inv;
}

std::array<double, 3> multiply(const Matrix3x3& m, const std::array<double, 3>& v) {
    std::array<double, 3> result{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            result[i] += m.m[i][j] * v[j];
        }
    }
    return result;
}

}  // namespace

void BrightPupilCalibrator::add_calibration_sample(const EyeFeatures& eye,
                                                   std::array<double, 2> screen_point) {
    eye_samples_.push_back(eye);
    screen_samples_.push_back(screen_point);
    result_.ready = false;
}

CalibrationResult BrightPupilCalibrator::compute() {
    if (eye_samples_.size() < 3) {
        throw std::runtime_error("至少需要3个样本才能求解标定矩阵");
    }

    std::vector<std::array<double, 3>> design;
    std::vector<double> target_x;
    std::vector<double> target_y;

    design.reserve(eye_samples_.size());
    target_x.reserve(eye_samples_.size());
    target_y.reserve(eye_samples_.size());

    for (std::size_t i = 0; i < eye_samples_.size(); ++i) {
        const auto feature = pupil_glint_vector(eye_samples_[i]);
        design.push_back({feature[0], feature[1], 1.0});
        target_x.push_back(screen_samples_[i][0]);
        target_y.push_back(screen_samples_[i][1]);
    }

    const auto ata = multiply_at_a(design);
    const auto atbx = multiply_at_b(design, target_x);
    const auto atby = multiply_at_b(design, target_y);

    const auto inv = invert(ata);
    if (!inv) {
        throw std::runtime_error("标定矩阵不可逆，样本可能共线");
    }

    const auto coeffx = multiply(*inv, atbx);
    const auto coeffy = multiply(*inv, atby);

    result_.ready = true;
    result_.coeff_x = coeffx;
    result_.coeff_y = coeffy;

    // 计算平均残差
    double errx = 0.0;
    double erry = 0.0;
    for (std::size_t i = 0; i < design.size(); ++i) {
        const double pred_x = coeffx[0] * design[i][0] + coeffx[1] * design[i][1] + coeffx[2];
        const double pred_y = coeffy[0] * design[i][0] + coeffy[1] * design[i][1] + coeffy[2];
        errx += std::fabs(pred_x - target_x[i]);
        erry += std::fabs(pred_y - target_y[i]);
    }

    result_.residual_error = {errx / design.size(), erry / design.size()};
    return result_;
}

std::optional<std::array<double, 2>> BrightPupilCalibrator::estimate_gaze(
    const EyeFeatures& eye) const {
    if (!result_.ready) {
        return std::nullopt;
    }
    const auto feature = pupil_glint_vector(eye);
    const double x = result_.coeff_x[0] * feature[0] + result_.coeff_x[1] * feature[1] +
                     result_.coeff_x[2];
    const double y = result_.coeff_y[0] * feature[0] + result_.coeff_y[1] * feature[1] +
                     result_.coeff_y[2];
    return std::array<double, 2>{x, y};
}

double BrightPupilCalibrator::estimate_blink_frequency(const std::vector<EyeFeatures>& frames,
                                                       double fps,
                                                       double blink_threshold) {
    if (frames.empty() || fps <= 0.0) {
        return 0.0;
    }

    bool currently_closed = false;
    int blink_count = 0;
    for (const auto& f : frames) {
        const bool closed = f.eyelid_opening < blink_threshold || f.pupil_radius <= 0.1;
        if (!currently_closed && closed) {
            currently_closed = true;
        }
        if (currently_closed && !closed) {
            ++blink_count;
            currently_closed = false;
        }
    }

    const double duration_seconds = static_cast<double>(frames.size()) / fps;
    const double blinks_per_minute = (duration_seconds > 0.0) ?
                                         (blink_count * 60.0 / duration_seconds) :
                                         0.0;
    return blinks_per_minute;
}

double BrightPupilCalibrator::pupil_diameter_mm(const EyeFeatures& eye, double pixel_to_mm) {
    if (pixel_to_mm <= 0.0) {
        throw std::invalid_argument("像素到毫米的换算系数必须大于0");
    }
    return 2.0 * eye.pupil_radius * pixel_to_mm;
}

std::array<double, 2> BrightPupilCalibrator::pupil_glint_vector(const EyeFeatures& eye) {
    if (eye.glints.empty()) {
        return {0.0, 0.0};
    }
    double cx = 0.0;
    double cy = 0.0;
    for (const auto& g : eye.glints) {
        cx += g[0];
        cy += g[1];
    }
    cx /= static_cast<double>(eye.glints.size());
    cy /= static_cast<double>(eye.glints.size());
    return {eye.pupil_center[0] - cx, eye.pupil_center[1] - cy};
}

std::string to_string(const CalibrationResult& result) {
    std::ostringstream oss;
    oss << (result.ready ? "标定完成" : "标定未完成") << "\n";
    oss << "X = " << result.coeff_x[0] << " * dx + " << result.coeff_x[1]
        << " * dy + " << result.coeff_x[2] << "\n";
    oss << "Y = " << result.coeff_y[0] << " * dx + " << result.coeff_y[1]
        << " * dy + " << result.coeff_y[2] << "\n";
    oss << "平均残差(px): (" << result.residual_error[0] << ", " << result.residual_error[1]
        << ")";
    return oss.str();
}

}  // namespace eyetracker

