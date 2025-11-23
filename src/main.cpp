#include "gaze_calibration.hpp"

#include <iostream>
#include <vector>

using eyetracker::BrightPupilCalibrator;
using eyetracker::CalibrationResult;
using eyetracker::EyeFeatures;

int main() {
    // 构造模拟标定数据：屏幕四个角加中心
    BrightPupilCalibrator calibrator;
    const std::vector<std::array<double, 2>> screen_points = {
        {0.0, 0.0}, {1920.0, 0.0}, {0.0, 1080.0}, {1920.0, 1080.0}, {960.0, 540.0}};

    const std::vector<EyeFeatures> samples = {
        EyeFeatures{{210.0, 130.0}, 22.0, {{180.0, 120.0}, {182.0, 118.0}}, 18.0},
        EyeFeatures{{115.0, 130.0}, 21.0, {{140.0, 120.0}, {138.0, 118.0}}, 17.0},
        EyeFeatures{{210.0, 240.0}, 20.0, {{182.0, 220.0}, {180.0, 218.0}}, 17.5},
        EyeFeatures{{115.0, 240.0}, 23.0, {{138.0, 220.0}, {140.0, 218.0}}, 18.0},
        EyeFeatures{{162.0, 185.0}, 21.5, {{160.0, 170.0}, {162.0, 168.0}}, 17.0},
    };

    for (std::size_t i = 0; i < samples.size(); ++i) {
        calibrator.add_calibration_sample(samples[i], screen_points[i]);
    }

    const CalibrationResult result = calibrator.compute();
    std::cout << eyetracker::to_string(result) << "\n\n";

    // 模拟一帧新的检测结果，估算视线落点
    const EyeFeatures new_frame{{175.0, 180.0}, 22.0, {{160.0, 170.0}, {162.0, 168.0}}, 18.5};
    if (auto gaze = calibrator.estimate_gaze(new_frame)) {
        std::cout << "估算视线坐标(px): (" << (*gaze)[0] << ", " << (*gaze)[1] << ")\n";
    }

    // 计算瞳孔直径（假设1像素=0.04mm）
    const double diameter = BrightPupilCalibrator::pupil_diameter_mm(new_frame, 0.04);
    std::cout << "瞳孔直径(mm): " << diameter << "\n";

    // 使用一段时间的帧序列估算眨眼频率
    std::vector<EyeFeatures> blink_sequence = samples;
    blink_sequence.insert(blink_sequence.end(), samples.begin(), samples.end());
    // 人为插入一段闭眼帧
    blink_sequence.push_back({{0.0, 0.0}, 0.0, {}, 2.0});
    blink_sequence.push_back({{0.0, 0.0}, 0.0, {}, 2.0});
    blink_sequence.push_back(samples.back());

    const double blink_per_minute =
        BrightPupilCalibrator::estimate_blink_frequency(blink_sequence, 30.0, 5.0);
    std::cout << "眨眼频率(次/分钟): " << blink_per_minute << "\n";

    return 0;
}

