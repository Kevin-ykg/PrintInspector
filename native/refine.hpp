#pragma once
#include <opencv2/core.hpp>
#include <array>

// excluded 使用全幅 W 坐标，非零表示未共同覆盖或被判为外部台面，用于生成触边标志。
struct Refinement
{
    cv::Mat excluded;
    bool weakBackground = false;
};
// 面积均按原分辨率的像素数计算。默认组合与首版 ABI 的空间清理行为一致。
struct MaskSettings
{
    int noiseArea = 96;
    int holeArea = 80;
    int holeAllowance = 46;
};

// bands 是原始采集帧；在 common 中以 W 像素加 offsets[b] 访问对应波段。
// votes 保存原始树票；mask 借用调用方全幅缓冲并写入最终 0/255 分类。
Refinement refineMask(const std::array<cv::Mat, 8> &bands, const std::array<cv::Point, 8> &offsets,
                      cv::Rect common, const cv::Mat &votes, int threshold, bool pcaRoute,
                      const MaskSettings &settings, cv::Mat mask, const int *cancel);

// 仅屏蔽与画框相连且所有波段均暗的区域。整体欠曝时置 ambiguous 并返回空屏蔽区。
cv::Mat detectDarkExterior(const std::array<cv::Mat, 8> &aligned, bool &ambiguous,
                           bool requireDarkCore = false);
