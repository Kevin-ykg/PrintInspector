#include "refine.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
using Spectrum = std::array<float, 9>;
constexpr int darkLimit = 30; // 对原始灰度计数设阈值，显示对比度不参与计算。

float median(std::vector<float> &values)
{
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

// 八个对数光谱比例描述相对响应，第九项保留整体亮度以识别只在明暗上不同的墨层。
void descriptor(const std::array<const uchar *, 8> &rows, int x, const std::array<float, 256> &logLut,
                Spectrum &f)
{
    float sum = 0;
    for (int b = 0; b < 8; b++)
    {
        f[b] = logLut[rows[b][x]];
        sum += f[b];
    }
    f[8] = sum / 8;
    for (int b = 0; b < 8; b++)
        f[b] -= f[8];
}

// 网格中只统计光谱接近本件底布的采样点。
// 图案占据大部分网格时，回退到整件底布的亮度参照。
// 网格中心之间插值，用于跟随缓慢渐变的阴影。
cv::Mat clothBrightnessGrid(const std::array<cv::Mat, 8> &aligned, const cv::Mat &outside,
                            const std::array<float, 256> &logLut, const Spectrum &center,
                            const Spectrum &scale)
{
    constexpr int cell = 96;
    cv::Mat grid((outside.rows + cell - 1) / cell, (outside.cols + cell - 1) / cell, CV_32F,
                 cv::Scalar(center[8]));
    for (int gy = 0; gy < grid.rows; ++gy)
        for (int gx = 0; gx < grid.cols; ++gx)
        {
            // 将全幅均匀分块，使插值中心与实际采样块中心一致。
            int top = int(int64_t(gy) * outside.rows / grid.rows),
                bottom = int(int64_t(gy + 1) * outside.rows / grid.rows);
            int left = int(int64_t(gx) * outside.cols / grid.cols),
                right = int(int64_t(gx + 1) * outside.cols / grid.cols);
            std::vector<float> levels;
            int examined = 0;
            for (int y = top + 3; y < bottom; y += 8)
            {
                std::array<const uchar *, 8> rows;
                for (int b = 0; b < 8; ++b)
                    rows[b] = aligned[b].ptr<uchar>(y);
                for (int x = left + 3; x < right; x += 8)
                {
                    if (outside.at<uchar>(y, x))
                        continue;
                    ++examined;
                    Spectrum f;
                    descriptor(rows, x, logLut, f);
                    int similar = 0;
                    for (int b = 0; b < 8; ++b)
                        similar += std::abs(f[b] - center[b]) < 2.5f * scale[b];
                    if (similar >= 7 && std::abs(f[8] - center[8]) < 4.f * scale[8])
                        levels.push_back(f[8]);
                }
            }
            if (levels.size() >= 6 && levels.size() * 3 >= size_t(2 * examined))
                grid.at<float>(gy, gx) = median(levels);
        }
    return grid;
}

cv::Mat exterior(const std::array<cv::Mat, 8> &aligned, bool &ambiguous)
{
    cv::Mat maximum = aligned[0].clone();
    for (int b = 1; b < 8; b++)
        cv::max(maximum, aligned[b], maximum);
    cv::medianBlur(maximum, maximum, 3);
    cv::Mat dark = maximum <= darkLimit;
    // 若大部分像素都欠曝，无法可靠区分布面和台面，留给人工复核。
    ambiguous = cv::countNonZero(dark) > 0.78 * dark.total();
    if (ambiguous)
        return cv::Mat::zeros(dark.size(), CV_8U);
    cv::Mat padded;
    cv::copyMakeBorder(dark, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    cv::floodFill(padded, cv::Point(0, 0), cv::Scalar(128), nullptr, {}, {}, 4);
    cv::Mat core = padded(cv::Rect(1, 1, dark.cols, dark.rows)) == 128;
    // 布边的散焦和配准误差会使紧邻台面的像素混入布面信号。
    // 只向邻近暗像素扩展，限制台面掩膜侵入深色印花的距离。
    cv::Mat nearby, transition;
    cv::dilate(core, nearby, cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15}));
    cv::bitwise_and(nearby, maximum <= 52, transition);
    cv::bitwise_or(core, transition, core);
    return core;
}

// 当布边与画框重合时，外侧深色台面可能位于视野之外。
// 沿法线向布内采样，用亮度变化判断细长分量是否属于边缘阴影。
bool isFrameShadow(cv::Rect box, const std::array<cv::Mat, 8> &bands)
{
    const cv::Size size = bands[7].size();
    if (size.width < 48 || size.height < 48)
        return false;
    const bool vertical = box.height >= 3 * box.width && box.width <= 20 && box.height >= 16;
    const bool horizontal = box.width >= 3 * box.height && box.height <= 20 && box.width >= 16;
    cv::Point normal;
    if (vertical && box.x <= 2)
        normal = {1, 0};
    else if (vertical && box.br().x >= size.width - 2)
        normal = {-1, 0};
    else if (horizontal && box.y <= 2)
        normal = {0, 1};
    else if (horizontal && box.br().y >= size.height - 2)
        normal = {0, -1};
    else
        return false;
    int dark = 0, total = 0;
    const int span = vertical ? box.height : box.width;
    for (int i = 0; i < span; i += 3)
    {
        cv::Point edge = vertical ? cv::Point(normal.x > 0 ? 0 : size.width - 1, box.y + i)
                                  : cv::Point(box.x + i, normal.y > 0 ? 0 : size.height - 1);
        int agreeing = 0;
        for (int b : {5, 6, 7})
        {
            const auto &im = bands[b];
            int outer = im.at<uchar>(edge) + im.at<uchar>(edge + normal);
            int inner = im.at<uchar>(edge + 24 * normal) + im.at<uchar>(edge + 30 * normal);
            if (outer + 16 < .80 * (inner + 16))
                agreeing++;
        }
        dark += agreeing >= 2 && bands[7].at<uchar>(edge) <= 45;
        total++;
    }
    return dark >= .6 * total;
}

// 按完整连通域判断布边，同一条靠边带内也可能有真正的印花。
void cleanForeground(cv::Mat mask, const cv::Mat &evidence, const cv::Mat &outside, int threshold,
                     const std::array<cv::Mat, 8> &aligned, int noiseArea)
{
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (n <= 1)
        return;

    cv::Mat distance;
    if (cv::countNonZero(outside))
        cv::distanceTransform(255 - outside, distance, cv::DIST_L2, cv::DIST_MASK_3);
    cv::Mat thick;
    cv::erode(mask, thick, cv::getStructuringElement(cv::MORPH_RECT, {11, 11}));

    struct Support
    {
        int near = 0, contact = 0, thick = 0, score = 0;
        float minDistance = 73, maxDistance = 0;
    };
    std::vector<Support> support(n);
    for (int y = 0; y < mask.rows; ++y)
    {
        const int *ids = labels.ptr<int>(y);
        const uchar *votes = evidence.ptr<uchar>(y);
        const uchar *core = thick.ptr<uchar>(y);
        const float *dist = distance.empty() ? nullptr : distance.ptr<float>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            int id = ids[x];
            if (!id)
                continue;
            auto &v = support[id];
            if (stats.at<int>(id, cv::CC_STAT_AREA) <= noiseArea)
                v.score += votes[x];
            if (dist)
            {
                v.near += dist[x] <= 72;
                v.contact += dist[x] <= 20;
                v.minDistance = std::min(v.minDistance, dist[x]);
                v.maxDistance = std::max(v.maxDistance, std::min(73.f, dist[x]));
            }
            v.thick += core[x] != 0;
        }
    }
    // 分两轮判定。先用可靠大分量作邻域参照，再决定旁边的小点是否有足够证据。
    std::vector<uchar> keep(n, 255), anchor(n, 0);
    keep[0] = 0;
    for (int id = 1; id < n; id++)
    {
        const int *st = stats.ptr<int>(id);
        const auto &v = support[id];
        int area = st[cv::CC_STAT_AREA], span = std::max(st[cv::CC_STAT_WIDTH], st[cv::CC_STAT_HEIGHT]);
        // 沿布边延伸的分量到外部的距离变化小；横穿布边方向的笔画通常不满足此条件。
        bool followsEdge =
            v.contact > 0 && v.near >= 0.96 * area && v.maxDistance - v.minDistance <= 0.5 * span;
        bool slender = span >= 16 && double(span) * span >= 4.0 * area && v.thick <= 0.6 * area;
        if (slender && (followsEdge ||
                        (!distance.empty() && isFrameShadow(cv::Rect(st[0], st[1], st[2], st[3]), aligned))))
            keep[id] = 0;
        // 锚点必须大于去噪上限，否则候选小分量会在邻域判定时支持自己。
        else if (area >= std::max(128, noiseArea + 1))
            anchor[id] = 255;
    }
    cv::Mat nearby(mask.size(), CV_8U);
    for (int y = 0; y < mask.rows; y++)
    {
        const int *ids = labels.ptr<int>(y);
        uchar *dst = nearby.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++)
            dst[x] = anchor[ids[x]];
    }
    cv::dilate(nearby, nearby, cv::getStructuringElement(cv::MORPH_RECT, {33, 33}));
    for (int id = 1; id < n; id++)
    {
        const int *st = stats.ptr<int>(id);
        int area = st[cv::CC_STAT_AREA];
        if (!keep[id] || area > noiseArea)
            continue;
        bool adjacent = cv::countNonZero(nearby(cv::Rect(st[0], st[1], st[2], st[3]))) != 0;
        float mean = float(support[id].score) / area;
        if (area <= noiseArea / 3)
        {
            // 远离主体的孤点要求更高的面积或证据，笔画旁的小点可保留更多。
            int minArea = adjacent ? 5 : std::max(1, noiseArea / 4);
            int minScore = threshold + (adjacent ? 45 : 60);
            if (area < minArea || mean < minScore)
                keep[id] = 0;
        }
        else if (!adjacent && mean < threshold + 35)
            keep[id] = 0;
    }
    for (int y = 0; y < mask.rows; y++)
    {
        const int *ids = labels.ptr<int>(y);
        uchar *dst = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++)
            dst[x] = keep[ids[x]];
    }
}

void fillSmallHoles(cv::Mat mask, const cv::Mat &evidence, const cv::Mat &outside, int threshold,
                    const MaskSettings &settings)
{
    if (settings.holeArea == 0)
        return;
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask == 0, labels, stats, centroids, 8, CV_32S);
    for (int id = 1; id < n; id++)
    {
        const int *st = stats.ptr<int>(id);
        if (st[cv::CC_STAT_AREA] > settings.holeArea)
            continue;
        cv::Rect box(st[0], st[1], st[2], st[3]);
        if (box.x == 0 || box.y == 0 || box.br().x == mask.cols || box.br().y == mask.rows)
            continue;
        // 仅将当前孔的像素纳入均值，避免包围框内相邻笔画抬高证据。
        cv::Mat component = labels(box) == id;
        if (cv::countNonZero(outside(box) & component))
            continue;
        if (cv::mean(evidence(box), component)[0] >= threshold - settings.holeAllowance)
            mask(box).setTo(255, component);
    }
}
} // 匿名命名空间

Refinement refineMask(const std::array<cv::Mat, 8> &images, const std::array<cv::Point, 8> &offsets,
                      cv::Rect common, const cv::Mat &votes, int threshold, bool pcaRoute,
                      const MaskSettings &settings, cv::Mat mask)
{
    std::array<cv::Mat, 8> aligned;
    for (int b = 0; b < 8; b++)
        aligned[b] = images[b](common + offsets[b]);
    Refinement result;
    cv::Mat outside = exterior(aligned, result.weakBackground);
    result.excluded = cv::Mat(mask.size(), CV_8U, cv::Scalar(255));
    outside.copyTo(result.excluded(common));
    const cv::Mat input = votes(common);
    std::array<float, 256> logLut{};
    for (int i = 0; i < 256; i++)
        logLut[i] = std::log(i + 8.f);
    auto rowsAt = [&](int y) {
        std::array<const uchar *, 8> rows;
        for (int b = 0; b < 8; b++)
            rows[b] = aligned[b].ptr<uchar>(y);
        return rows;
    };
    std::array<std::vector<float>, 9> samples;
    for (int y = 4; y < input.rows; y += 16)
    {
        auto rows = rowsAt(y);
        for (int x = 4; x < input.cols; x += 16)
            if (!outside.at<uchar>(y, x))
            {
                Spectrum point;
                descriptor(rows, x, logLut, point);
                for (int i = 0; i < 9; i++)
                    samples[i].push_back(point[i]);
            }
    }
    // 本件底布的分布单独用于细化；它与训练期间冻结的 PCA 均值、尺度是两套统计量。
    Spectrum center{}, scale{};
    if (samples[0].size() < 64)
        result.weakBackground = true;
    if (!result.weakBackground)
    {
        for (int i = 0; i < 9; i++)
        {
            center[i] = median(samples[i]);
            for (float &v : samples[i])
                v = std::abs(v - center[i]);
            scale[i] = std::max(i == 8 ? .14f : .045f, 1.4826f * median(samples[i]));
        }
        // 多数波段离散时，单一底布参照无法代表画面中不同材料。
        int dispersed = 0;
        for (int b = 0; b < 8; b++)
            dispersed += scale[b] > .22f;
        result.weakBackground = dispersed >= 5;
    }
    cv::Mat clothGrid;
    if (pcaRoute && !result.weakBackground)
        clothGrid = clothBrightnessGrid(aligned, outside, logLut, center, scale);
    cv::Mat evidence = input.clone();
    cv::parallel_for_(cv::Range(0, input.rows), [&](const cv::Range &range) {
        cv::Mat brightnessRow;
        for (int y = range.start; y < range.end; y++)
        {
            auto rows = rowsAt(y);
            auto out = evidence.ptr<uchar>(y);
            if (!clothGrid.empty())
            {
                // 逐行插值避免为每个线程分配全幅浮点亮度图。
                float fy =
                    std::clamp((y + .5f) * clothGrid.rows / input.rows - .5f, 0.f, float(clothGrid.rows - 1));
                int iy = int(fy);
                cv::Mat row;
                cv::addWeighted(clothGrid.row(iy), 1 - (fy - iy),
                                clothGrid.row(std::min(iy + 1, clothGrid.rows - 1)), fy - iy, 0, row);
                cv::resize(row, brightnessRow, {input.cols, 1}, 0, 0, cv::INTER_LINEAR);
            }
            for (int x = 0; x < input.cols; x++)
            {
                if (outside.at<uchar>(y, x))
                {
                    out[x] = 0;
                    continue;
                }
                // 只在白光、红外和紫外均相似的四邻域内平滑树票，减少跨边缘混合。
                int total = 4 * input.at<uchar>(y, x), weight = 4;
                for (const auto delta : std::array<cv::Point, 4>{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}})
                {
                    int xx = x + delta.x, yy = y + delta.y;
                    if (unsigned(xx) >= unsigned(input.cols) || unsigned(yy) >= unsigned(input.rows) ||
                        outside.at<uchar>(yy, xx))
                        continue;
                    int change = 0;
                    for (int b : {0, 6, 7})
                        change = std::max(change, std::abs(rows[b][x] - aligned[b].at<uchar>(yy, xx)));
                    if (change <= 12)
                    {
                        total += input.at<uchar>(yy, xx);
                        weight++;
                    }
                }
                float value = float(total) / weight;
                if (!result.weakBackground)
                {
                    Spectrum f;
                    descriptor(rows, x, logLut, f);
                    float first = 0, second = 0;
                    int peakBand = 0;
                    for (int b = 0; b < 8; b++)
                    {
                        float z = std::abs(f[b] - center[b]) / scale[b];
                        if (z > first)
                        {
                            second = first;
                            first = z;
                            peakBand = b;
                        }
                        else
                            second = std::max(second, z);
                    }
                    float distance = std::max(second, std::abs(f[8] - center[8]) / scale[8]);
                    // 即使树票偏低，也允许足够强的光谱差异提供前景证据。
                    float support = std::clamp((distance - 1.8f) / 3.2f, 0.f, 1.f);
                    value = .54f * value + .46f * (threshold - 100 + 200 * support);
                    // 浅色墨可能主要在一个高信号波段出现偏离，还要求次强波段、
                    // 整体亮度和原始票数同时支持，才恢复为前景。
                    if (!pcaRoute && rows[peakBand][x] >= 32 && first >= 5.f && second >= 1.1f &&
                        f[8] > center[8] + .8f * scale[8] && input.at<uchar>(y, x) >= threshold / 2)
                        value = std::max(value, float(threshold + 24));
                    if (!brightnessRow.empty())
                    {
                        float clothLevel = brightnessRow.ptr<float>()[x];
                        float shade = (center[8] - clothLevel) / scale[8];
                        float localDeviation = std::abs(f[8] - clothLevel) / scale[8];
                        // 当局部渐暗区域与周围底布具有相同光谱比例时，PCA 仍可能给出高票。
                        // 仅在局部参照也支持阴影解释时降低得分。
                        if (shade > .7f && localDeviation < 1.8f && first < 4.8f && second < 3.0f)
                            value -= 75.f * std::clamp(shade - .5f, 0.f, 1.f);
                    }
                    if (distance >= 6.5f)
                        value = std::max(value, float(threshold + 24));
                }
                out[x] = cv::saturate_cast<uchar>(value);
            }
        }
    });
    mask.setTo(0);
    cv::Mat local = mask(common);
    // 调试漏笔可在此比较 input、evidence 与 local，再单步观察连通域清理和补洞。
    cv::compare(evidence, threshold, local, cv::CMP_GE);
    local.setTo(0, outside);
    cleanForeground(local, evidence, outside, threshold, aligned, settings.noiseArea);
    fillSmallHoles(local, evidence, outside, threshold, settings);
    return result;
}
