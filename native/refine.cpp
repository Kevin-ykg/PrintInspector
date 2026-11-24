#include "refine.hpp"
#include <intrin.h>
#include <stdexcept>
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

// 织纹高光可与墨层一样获得高票。只对已通过面积和得分检查的微小紧凑分量，
// 比较它与周围多个独立布块的八通道响应；真实墨点在周围找不到相同的亮峰。
bool repeatsClothHighlight(int id, cv::Rect box, const cv::Mat &labels, const cv::Mat &stats,
                           const cv::Mat &outside, const std::array<cv::Mat, 8> &aligned)
{
    const int area = stats.at<int>(id, cv::CC_STAT_AREA);
    // 仅处理织纹亮峰尺度；较大的印点和延伸的细线沿用原有证据判定。
    if (area > 16 || box.width > 5 || box.height > 5)
        return false;
    const cv::Rect bounds(0, 0, labels.cols, labels.rows);
    // 隔一个背景像素便接到可靠笔画的分量可能是笔画边缘，保留它。
    const cv::Rect halo = cv::Rect(box.x - 2, box.y - 2, box.width + 4, box.height + 4) & bounds;
    for (int y = halo.y; y < halo.br().y; ++y)
        for (int x = halo.x; x < halo.br().x; ++x)
        {
            int neighbor = labels.ptr<int>(y)[x];
            if (!neighbor || stats.at<int>(neighbor, cv::CC_STAT_AREA) < 128)
                continue;
            for (int yy = std::max(box.y, y - 2); yy < std::min(box.br().y, y + 3); ++yy)
                for (int xx = std::max(box.x, x - 2); xx < std::min(box.br().x, x + 3); ++xx)
                    if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= 4 && labels.ptr<int>(yy)[xx] == id)
                        return false;
        }

    std::array<float, 8> center{}, tolerance{};
    for (int band = 0; band < 8; ++band)
    {
        for (int y = box.y; y < box.br().y; ++y)
            for (int x = box.x; x < box.br().x; ++x)
                if (labels.ptr<int>(y)[x] == id)
                    center[band] += aligned[band].ptr<uchar>(y)[x];
        center[band] /= area;
        tolerance[band] = 12.f + .18f * center[band];
    }

    // 以点为中心，检查外围八块互不重叠的布。排除台面、笔画和其他成形墨点。
    // 每块需有足够底布、平均白光明显低于候选亮点，同时至少两处像素光谱吻合。
    constexpr int side = 24;
    const cv::Point origin(box.x + box.width / 2 - side / 2, box.y + box.height / 2 - side / 2);
    int agreeing = 0;
    for (int dy : {-side, 0, side})
        for (int dx : {-side, 0, side})
        {
            if (dx == 0 && dy == 0)
                continue;
            const cv::Rect patch(origin + cv::Point(dx, dy), cv::Size(side, side));
            if ((patch & bounds) != patch)
                continue;
            int cloth = 0, whiteSum = 0, matches = 0;
            for (int y = patch.y; y < patch.br().y; ++y)
            {
                const int *ids = labels.ptr<int>(y);
                const uchar *excluded = outside.ptr<uchar>(y);
                std::array<const uchar *, 8> rows;
                for (int b = 0; b < 8; ++b)
                    rows[b] = aligned[b].ptr<uchar>(y);
                for (int x = patch.x; x < patch.br().x; ++x)
                {
                    if (excluded[x] || (ids[x] && stats.at<int>(ids[x], cv::CC_STAT_AREA) > 3))
                        continue;
                    cloth++;
                    whiteSum += rows[7][x];
                    if (matches >= 2 || std::abs(rows[7][x] - center[7]) > tolerance[7])
                        continue;
                    bool similar = true;
                    for (int b = 0; b < 7 && similar; ++b)
                        similar = std::abs(rows[b][x] - center[b]) <= tolerance[b];
                    matches += similar;
                }
            }
            if (cloth >= patch.area() / 2 && matches >= 2 && center[7] * cloth >= whiteSum + 16 * cloth)
                if (++agreeing == 2)
                    return true;
        }
    return false;
}

// 按完整连通域判断布边，同一条靠边带内也可能有真正的印花。
void cleanForeground(cv::Mat mask, const cv::Mat &evidence, const cv::Mat &outside, int threshold,
                     const std::array<cv::Mat, 8> &aligned, int noiseArea, bool checkWeave)
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
    distance.release();
    thick.release();
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
        if (keep[id] && checkWeave &&
            repeatsClothHighlight(id, cv::Rect(st[0], st[1], st[2], st[3]), labels, stats, outside, aligned))
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

// 布面延伸到画框时，各照明通道的边缘衰减不一定同步。只在内侧有明确背景、
// 近红外和白光连续渐暗而紫外仍有响应的条带内，核对候选像素与当地布面的光谱。
void suppressEdgeShading(cv::Mat mask, const std::array<cv::Mat, 8> &bands, const cv::Mat &outside)
{
    constexpr int cells = 16;
    const int step = std::clamp(mask.cols / 240, 8, 32);
    if (mask.cols < cells * step * 4 || mask.rows < 4 * step)
        return;
    std::vector<uchar> values;
    for (int y = 0; y < mask.rows; y += step)
        for (bool right : {false, true})
        {
            auto column = [&](int distance) { return right ? mask.cols - 1 - distance : distance; };
            int height = std::min(step, mask.rows - y);
            int anchorX = right ? mask.cols - cells * step : (cells - 2) * step;
            cv::Rect anchor(anchorX, y, 2 * step, height);
            if (cv::countNonZero(mask(anchor)) > .1 * anchor.area() || cv::countNonZero(outside(anchor)))
                continue;
            cv::Rect strip(right ? mask.cols - cells * step : 0, y, cells * step, height);
            if (!cv::countNonZero(mask(strip)))
                continue;

            std::array<std::array<int, 8>, cells> profile{};
            for (int k = 0; k < cells; ++k)
                for (int b = 0; b < 8; ++b)
                {
                    values.clear();
                    // 纵向邻近布块共同估计中位值，减少织纹和细小异物对光谱参照的影响。
                    for (int yy = std::max(0, y - 5 * step); yy < std::min(mask.rows, y + 6 * step); yy += 7)
                        for (int dx = 2; dx < step; dx += 4)
                            values.push_back(bands[b].ptr<uchar>(yy)[column(k * step + dx)]);
                    auto mid = values.begin() + values.size() / 2;
                    std::nth_element(values.begin(), mid, values.end());
                    profile[k][b] = *mid;
                }
            const auto &edge = profile.front(), &inner = profile.back();
            if (edge[0] < 18 || edge[0] < inner[0] * .9 || edge[7] > inner[7] * .87)
                continue;
            bool gradual = true;
            for (int b : {5, 6, 7})
            {
                int drop = inner[b] - edge[b], increasing = 0;
                for (int k = 1; k < cells; ++k)
                {
                    int change = profile[k][b] - profile[k - 1][b];
                    gradual &= change >= -5 && change <= std::max(8., drop * .55);
                    increasing += change > 0;
                }
                gradual &= drop >= 10 && increasing >= 4;
            }
            if (!gradual)
                continue;
            for (int yy = y; yy < y + height; ++yy)
            {
                uchar *out = mask.ptr<uchar>(yy);
                for (int d = 0; d < (cells - 2) * step; ++d)
                {
                    int x = column(d);
                    if (!out[x])
                        continue;
                    const auto &cloth = profile[d / step];
                    bool matches = true;
                    for (int b = 0; b < 8 && matches; ++b)
                        matches = std::abs(bands[b].ptr<uchar>(yy)[x] - cloth[b]) <= 8 + .18 * cloth[b];
                    if (matches)
                        out[x] = 0;
                }
            }
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
// 中心像素翻转前后，八连通前景与四连通背景都应保持连通关系。
// 八邻域仅有 256 种组合；在首次调用时用 3×3 连通域枚举生成查找表。
const std::array<bool, 256> &simplePointTable()
{
    static const auto table = [] {
        std::array<bool, 256> allowed{};
        for (int bits = 0; bits < 256; ++bits)
        {
            cv::Mat fg(3, 3, CV_8U, cv::Scalar(0));
            int bit = 0;
            for (int i = 0; i < 9; ++i)
                if (i != 4)
                    fg.ptr<uchar>()[i] = (bits >> bit++) & 1;
            cv::Mat labels;
            if (cv::connectedComponents(fg, labels, 8) != 2)
                continue;
            cv::Mat bg = 1 - fg;
            bg.at<uchar>(1, 1) = 0;
            cv::connectedComponents(bg, labels, 4);
            int adjacent = 0;
            bool same = true;
            for (int i : {1, 3, 5, 7})
            {
                int id = labels.ptr<int>()[i];
                if (id)
                {
                    same &= !adjacent || adjacent == id;
                    adjacent = id;
                }
            }
            allowed[bits] = same && adjacent;
        }
        return allowed;
    }();
    return table;
}

// 在掩膜边缘沿法线比较 W 的内外灰度平台。只有两侧可辨且稳定时，
// 才用两侧亮度的中点定位光学边界；暗墨和亮墨均适用。
void fitWhiteBoundary(cv::Mat mask, const cv::Mat &white, const cv::Mat &outside, const int *cancel)
{
    constexpr int margin = 8;
    if (mask.rows <= 2 * margin || mask.cols <= 2 * margin)
        return;
    cv::Mat shape, gray, proposed = mask.clone();
    cv::GaussianBlur(mask, shape, {9, 9}, 2., 2., cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    cv::GaussianBlur(white, gray, {5, 5}, 1., 1., cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    cv::parallel_for_(cv::Range(margin, mask.rows - margin), [&](const cv::Range &range) {
        for (int y = range.start; y < range.end; ++y)
        {
            if (cancel && _InterlockedCompareExchange((volatile long *)cancel, 0, 0))
                return;
            const uchar *level = shape.ptr<uchar>(y), *above = shape.ptr<uchar>(y - 1),
                        *below = shape.ptr<uchar>(y + 1), *excluded = outside.ptr<uchar>(y);
            uchar *dst = proposed.ptr<uchar>(y);
            for (int x = margin; x < mask.cols - margin; ++x)
            {
                if (level[x] < 18 || level[x] > 237 || excluded[x])
                    continue;
                float dx = float(level[x + 1] - level[x - 1]), dy = float(below[x] - above[x]);
                float length = std::sqrt(dx * dx + dy * dy);
                if (length < 32)
                    continue;
                dx /= length;
                dy /= length;
                const cv::Point p(x, y);
                cv::Point near(cvRound(4 * dx), cvRound(4 * dy)), far(cvRound(7 * dx), cvRound(7 * dy));
                // 深入两侧后必须是实心前景和明确背景；细线、小点、窄孔及台面附近缺少参照时保留原判。
                if (shape.at<uchar>(p + near) < 236 || shape.at<uchar>(p + far) < 246 ||
                    shape.at<uchar>(p - near) > 19 || shape.at<uchar>(p - far) > 9 ||
                    outside.at<uchar>(p - near) || outside.at<uchar>(p - far))
                    continue;
                const int in1 = gray.at<uchar>(p + near), in2 = gray.at<uchar>(p + far);
                const int out1 = gray.at<uchar>(p - near), out2 = gray.at<uchar>(p - far);
                const float inside = .5f * (in1 + in2), background = .5f * (out1 + out2);
                const float contrast = std::abs(inside - background);
                if (contrast < 24 || std::max(std::abs(in1 - in2), std::abs(out1 - out2)) > .35f * contrast)
                    continue;
                dst[x] =
                    std::abs(gray.at<uchar>(p) - inside) < std::abs(gray.at<uchar>(p) - background) ? 255 : 0;
            }
        }
    });
    shape.release();
    gray.release();
    std::vector<cv::Point> changes;
    cv::findNonZero(proposed != mask, changes);
    const auto &allowed = simplePointTable();
    // 从旧边界逐点移动，避免一批各自可删的点同时消失后割断笔画。只遍历有修改提议的位置。
    for (int pass = 0; pass < 6; ++pass)
    {
        if (cancel && _InterlockedCompareExchange((volatile long *)cancel, 0, 0))
            return;
        bool moved = false;
        for (cv::Point p : changes)
        {
            uchar &value = mask.at<uchar>(p);
            if (value == proposed.at<uchar>(p))
                continue;
            int bits = 0, index = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                const uchar *row = mask.ptr<uchar>(p.y + dy) + p.x;
                for (int dx = -1; dx <= 1; ++dx)
                    if (dy || dx)
                        bits |= (row[dx] != 0) << index++;
            }
            if (allowed[bits])
            {
                value = proposed.at<uchar>(p);
                moved = true;
            }
        }
        if (!moved)
            break;
        std::reverse(changes.begin(), changes.end());
    }
}
} // 匿名命名空间

cv::Mat detectDarkExterior(const std::array<cv::Mat, 8> &aligned, bool &ambiguous, bool requireDarkCore)
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
    if (requireDarkCore)
    {
        // 自适应布底的拟合也依赖这个屏蔽区。暗布大面积连到画框时，单靠 30 灰度会误删布面。
        // 要求候选区至少一半像素仍落在更低的亮度范围，防止较宽阈值吞并只有微弱反射的布料。
        const int area = cv::countNonZero(core);
        if (!area)
            return core;
        if (2 * cv::countNonZero(core & (maximum <= 20)) < area)
        {
            ambiguous = true;
            core.setTo(0);
            return core;
        }
    }
    // 布边的散焦和配准误差会使紧邻台面的像素混入布面信号。
    // 只向邻近暗像素扩展，限制台面掩膜侵入深色印花的距离。
    cv::Mat nearby, transition;
    cv::dilate(core, nearby, cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15}));
    cv::bitwise_and(nearby, maximum <= 52, transition);
    cv::bitwise_or(core, transition, core);
    return core;
}

Refinement refineMask(const std::array<cv::Mat, 8> &images, const std::array<cv::Point, 8> &offsets,
                      cv::Rect common, const cv::Mat &votes, int threshold, bool pcaRoute,
                      const MaskSettings &settings, cv::Mat mask, const int *cancel)
{
    std::array<cv::Mat, 8> aligned;
    for (int b = 0; b < 8; b++)
        aligned[b] = images[b](common + offsets[b]);
    Refinement result;
    cv::Mat outside = detectDarkExterior(aligned, result.weakBackground);
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
    auto cancelled = [&] {
        return cancel && _InterlockedCompareExchange((volatile long *)cancel, 0, 0) != 0;
    };
    cv::Mat evidence(input.size(), CV_8U);
    cv::parallel_for_(cv::Range(0, input.rows), [&](const cv::Range &range) {
        cv::Mat brightnessRow;
        for (int y = range.start; y < range.end; y++)
        {
            auto rows = rowsAt(y);
            if (cancelled())
                return;
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
            // 每行只取得一次邻行的树票、屏蔽区和光谱指针，避免在四邻域循环中反复访问矩阵头。
            struct Neighbor
            {
                int dx;
                const uchar *votes, *excluded, *uv, *ir, *white;
            };
            Neighbor neighbors[4];
            int count = 0;
            for (const auto delta : std::array<cv::Point, 4>{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}})
            {
                const int yy = y + delta.y;
                if (unsigned(yy) < unsigned(input.rows))
                    neighbors[count++] = {delta.x,
                                          input.ptr<uchar>(yy),
                                          outside.ptr<uchar>(yy),
                                          aligned[0].ptr<uchar>(yy),
                                          aligned[6].ptr<uchar>(yy),
                                          aligned[7].ptr<uchar>(yy)};
            }
            const auto *rawVotes = input.ptr<uchar>(y);
            const auto *excluded = outside.ptr<uchar>(y);
            for (int x = 0; x < input.cols; x++)
            {
                if (excluded[x])
                {
                    out[x] = 0;
                    continue;
                }
                int total = 4 * rawVotes[x], weight = 4;
                for (int k = 0; k < count; k++)
                {
                    const auto &neighbor = neighbors[k];
                    const int xx = x + neighbor.dx;
                    if (unsigned(xx) < unsigned(input.cols) && !neighbor.excluded[xx] &&
                        std::abs(rows[0][x] - neighbor.uv[xx]) <= 12 &&
                        std::abs(rows[6][x] - neighbor.ir[xx]) <= 12 &&
                        std::abs(rows[7][x] - neighbor.white[xx]) <= 12)
                    {
                        total += neighbor.votes[xx];
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
                        f[8] > center[8] + .8f * scale[8] && rawVotes[x] >= threshold / 2)
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
    if (cancelled())
        throw std::runtime_error("Cancelled");
    mask.setTo(0);
    cv::Mat local = mask(common);
    // 调试漏笔可在此比较 input、evidence 与 local，再单步观察连通域清理和补洞。
    cv::compare(evidence, threshold, local, cv::CMP_GE);
    local.setTo(0, outside);
    suppressEdgeShading(local, aligned, outside);
    cleanForeground(local, evidence, outside, threshold, aligned, settings.noiseArea, !pcaRoute);
    if (cancelled())
        throw std::runtime_error("Cancelled");
    fillSmallHoles(local, evidence, outside, threshold, settings);
    evidence.release();
    fitWhiteBoundary(local, aligned[7], outside, cancel);
    if (common.size() == mask.size())
        result.excluded = outside;
    else
    {
        result.excluded = cv::Mat(mask.size(), CV_8U, cv::Scalar(255));
        outside.copyTo(result.excluded(common));
    }
    return result;
}
