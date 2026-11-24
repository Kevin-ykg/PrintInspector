#include "segmentation.h"
#include <opencv2/core.hpp>
#include <intrin.h>
#include <stdexcept>
#include "refinement.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace spectral
{
namespace
{

constexpr int channels = 8;
constexpr int terms = 10;
using Spectrum = std::array<float, channels>;
using Bands = std::array<const std::uint8_t *, channels>;
using Basis = std::array<double, terms>;

template <size_t N> using Matrix = std::array<std::array<double, N>, N>;

// 对已正则化的正规方程矩阵或协方差矩阵做原位 Cholesky 分解，下三角部分保存分解结果。
template <size_t N> void cholesky(Matrix<N> &a)
{
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            double sum = a[i][j];
            for (size_t k = 0; k < j; ++k)
            {
                sum -= a[i][k] * a[j][k];
            }
            a[i][j] = i == j ? std::sqrt(std::max(sum, 1e-12)) : sum / a[j][j];
        }
    }
}

float median(std::vector<float> &values)
{
    if (values.empty())
    {
        return 0;
    }
    const auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

// 每个节点保存采样值、预测布底和残差；weight 表示它参与布底估计的可信程度。
struct Tile
{
    Basis basis{};
    Spectrum observed{};
    Spectrum expected{};
    Spectrum residual{};
    double weight = 0;
    bool fit_anchor = true;
    bool outside = false;
    bool isolated_band = false;
};

struct Grid
{
    int columns, rows;
    std::vector<Tile> tiles;
    std::vector<int> xs, ys;

    Tile &at(int x, int y)
    {
        return tiles[static_cast<size_t>(y) * columns + x];
    }
    const Tile &at(int x, int y) const
    {
        return tiles[static_cast<size_t>(y) * columns + x];
    }
};

// 相邻两个网格节点及其插值权重，用于在不等距网格上还原像素位置的布底响应。
struct Axis
{
    int left, right;
    float fraction;
};

// 镜头边缘的照度下降较快，因此在视场边缘加密布底采样。节点位置使用原始相机像素坐标。
std::vector<int> grid_nodes(int size, int count)
{
    if (count == 1)
    {
        return {size / 2};
    }
    std::vector<int> nodes;
    for (int i = 0; i + 1 < count; ++i)
    {
        const int edge_distance = std::min(i, count - 2 - i);
        const int subdivisions = edge_distance == 0 ? 12 : (edge_distance < 3 ? 6 : 1);
        for (int k = 0; k < subdivisions; ++k)
        {
            const double t = (i + static_cast<double>(k) / subdivisions) / (count - 1);
            const int coordinate = static_cast<int>(std::lround(t * (size - 1)));
            if (nodes.empty() || nodes.back() < coordinate)
            {
                nodes.push_back(coordinate);
            }
        }
    }
    nodes.push_back(size - 1);
    return nodes;
}

Axis grid_position(int pixel, const std::vector<int> &nodes)
{
    const auto next = std::upper_bound(nodes.begin(), nodes.end(), pixel);
    const int right = std::min(static_cast<int>(nodes.size()) - 1, static_cast<int>(next - nodes.begin()));
    const int left = next == nodes.end() ? right : std::max(0, right - 1);
    const float fraction =
        left == right ? 0 : static_cast<float>(pixel - nodes[left]) / (nodes[right] - nodes[left]);
    return {left, right, fraction};
}

int local_spacing(const std::vector<int> &nodes, int i, int length)
{
    if (nodes.size() == 1)
    {
        return length;
    }
    const int left = std::max(0, i - 1), right = std::min(static_cast<int>(nodes.size()) - 1, i + 1);
    return std::max(1, (nodes[right] - nodes[left]) / (right - left));
}

Spectrum interpolate(const Grid &grid, const Axis &x, const Axis &y)
{
    Spectrum value{};
    const auto &a = grid.at(x.left, y.left).expected;
    const auto &b = grid.at(x.right, y.left).expected;
    const auto &c = grid.at(x.left, y.right).expected;
    const auto &d = grid.at(x.right, y.right).expected;
    for (int k = 0; k < channels; ++k)
    {
        const float upper = a[k] + x.fraction * (b[k] - a[k]);
        const float lower = c[k] + x.fraction * (d[k] - c[k]);
        value[k] = upper + y.fraction * (lower - upper);
    }
    return value;
}

struct Image
{
    const Bands &bands;
    int width, height, radius;
    std::vector<float> log_table;

    Image(const Bands &input, int w, int h)
        : bands(input), width(w), height(h),
          radius(std::min(w, h) >= 1600 ? 2 : (std::min(w, h) >= 500 ? 1 : 0))
    {
        const int side = radius * 2 + 1;
        const int area = side * side;
        log_table.resize(255 * area + 1);
        for (size_t i = 0; i < log_table.size(); ++i)
        {
            // 暗曝光先加常量再取对数，避免微小灰度波动被过度放大。
            log_table[i] = std::log2(16.0f + static_cast<float>(i) / area);
        }
    }

    Spectrum sample(int x, int y) const
    {
        Spectrum s{};
        for (int b = 0; b < channels; ++b)
        {
            int sum = 0;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                const auto *row = bands[b] + static_cast<size_t>(std::clamp(y + dy, 0, height - 1)) * width;
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    sum += row[std::clamp(x + dx, 0, width - 1)];
                }
            }
            s[b] = log_table[sum];
        }
        return s;
    }
};

// 每个网格节点从邻域分层取 7×7 个样本。逐波段中位数能降低少量细笔画对布底估计的影响。
Grid sample_tiles(const Image &image, const std::uint8_t *outside)
{
    const int step = std::max(8, std::min(image.width, image.height) / 40);
    const int base_x = std::min(96, (image.width - 1) / step + 1);
    const int base_y = std::min(96, (image.height - 1) / step + 1);
    auto xs = grid_nodes(image.width, base_x);
    auto ys = grid_nodes(image.height, base_y);
    const int nx = static_cast<int>(xs.size()), ny = static_cast<int>(ys.size());
    Grid grid{nx, ny, std::vector<Tile>(nx * ny), std::move(xs), std::move(ys)};
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            const double u = static_cast<double>(grid.xs[i]) / std::max(1, image.width - 1);
            const double v = static_cast<double>(grid.ys[j]) / std::max(1, image.height - 1);
            const int step_x = local_spacing(grid.xs, i, image.width);
            const int step_y = local_spacing(grid.ys, j, image.height);
            const double x = 2 * u - 1, y = 2 * v - 1;
            auto &tile = grid.at(i, j);
            // 全局曲面仅使用原始等距节点拟合，额外的边缘节点留给局部修正，避免边缘过度加权。
            const auto uniform_node = [](double fraction, int count, int length, int pixel) {
                if (count == 1)
                {
                    return true;
                }
                const double nearest = std::round(fraction * (count - 1));
                return std::lround(nearest * (length - 1) / (count - 1)) == pixel;
            };
            tile.fit_anchor = uniform_node(u, base_x, image.width, grid.xs[i]) &&
                              uniform_node(v, base_y, image.height, grid.ys[j]);
            tile.basis = {1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y};
            if (base_x < 4 || base_y < 4)
            {
                tile.basis = {1};
            }
            tile.weight = nx < 4 || ny < 4 || std::max(std::abs(x), std::abs(y)) > 0.72 ? 1 : 0;
            std::array<std::array<float, 49>, channels> patch{};
            int sampled = 0;
            for (int dy = -3; dy <= 3; ++dy)
            {
                for (int dx = -3; dx <= 3; ++dx)
                {
                    const int px =
                        std::clamp(static_cast<int>(std::lround(u * (image.width - 1) + dx * step_x / 7.0)),
                                   0, image.width - 1);
                    const int py =
                        std::clamp(static_cast<int>(std::lround(v * (image.height - 1) + dy * step_y / 7.0)),
                                   0, image.height - 1);
                    if (outside[static_cast<size_t>(py) * image.width + px])
                        continue;
                    const auto sample = image.sample(px, py);
                    for (int b = 0; b < channels; ++b)
                    {
                        patch[b][sampled] = sample[b];
                    }
                    ++sampled;
                }
            }
            if (sampled < 25 || outside[static_cast<size_t>(grid.ys[j]) * image.width + grid.xs[i]])
            {
                tile.outside = true;
                tile.fit_anchor = false;
                tile.weight = 0;
                continue;
            }
            for (int b = 0; b < channels; ++b)
            {
                auto middle = patch[b].begin() + sampled / 2;
                std::nth_element(patch[b].begin(), middle, patch[b].begin() + sampled);
                tile.observed[b] = *middle;
            }
        }
    }
    // 衣物没有延伸到四边时，改用仍有有效样本的节点初始化裸布估计。
    const auto anchors = std::count_if(grid.tiles.begin(), grid.tiles.end(),
                                     [](const Tile &t) { return t.fit_anchor && t.weight > 0; });
    if (anchors < 10)
        for (auto &tile : grid.tiles)
            tile.weight = tile.fit_anchor ? 1 : 0;
    return grid;
}

// 按波段拟合对数亮度曲面。八个波段共享每个节点的权重，使异常亮墨和暗墨都能被降权。
Spectrum fit_background(Grid &grid)
{
    Spectrum scale{};
    for (int pass = 0; pass < 7; ++pass)
    {
        Matrix<terms> normal{};
        std::array<std::array<double, channels>, terms> coefficients{};
        for (auto &t : grid.tiles)
        {
            if (!t.fit_anchor)
            {
                continue;
            }
            for (int i = 0; i < terms; ++i)
            {
                const double w = t.weight * t.basis[i];
                for (int j = 0; j <= i; ++j)
                {
                    normal[i][j] += w * t.basis[j];
                }
                for (int b = 0; b < channels; ++b)
                {
                    coefficients[i][b] += w * t.observed[b];
                }
            }
        }
        for (int i = 0; i < terms; ++i)
        {
            normal[i][i] += 1e-5;
        }
        cholesky(normal);
        for (int b = 0; b < channels; ++b)
        {
            for (int i = 0; i < terms; ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    coefficients[i][b] -= normal[i][j] * coefficients[j][b];
                }
                coefficients[i][b] /= normal[i][i];
            }
            for (int i = terms - 1; i >= 0; --i)
            {
                for (int j = i + 1; j < terms; ++j)
                {
                    coefficients[i][b] -= normal[j][i] * coefficients[j][b];
                }
                coefficients[i][b] /= normal[i][i];
            }
        }
        scale.fill(0);
        int count = 0;
        for (auto &t : grid.tiles)
        {
            for (int b = 0; b < channels; ++b)
            {
                double fitted = 0;
                for (int i = 0; i < terms; ++i)
                {
                    fitted += t.basis[i] * coefficients[i][b];
                }
                t.expected[b] = static_cast<float>(fitted);
                t.residual[b] = t.observed[b] - t.expected[b];
                if (t.fit_anchor && t.weight > .45)
                {
                    scale[b] += std::abs(t.residual[b]);
                }
            }
            count += t.fit_anchor && t.weight > .45;
        }
        for (auto &s : scale)
        {
            s = std::max(.024f, s / std::max(1, count));
        }
        for (auto &t : grid.tiles)
        {
            if (t.outside)
                continue;
            double squared = 0;
            for (int b = 0; b < channels; ++b)
            {
                const double normalized = std::min(10.f, std::abs(t.residual[b]) / scale[b]);
                squared += normalized * normalized;
            }
            const double ratio = squared / (4 * channels);
            t.weight = 1 / (1 + ratio * ratio * ratio);
        }
    }
    return scale;
}

// 用可信裸布附近的残差校正局部光照。大面积印花内部缺少布底观测时沿用光滑曲面。
void refine_background(Grid &grid, const Spectrum &scale)
{
    for (auto &t : grid.tiles)
    {
        if (t.outside)
            continue;
        double distance = 0;
        double largest = 0;
        double second_largest = 0;
        for (int b = 0; b < channels; ++b)
        {
            const double z = std::abs(t.residual[b]) / std::max(.055f, scale[b]);
            distance += z * z;
            second_largest = std::max(second_largest, std::min(largest, z));
            largest = std::max(largest, z);
        }
        // 暗角通常同时影响多个波段。仅在单一波段显现的强信号可能来自油墨，不能作为裸布锚点。
        t.isolated_band = largest > 3.2 && second_largest < .45 * largest;
        const double ratio = distance / (channels * 2.8 * 2.8);
        t.weight = !t.isolated_band && distance < channels * 9 ? 1 / (1 + ratio * ratio * ratio) : 0;
    }
    // 只沿光谱变化连续的相邻节点扩展裸布，以跟随画面边缘渐变的照明。
    for (int pass = 0; pass < 30; ++pass)
    {
        std::vector<size_t> added;
        for (int y = 0; y < grid.rows; ++y)
        {
            for (int x = 0; x < grid.columns; ++x)
            {
                const auto &t = grid.at(x, y);
                if (t.outside || t.isolated_band || t.weight > .4)
                {
                    continue;
                }
                // 光照斜坡外推限于视场边缘。内部只允许少数轮保守传播，防止将连续实心墨区吸收到布底模型。
                const bool at_field_edge = std::max(std::abs(t.basis[1]), std::abs(t.basis[2])) > .86;
                if (!at_field_edge && pass >= 4)
                {
                    continue;
                }
                int support = 0;
                for (int yy = std::max(0, y - 1); yy <= std::min(grid.rows - 1, y + 1); ++yy)
                {
                    for (int xx = std::max(0, x - 1); xx <= std::min(grid.columns - 1, x + 1); ++xx)
                    {
                        const auto &n = grid.at(xx, yy);
                        if (n.weight <= .4)
                        {
                            continue;
                        }
                        // 以已测得的照明斜率预测下一个节点，用预测残差区分暗角延续与墨区边界的突变。
                        const int px = 2 * xx - x, py = 2 * yy - y;
                        const bool has_prior = at_field_edge && px >= 0 && px < grid.columns && py >= 0 &&
                                               py < grid.rows && grid.at(px, py).weight > .4;
                        double distance = 0, curvature = 0;
                        double ratio = 0;
                        if (has_prior)
                        {
                            const double previous =
                                std::hypot(grid.xs[xx] - grid.xs[px], grid.ys[yy] - grid.ys[py]);
                            ratio =
                                std::min(2.0, std::hypot(grid.xs[x] - grid.xs[xx], grid.ys[y] - grid.ys[yy]) /
                                                  previous);
                        }
                        for (int b = 0; b < channels; ++b)
                        {
                            const double floor = std::max(.055f, scale[b]);
                            const double change = t.residual[b] - n.residual[b];
                            const double slope =
                                has_prior ? ratio * (n.residual[b] - grid.at(px, py).residual[b]) : 0;
                            distance += change * change / (floor * floor);
                            curvature += (change - slope) * (change - slope) / (floor * floor);
                        }
                        support += std::min(distance, curvature) < channels * 1.35 * 1.35;
                    }
                }
                if (support >= 2)
                {
                    added.push_back(static_cast<size_t>(y) * grid.columns + x);
                }
            }
        }
        if (added.empty())
        {
            break;
        }
        for (auto i : added)
        {
            grid.tiles[i].weight = .9;
        }
    }
    for (int y = 0; y < grid.rows; ++y)
    {
        for (int x = 0; x < grid.columns; ++x)
        {
            auto &t = grid.at(x, y);
            Spectrum correction{};
            double total = 0;
            for (int yy = std::max(0, y - 2); yy <= std::min(grid.rows - 1, y + 2); ++yy)
            {
                for (int xx = std::max(0, x - 2); xx <= std::min(grid.columns - 1, x + 2); ++xx)
                {
                    const auto &n = grid.at(xx, yy);
                    const int r2 = (xx - x) * (xx - x) + (yy - y) * (yy - y);
                    const double weight = std::exp(-.75 * r2) * n.weight;
                    for (int b = 0; b < channels; ++b)
                    {
                        correction[b] += static_cast<float>(weight * n.residual[b]);
                    }
                    total += weight;
                }
            }
            for (int b = 0; b < channels; ++b)
            {
                t.expected[b] += correction[b] / static_cast<float>(std::max(.12, total));
            }
        }
    }
}

struct FabricNoise
{
    Spectrum center{};
    std::array<Spectrum, channels> inverse_cholesky{};
};

// 用布底残差协方差抑制相关的布纹和照明波动。对角正则项保证平坦、饱和或冗余波段下仍可分解矩阵。
FabricNoise estimate_noise(const Image &image, const Grid &grid, const std::uint8_t *outside)
{
    const int stride_x = std::max(1, image.width / 360);
    const int stride_y = std::max(1, image.height / 240);
    std::vector<Spectrum> residuals;
    for (int y = stride_y / 2; y < image.height; y += stride_y)
    {
        const Axis gy = grid_position(y, grid.ys);
        for (int x = stride_x / 2; x < image.width; x += stride_x)
        {
            if (outside[static_cast<size_t>(y) * image.width + x])
                continue;
            const Axis gx = grid_position(x, grid.xs);
            if (grid.at(gx.fraction < .5f ? gx.left : gx.right, gy.fraction < .5f ? gy.left : gy.right)
                    .weight < .7)
            {
                continue;
            }
            Spectrum sample = image.sample(x, y);
            const Spectrum predicted = interpolate(grid, gx, gy);
            for (int b = 0; b < channels; ++b)
            {
                sample[b] -= predicted[b];
            }
            residuals.push_back(sample);
        }
    }
    // 先按波段减去残差中位数，再估计波段间的相关性，避免少量墨点主导噪声尺度。
    FabricNoise noise;
    Spectrum deviation{};
    std::vector<float> values(residuals.size());
    for (int b = 0; b < channels; ++b)
    {
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = residuals[i][b];
        }
        noise.center[b] = median(values);
        for (size_t i = 0; i < values.size(); ++i)
        {
            residuals[i][b] -= noise.center[b];
            values[i] = std::abs(residuals[i][b]);
        }
        deviation[b] = std::max(.025f, 1.4826f * median(values));
    }
    Matrix<channels> cov{};
    int count = 0;
    for (const auto &r : residuals)
    {
        double energy = 0;
        for (int b = 0; b < channels; ++b)
        {
            const double z = r[b] / deviation[b];
            energy += z * z;
        }
        if (energy >= channels * 3)
        {
            continue;
        }
        ++count;
        for (int i = 0; i < channels; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                cov[i][j] += static_cast<double>(r[i]) * r[j];
            }
        }
    }
    for (int i = 0; i < channels; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            cov[i][j] /= std::max(1, count);
        }
        const double floor = std::max(.022f, deviation[i] * .45f);
        cov[i][i] += floor * floor;
    }
    cholesky(cov);
    for (int j = 0; j < channels; ++j)
    {
        for (int i = j; i < channels; ++i)
        {
            double value = i == j ? 1.0 : 0.0;
            for (int k = j; k < i; ++k)
            {
                value -= cov[i][k] * noise.inverse_cholesky[k][j];
            }
            noise.inverse_cholesky[i][j] = static_cast<float>(value / cov[i][i]);
        }
    }
    return noise;
}

// 按行更新纵向、横向滑动和。只保留行工作区，避免为八个波段分别分配整幅浮点平滑图。
void classify(const Image &image, const Grid &grid, const FabricNoise &noise, int threshold,
              std::uint8_t *score, std::uint8_t *mask, const int *cancel)
{
    const int w = image.width, h = image.height, radius = image.radius;
    std::vector<Axis> position(w);
    for (int x = 0; x < w; ++x)
    {
        position[x] = grid_position(x, grid.xs);
    }
    // 每个条带独立初始化滑动窗口；条带边界仍然从整幅原图取邻域。
    cv::parallel_for_(
        cv::Range(0, h),
        [&](const cv::Range &range) {
            std::array<std::vector<int>, channels> vertical;
            std::vector<Spectrum> observed(w);
            for (int b = 0; b < channels; ++b)
            {
                vertical[b].resize(w);
                for (int dy = -radius; dy <= radius; ++dy)
                {
                    const auto *row =
                        image.bands[b] + static_cast<size_t>(std::clamp(range.start + dy, 0, h - 1)) * w;
                    for (int x = 0; x < w; ++x)
                    {
                        vertical[b][x] += row[x];
                    }
                }
            }
            for (int y = range.start; y < range.end; ++y)
            {
                if (cancel && _InterlockedCompareExchange((volatile long *)cancel, 0, 0))
                    return;
                const Axis gy = grid_position(y, grid.ys);
                for (int b = 0; b < channels; ++b)
                {
                    auto &sums = vertical[b];
                    int sum = 0;
                    for (int dx = -radius; dx <= radius; ++dx)
                    {
                        sum += sums[std::clamp(dx, 0, w - 1)];
                    }
                    for (int x = 0; x < w; ++x)
                    {
                        observed[x][b] = image.log_table[sum];
                        sum += sums[std::min(w - 1, x + radius + 1)] - sums[std::max(0, x - radius)];
                    }
                    const auto *leaving = image.bands[b] + static_cast<size_t>(std::max(0, y - radius)) * w;
                    const auto *entering =
                        image.bands[b] + static_cast<size_t>(std::min(h - 1, y + radius + 1)) * w;
                    for (int x = 0; x < w; ++x)
                    {
                        sums[x] += entering[x] - leaving[x];
                    }
                }
                for (int x = 0; x < w; ++x)
                {
                    const auto expected = interpolate(grid, position[x], gy);
                    Spectrum r{};
                    for (int b = 0; b < channels; ++b)
                    {
                        r[b] = observed[x][b] - expected[b] - noise.center[b];
                    }
                    float energy = 0;
                    for (int i = 0; i < channels; ++i)
                    {
                        float whitened = 0;
                        for (int j = 0; j <= i; ++j)
                        {
                            whitened += noise.inverse_cholesky[i][j] * r[j];
                        }
                        energy += whitened * whitened;
                    }
                    // 距离乘 22.5 后量化到 Gray8，默认阈值 128 约对应 5.7 的白化距离。
                    const int value = std::min(255, static_cast<int>(std::sqrt(energy) * 22.5f + .5f));
                    const size_t index = static_cast<size_t>(y) * w + x;
                    score[index] = static_cast<std::uint8_t>(value);
                    mask[index] = value >= threshold ? 255 : 0;
                }
            }
        },
        std::min(12, h));
}

} // 匿名命名空间

bool segment(const Bands &bands, int width, int height, int threshold, std::uint8_t *score,
             std::uint8_t *mask, const std::uint8_t *outside, const int *cancel)
{
    // 先估计空间变化的布底，再对白化后的八维残差打分，最后修正特定深色笔画的光晕。
    const Image image(bands, width, height);
    auto grid = sample_tiles(image, outside);
    const auto scale = fit_background(grid);
    refine_background(grid, scale);
    // 网格拟合覆盖不足时仍返回可查看的结果，同时通知上位机复核。
    int anchors = 0, trusted = 0;
    for (const auto &tile : grid.tiles)
        if (tile.fit_anchor)
        {
            ++anchors;
            trusted += tile.weight >= .7;
        }
    const auto noise = estimate_noise(image, grid, outside);
    classify(image, grid, noise, threshold, score, mask, cancel);
    if (cancel && _InterlockedCompareExchange((volatile long *)cancel, 0, 0))
        throw std::runtime_error("Cancelled");
    refine_edges(bands, width, height, threshold, score, mask);
    return trusted >= 10 && trusted * 5 >= anchors;
}

} // 光谱算法命名空间
