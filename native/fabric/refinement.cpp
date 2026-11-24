#include "refinement.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace spectral
{
namespace
{

using Bands = std::array<const std::uint8_t *, 8>;
constexpr int tile_size = 16;

struct FabricSamples
{
    std::array<int, 8> sum{}, squares{};
    int count = 0, area = 0;
    void merge(const FabricSamples &other)
    {
        count += other.count;
        area += other.area;
        for (int b = 0; b < 8; ++b)
        {
            sum[b] += other.sum[b];
            squares[b] += other.squares[b];
        }
    }
};

// 每个块记录布底均值、噪声倒数及邻域峰值；只有得到光谱证据的块才分配精化工作区。
struct LocalModel
{
    bool active = false;
    std::array<float, 8> background{}, inverse_sd{};
    int peak_sharp = 0, peak_all = 0, dark_votes = 0;
    size_t scratch_offset = 0;
};

// 十字邻域均值抑制布纹，同时控制窄笔画的横向扩张。
int sample(const std::uint8_t *band, int x, int y, int width, int height)
{
    const size_t index = static_cast<size_t>(y) * width + x;
    const int left = x > 0, right = x < width - 1;
    const int up = y > 0 ? width : 0, down = y < height - 1 ? width : 0;
    return (band[index] + band[index - left] + band[index + right] + band[index - up] + band[index + down] +
            2) /
           5;
}

// 以固定分数门限筛选裸布锚点，用它们估计局部背景均值与波动。
std::vector<LocalModel> estimate_fabric(const Bands &bands, int width, int height, int nx, int ny,
                                        const std::uint8_t *score)
{
    std::vector<FabricSamples> local(static_cast<size_t>(nx) * ny);
    std::vector<LocalModel> models(local.size());
    for (int y = 0; y < height; y += 4)
    {
        for (int x = 0; x < width; x += 4)
        {
            size_t i = static_cast<size_t>(y) * width + x;
            auto &tile = local[static_cast<size_t>(y / tile_size) * nx + x / tile_size];
            ++tile.area;
            if (score[i] >= 128)
            {
                continue;
            }
            ++tile.count;
            for (int b = 0; b < 8; ++b)
            {
                const int value = sample(bands[b], x, y, width, height);
                tile.sum[b] += value;
                tile.squares[b] += value * value;
            }
        }
    }
    // 每个块从周围约 80×80 像素范围收集裸布样本，边界处只取有效区域。
    for (int v = 0; v < ny; ++v)
    {
        for (int u = 0; u < nx; ++u)
        {
            FabricSamples total;
            for (int j = std::max(0, v - 2); j <= std::min(ny - 1, v + 2); ++j)
            {
                for (int k = std::max(0, u - 2); k <= std::min(nx - 1, u + 2); ++k)
                {
                    total.merge(local[static_cast<size_t>(j) * nx + k]);
                }
            }
            if (total.count < 8 || total.count < total.area * .065 || total.count > total.area * .985)
            {
                continue;
            }
            auto &model = models[static_cast<size_t>(v) * nx + u];
            model.active = true;
            for (int b = 0; b < 8; ++b)
            {
                float mean = static_cast<float>(total.sum[b]) / total.count;
                float variance = static_cast<float>(total.squares[b]) / total.count - mean * mean;
                model.background[b] = mean;
                model.inverse_sd[b] = 1.f / std::sqrt(std::max(9.f, variance));
            }
        }
    }
    return models;
}

size_t select_dark_strokes(const Bands &bands, int width, int height, int nx, int ny,
                           std::vector<LocalModel> &models)
{
    // 仅在 660、730 nm 同时变暗且 405 nm 变亮的深色笔画附近精化边缘，保留仅在紫光或红外可见的其他图案。
    for (int y = 0; y < height; y += 4)
    {
        for (int x = 0; x < width; x += 4)
        {
            auto &model = models[static_cast<size_t>(y / tile_size) * nx + x / tile_size];
            if (!model.active)
            {
                continue;
            }
            const auto delta = [&](int b) {
                return sample(bands[b], x, y, width, height) - model.background[b];
            };
            float red = delta(4), infrared = delta(5), violet = delta(0);
            if (red < -25 && infrared < -15 && -red * model.inverse_sd[4] > 5 &&
                -infrared * model.inverse_sd[5] > 4 && violet > 7 && violet * model.inverse_sd[0] > 2)
            {
                ++model.dark_votes;
            }
        }
    }
    size_t active_tiles = 0;
    for (int v = 0; v < ny; ++v)
    {
        for (int u = 0; u < nx; ++u)
        {
            auto &model = models[static_cast<size_t>(v) * nx + u];
            if (!model.active)
            {
                continue;
            }
            int votes = 0;
            for (int j = std::max(0, v - 1); j <= std::min(v + 1, ny - 1); ++j)
            {
                for (int k = std::max(0, u - 1); k <= std::min(u + 1, nx - 1); ++k)
                {
                    votes += models[static_cast<size_t>(j) * nx + k].dark_votes;
                }
            }
            model.active = votes >= 2;
            if (model.active)
            {
                model.scratch_offset = (active_tiles++) * tile_size * tile_size;
            }
        }
    }
    return active_tiles;
}

void measure_contrast(const Bands &bands, int width, int height, int nx, int ny,
                      std::vector<LocalModel> &models, std::uint8_t *mask,
                      std::vector<std::uint8_t> &all_band_response)
{
    for (int v = 0; v < ny; ++v)
    {
        for (int u = 0; u < nx; ++u)
        {
            auto &model = models[static_cast<size_t>(v) * nx + u];
            if (!model.active)
            {
                continue;
            }
            const int y0 = v * tile_size, x0 = u * tile_size;
            const int y1 = std::min(height, y0 + tile_size);
            const int x1 = std::min(width, x0 + tile_size);
            // 先对 4×4 单元求平均，再取最大响应，避免单个噪点主导邻域峰值。
            std::array<int, 16> sharp_cells{}, full_cells{}, counts{};
            for (int y = y0; y < y1; ++y)
            {
                for (int x = x0; x < x1; ++x)
                {
                    size_t i = static_cast<size_t>(y) * width + x;
                    std::array<float, 8> delta;
                    for (int b = 0; b < 8; ++b)
                    {
                        int value = sample(bands[b], x, y, width, height);
                        delta[b] = std::abs(value - model.background[b]) * model.inverse_sd[b];
                    }

                    float sharp = std::max({delta[1], delta[2], delta[3], delta[4], delta[7]});
                    float all = std::max({sharp, delta[0], delta[5], delta[6]});
                    // 在待精化块中临时借用掩膜缓冲区保存可见波段对比度。最后按修正分数重新写入二值结果。
                    mask[i] = static_cast<std::uint8_t>(std::min(255.f, sharp * 3.f + .5f));
                    auto &full_response =
                        all_band_response[model.scratch_offset + (y - y0) * tile_size + x - x0];
                    full_response = static_cast<std::uint8_t>(std::min(255.f, all * 3.f + .5f));
                    const int cell = (y - y0) / 4 * 4 + (x - x0) / 4;
                    sharp_cells[cell] += mask[i];
                    full_cells[cell] += full_response;
                    ++counts[cell];
                }
            }
            for (size_t c = 0; c < counts.size(); ++c)
            {
                if (counts[c] == 0)
                {
                    continue;
                }
                model.peak_sharp = std::max(model.peak_sharp, sharp_cells[c] / counts[c]);
                model.peak_all = std::max(model.peak_all, full_cells[c] / counts[c]);
            }
        }
    }
}

void update_scores(int width, int height, int nx, int ny, const std::vector<LocalModel> &models,
                   const std::vector<std::uint8_t> &all_band_response, int threshold, std::uint8_t *score,
                   std::uint8_t *mask)
{
    for (int v = 0; v < ny; ++v)
    {
        for (int u = 0; u < nx; ++u)
        {
            auto &model = models[static_cast<size_t>(v) * nx + u];
            if (!model.active)
            {
                continue;
            }
            int near_sharp = 0, near_all = 0;
            for (int j = std::max(0, v - 1); j <= std::min(v + 1, ny - 1); ++j)
            {
                for (int k = std::max(0, u - 1); k <= std::min(u + 1, nx - 1); ++k)
                {
                    const auto &next = models[static_cast<size_t>(j) * nx + k];
                    near_sharp = std::max(near_sharp, next.peak_sharp);
                    near_all = std::max(near_all, next.peak_all);
                }
            }
            const float visible_scale = 128.f / std::max(3 * 3.8f, .28f * near_sharp);
            const float multispectral_scale = 128.f / std::max(3 * 8.f, .62f * near_all);
            const int y0 = v * tile_size, x0 = u * tile_size;
            for (int y = y0; y < std::min(height, y0 + tile_size); ++y)
            {
                for (int x = x0; x < std::min(width, x0 + tile_size); ++x)
                {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    if (near_sharp > 24)
                    {
                        // 清晰可见波段或全谱的强响应均能保留该像素。绝对响应下限用于保护混色边界。
                        const auto visible_response = mask[i];
                        const auto full_response =
                            all_band_response[model.scratch_offset + (y - y0) * tile_size + x - x0];
                        const float refined =
                            std::max({visible_response * visible_scale, full_response * multispectral_scale,
                                      visible_response * (128.f / 24), full_response * (128.f / 42)});
                        score[i] =
                            static_cast<std::uint8_t>(std::min(static_cast<float>(score[i]), refined + .5f));
                    }
                    mask[i] = score[i] >= threshold ? 255 : 0;
                }
            }
        }
    }
}
} // 匿名命名空间

void refine_edges(const Bands &bands, int width, int height, int threshold, std::uint8_t *score,
                  std::uint8_t *mask)
{
    const int nx = (width - 1) / tile_size + 1;
    const int ny = (height - 1) / tile_size + 1;
    auto models = estimate_fabric(bands, width, height, nx, ny, score);
    const size_t active_tiles = select_dark_strokes(bands, width, height, nx, ny, models);
    if (active_tiles == 0)
    {
        return;
    }
    std::vector<std::uint8_t> full(active_tiles * tile_size * tile_size);
    measure_contrast(bands, width, height, nx, ny, models, mask, full);
    update_scores(width, height, nx, ny, models, full, threshold, score, mask);
}
} // 光谱算法命名空间
