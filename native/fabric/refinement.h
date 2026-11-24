#pragma once
#include <array>
#include <cstdint>
namespace spectral
{
// 在全谱分类后，结合 405、660、730 nm 的局部响应，抑制深色笔画边缘的光晕。
// 图像排列与 segment 相同。原位更新候选区域的 score 与 mask，最终 mask 等于 (score >= threshold) × 255。
void refine_edges(const std::array<const std::uint8_t *, 8> &bands, int width, int height, int threshold,
                  std::uint8_t *score, std::uint8_t *mask);
}
