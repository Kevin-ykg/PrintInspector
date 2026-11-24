#pragma once

#include <array>
#include <cstdint>

namespace spectral
{

// 八个输入平面均为借用的 Gray8 缓冲区，自上而下紧密排列，共用相机坐标。
// 顺序为 405、457、527、600、660、730、860 nm 和白光。调用方提供两个互不重叠的输出，各占宽×高字节。
// score 随像素与布底的光谱差异增大。score 达到阈值（1～254）时 mask 为 255，否则为
// 0。调用结束后释放内部工作区。
// outside 非零表示已确认的台面，布局与输入相同。
// 返回值表示本次有足够的可信布底拟合点。
bool segment(const std::array<const std::uint8_t *, 8> &bands, int width, int height, int threshold,
             std::uint8_t *score, std::uint8_t *mask, const std::uint8_t *outside, const int *cancel);

} // 光谱算法命名空间
