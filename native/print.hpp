#pragma once
#include <cstdint>
#define API extern "C" __declspec(dllexport)
// 输入为借用的 Gray8 行缓冲，最后一行只要求 width 个有效字节，允许每行尾部填充。
struct Plane
{
    const uint8_t *data;
    int64_t bytes;
    int width, height, stride, band;
};
// flags 可组合：1 配准需复核，2 触边，4 前景为空或不确定像素较多，8 底布参照不足。
// foreground 为最终前景像素数，millis 为一次 ms_run 的原生耗时。
struct Result
{
    double millis;
    int flags, foreground;
    float shift, response;
};
// 通道编号 0～7 依次为 UV、B、G、AM、R、FR、IR、W。输出逐行紧密排列。
// score 是原始树票得分，mask 是经过空间清理的二值掩膜，两者在同一位置可能跨过不同判定边界。
// 一个工作线程独占一个模型句柄；输入输出内存由调用方持有至调用结束。
API int ms_version();
// 返回线程局部字符串，下次受保护的调用会更新它，调用方如需长期保存应立即复制。
API const char *ms_error();
API int ms_load(const wchar_t *filename, void **context);
API void ms_free(void *context);
API int ms_run(void *context, const Plane *planes, int count, uint8_t threshold, uint8_t *mask,
               uint8_t *score, int64_t capacity, Result *result, const int *cancel);
// 可调接口在末尾接收去噪面积、补洞面积、补洞证据容差。前两项为 0 时关闭对应操作。
API int ms_run_configured(void *context, const Plane *planes, int count, uint8_t threshold, uint8_t *mask,
                          uint8_t *score, int64_t capacity, Result *result, const int *cancel, int noiseArea,
                          int holeArea, int holeAllowance);
// 训练时每行为八个波段强度和两个白光纹理量；采样点使用 W 图坐标。
API int ms_sample(const Plane *planes, int count, const int *xy, int rows, float *output);
API int ms_train(const float *data, const int *labels, int rows, int route, const wchar_t *filename);

API int ms_sample_mode(const Plane *p, int count, const int *xy, int rows, float *output, int relative);

API int ms_score_rows(void *context, const float *features, int rows, uint8_t *score);
