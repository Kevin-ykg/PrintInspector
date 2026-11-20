#include "print.hpp"
#include "refine.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
using namespace cv;
namespace
{
// C ABI 不允许 C++ 异常穿过 P/Invoke；调用线程通过 ms_error 读取最近一次失败。
thread_local std::string lastError;
template <class F> int guarded(F f) noexcept
{
    try
    {
        lastError.clear();
        f();
        return 0;
    }
    catch (const std::exception &e)
    {
        lastError = e.what();
        return -1;
    }
    catch (...)
    {
        lastError = "Native processing failed";
        return -1;
    }
}
void require(bool ok, const char *msg)
{
    if (!ok)
        throw std::runtime_error(msg);
}
struct Stack
{
    std::array<Mat, 8> im;
    std::array<Point, 8> offset{};
    int w = 0, h = 0, flags = 0;
    float maxShift = 0, minResponse = 1;
    std::array<std::array<float, 256>, 8> lookup{};
    Stack(const Plane *p, int n, bool relative = false)
    {
        require(p && n == 8, "Expected eight planes");
        w = p[0].width;
        h = p[0].height;
        require(w >= 16 && h >= 16 && int64_t(w) * h <= 100000000, "Invalid image dimensions");
        for (int i = 0; i < n; i++)
        {
            const auto &a = p[i];
            require(a.band >= 0 && a.band < 8 && im[a.band].empty(), "Invalid or duplicate band");
            require(a.data && a.width == w && a.height == h && a.stride >= w &&
                        a.bytes >= int64_t(a.stride) * (h - 1) + w,
                    "Invalid plane shape or capacity");
            // 直接以真实行步长包装外部像素，避免复制八幅图像；band 决定内部顺序。
            im[a.band] = Mat(h, w, CV_8U, const_cast<uint8_t *>(a.data), a.stride);
        }
        // 从整帧按网格取中位值，作为该波段当前曝光下的参照。
        // 预计算查找表，避免逐像素求对数；旧绝对强度模型沿用原值。
        for (int b = 0; b < 8; b++)
        {
            int reference = 0;
            if (relative)
            {
                std::vector<int> grid;
                for (int y = 4; y < h; y += 16)
                    for (int x = 4; x < w; x += 16)
                        grid.push_back(im[b].ptr<uchar>(y)[x]);
                auto mid = grid.begin() + grid.size() / 2;
                std::nth_element(grid.begin(), mid, grid.end());
                reference = *mid;
            }
            for (int i = 0; i < 256; i++)
                lookup[b][i] = relative ? float(128 * std::log2((i + 8.0) / (reference + 8.0))) : float(i);
        }
        // 在三个内部窗口比较梯度幅值，容许同一边缘在不同照明下亮暗反转。
        int side = std::min({256, w / 2, h / 2});
        if (side < 32)
        {
            flags |= 1;
            return;
        }
        for (int b = 0; b < 7; b++)
        {
            std::vector<Point2d> shifts;
            for (double f : {0.3, 0.5, 0.7})
            {
                Rect roi(int(f * (w - side)), int(f * (h - side)), side, side);
                auto gradient = [&](int id) {
                    Mat x, y, g;
                    Sobel(im[id](roi), x, CV_32F, 1, 0);
                    Sobel(im[id](roi), y, CV_32F, 0, 1);
                    magnitude(x, y, g);
                    return g;
                };
                Mat a = gradient(7), c = gradient(b), window;
                createHanningWindow(window, a.size(), CV_32F);
                double response = 0;
                Point2d d = phaseCorrelate(a, c, window, &response);
                if (response >= 0.15 && std::isfinite(d.x) && std::isfinite(d.y))
                    shifts.push_back(d);
                minResponse = std::min(minResponse, float(response));
            }
            bool agreed = false;
            for (size_t i = 0; i < shifts.size() && !agreed; i++)
                for (size_t j = i + 1; j < shifts.size(); j++)
                {
                    if (norm(shifts[i] - shifts[j]) > 1)
                        continue;
                    Point2d d = (shifts[i] + shifts[j]) * 0.5;
                    maxShift = std::max(maxShift, float(norm(d)));
                    if (norm(d) <= 5)
                    {
                        offset[b] = {cvRound(d.x), cvRound(d.y)};
                        agreed = true;
                        break;
                    }
                }
            if (!agreed)
                flags |= 1;
        }
    }
    // offset 是 W 到目标波段的采样位移。边界钳位保证可读，返回值供调用方屏蔽共同视野以外的行。
    bool rowInfo(int y, std::array<const uchar *, 8> &ptr) const
    {
        bool valid = true;
        for (int b = 0; b < 8; b++)
        {
            int by = y + offset[b].y;
            valid &= by >= 0 && by < h;
            ptr[b] = im[b].ptr<uchar>(std::clamp(by, 0, h - 1));
        }
        return valid;
    }
    // 前八列是对齐后的光谱，末两列是 W 上相距两像素的梯度幅度和中心亮暗差。
    // 训练坐标抽样与全图条带推理共用同一实现，确保特征顺序一致。
    void pixel(int x, int y, const std::array<const uchar *, 8> &ptr, float *v) const
    {
        for (int b = 0; b < 8; b++)
            v[b] = lookup[b][ptr[b][std::clamp(x + offset[b].x, 0, w - 1)]];
        int l = ptr[7][std::max(0, x - 2)], r = ptr[7][std::min(w - 1, x + 2)];
        int t = im[7].ptr<uchar>(std::max(0, y - 2))[x], d = im[7].ptr<uchar>(std::min(h - 1, y + 2))[x];
        v[8] = float(std::abs(l - r) + std::abs(t - d));
        v[9] = ptr[7][x] - (l + r + t + d) * 0.25f;
    }
};
struct Model
{
    int route = 0;
    bool relative = false;
    Mat mean, scale;
    PCA pca;
    Ptr<ml::RTrees> forest;
    // 原始 N×10 行转为 N×17（八波段）或 N×(k+2)（PCA）；每一行仍对应同一像素。
    Mat features(const Mat &raw) const
    {
        Mat spectral;
        if (route == 8)
        {
            spectral.create(raw.rows, 15, CV_32F);
            raw.colRange(0, 8).copyTo(spectral.colRange(0, 8));
            for (int b = 0; b < 7; b++)
                subtract(raw.col(b), raw.col(7), spectral.col(8 + b));
        }
        else
        {
            Mat z(raw.rows, 8, CV_32F);
            for (int b = 0; b < 8; b++)
                raw.col(b).convertTo(z.col(b), CV_32F, 1.0 / scale.at<float>(b),
                                     -mean.at<float>(b) / scale.at<float>(b));
            spectral = pca.project(z);
        }
        Mat all;
        hconcat(spectral, raw.colRange(8, 10), all);
        return all;
    }
};
} // 匿名命名空间
API int ms_version()
{
    return 1;
}
API const char *ms_error()
{
    return lastError.c_str();
}
API void ms_free(void *p)
{
    delete static_cast<Model *>(p);
}
API int ms_sample_mode(const Plane *p, int n, const int *xy, int rows, float *out, int relative)
{
    return guarded([&] {
        require(xy && out && rows > 0, "Invalid training coordinates");
        Stack s(p, n, relative != 0);
        std::array<const uchar *, 8> ptr{};
        for (int i = 0; i < rows; i++)
        {
            int x = xy[2 * i], y = xy[2 * i + 1];
            require(x >= 0 && y >= 0 && x < s.w && y < s.h, "Point outside frame");
            s.rowInfo(y, ptr);
            s.pixel(x, y, ptr, out + 10 * i);
        }
    });
}
API int ms_sample(const Plane *p, int n, const int *xy, int rows, float *out)
{
    return ms_sample_mode(p, n, xy, rows, out, 0);
}
API int ms_train(const float *values, const int *labels, int rows, int route, const wchar_t *path)
{
    return guarded([&] {
        // 训练 ABI 用路线编号加 10 表示相对强度，模型中分开保存 route 与 relative。
        const bool relative = route > 10;
        if (relative)
            route -= 10;
        require(values && labels && rows >= 2 && path && (route == 8 || (route >= 2 && route <= 4)),
                "Invalid training inputs");
        Mat raw(rows, 10, CV_32F, const_cast<float *>(values)), y(rows, 1, CV_32S, const_cast<int *>(labels));
        require(countNonZero(y == 0) > 0 && countNonZero(y == 1) > 0 &&
                    countNonZero((y != 0) & (y != 1)) == 0,
                "Both training classes required");
        Model m;
        m.route = route;
        if (route >= 2 && route <= 4)
        {
            Mat z(rows, 8, CV_32F);
            m.mean.create(1, 8, CV_32F);
            m.scale.create(1, 8, CV_32F);
            for (int b = 0; b < 8; b++)
            {
                Scalar u, v;
                meanStdDev(raw.col(b), u, v);
                double sd = std::max(v[0], 1.e-5);
                m.mean.at<float>(b) = float(u[0]);
                m.scale.at<float>(b) = float(sd);
                raw.col(b).convertTo(z.col(b), CV_32F, 1 / sd, -u[0] / sd);
            }
            // 只在训练行上拟合；加载后的 Model::features 对新样品复用这些统计量和投影基。
            m.pca = PCA(z, noArray(), PCA::DATA_AS_ROW, route);
        }
        setRNGSeed(1789);
        m.forest = ml::RTrees::create();
        m.forest->setMaxDepth(12);
        m.forest->setMinSampleCount(8);
        m.forest->setTermCriteria({TermCriteria::MAX_ITER, 24, 0});
        require(m.forest->train(m.features(raw), ml::ROW_SAMPLE, y), "Training failed");
        FileStorage fs("model.yml", FileStorage::WRITE | FileStorage::MEMORY);
        fs << "version" << 1 << "route" << route << "relative" << int(relative);
        // PCA 模型连同标准化参数和投影基一起保存，确保训练与部署使用相同坐标轴。
        if (!m.mean.empty())
        {
            fs << "mean" << m.mean << "scale" << m.scale << "pca" << "{";
            m.pca.write(fs);
            fs << "}";
        }
        fs << "forest" << "{";
        m.forest->write(fs);
        fs << "}";
        std::ofstream f(std::filesystem::path(path), std::ios::binary);
        f << fs.releaseAndGetString();
        f.close();
        require(bool(f), "Cannot save model");
    });
}
API int ms_load(const wchar_t *path, void **ctx)
{
    return guarded([&] {
        require(path && ctx, "Missing model path");
        *ctx = nullptr;
        std::ifstream f(std::filesystem::path(path), std::ios::binary);
        require(bool(f), "Cannot read model");
        std::ostringstream buf;
        buf << f.rdbuf();
        FileStorage fs(buf.str(), FileStorage::READ | FileStorage::MEMORY);
        require(int(fs["version"]) == 1, "Unsupported model");
        auto m = std::make_unique<Model>();
        m->route = int(fs["route"]);
        m->relative = int(fs["relative"]) != 0;
        require(m->route == 8 || (m->route >= 2 && m->route <= 4), "Invalid model route");
        if (m->route >= 2 && m->route <= 4)
        {
            fs["mean"] >> m->mean;
            fs["scale"] >> m->scale;
            m->pca.read(fs["pca"]);
            require(m->mean.type() == CV_32F && m->scale.type() == CV_32F && m->mean.total() == 8 &&
                        m->scale.total() == 8 && checkRange(m->scale, true, nullptr, 1e-8, 1e8) &&
                        m->pca.eigenvectors.rows == m->route && m->pca.eigenvectors.cols == 8,
                    "Invalid PCA model");
        }
        m->forest = ml::RTrees::create();
        m->forest->read(fs["forest"]);
        require(m->forest->isTrained(), "Empty forest");
        *ctx = m.release();
    });
}
API int ms_score_rows(void *ctx, const float *data, int rows, uint8_t *out)
{
    return guarded([&] {
        require(ctx && data && out && rows > 0, "Invalid sample buffer");
        auto &m = *static_cast<Model *>(ctx);
        Mat raw(rows, 10, CV_32F, const_cast<float *>(data)), votes;
        m.forest->getVotes(m.features(raw), votes, 0);
        require(votes.type() == CV_32S && votes.cols == 2 && votes.rows == rows + 1, "Invalid class votes");
        // getVotes 第一行保存类别编号，其余行才是各像素的票数。
        int pos = votes.at<int>(0, 0) == 1 ? 0 : 1;
        for (int i = 0; i < rows; i++)
        {
            auto v = votes.ptr<int>(i + 1);
            out[i] = uchar(cvRound(255.0 * v[pos] / (v[0] + v[1])));
        }
    });
}
API int ms_run(void *context, const Plane *p, int n, uint8_t threshold, uint8_t *mask, uint8_t *score,
               int64_t cap, Result *result, const int *cancel)
{
    // 已发布的调用方继续使用原参数表，默认配置与可调接口走同一条执行路径。
    const MaskSettings defaults;
    return ms_run_configured(context, p, n, threshold, mask, score, cap, result, cancel, defaults.noiseArea,
                             defaults.holeArea, defaults.holeAllowance);
}

API int ms_run_configured(void *context, const Plane *p, int n, uint8_t threshold, uint8_t *mask,
                          uint8_t *score, int64_t cap, Result *result, const int *cancel, int noiseArea,
                          int holeArea, int holeAllowance)
{
    return guarded([&] {
        require(noiseArea >= 0 && noiseArea <= 1024 && holeArea >= 0 && holeArea <= 8192 &&
                    holeAllowance >= 0 && holeAllowance <= 255,
                "Invalid mask settings");
        const MaskSettings settings{noiseArea, holeArea, holeAllowance};
        auto start = std::chrono::steady_clock::now();
        require(context && mask && score && result && mask != score, "Invalid output or context");
        Stack s(p, n, static_cast<Model *>(context)->relative);
        require(cap >= int64_t(s.w) * s.h, "Output too small");
        auto &m = *static_cast<Model *>(context);
        int left = 0, right = s.w, firstRow = 0, lastRow = s.h - 1;
        require(threshold > 0, "Threshold must be positive");
        // 各工作线程仅保存当前条带的特征和树票，避免分配整幅浮点特征图。
        const int tiles = (s.h + 15) / 16;
        struct TileResult
        {
            int uncertain = 0;
        };
        std::vector<TileResult> stats(tiles);
        std::atomic<bool> stopped = false;
        // 输出固定在 W 坐标系。只有八帧采样坐标都有效的共同视野才能产生前景。
        for (auto off : s.offset)
        {
            left = std::max(left, -off.x);
            right = std::min(right, s.w - off.x);
            firstRow = std::max(firstRow, -off.y);
            lastRow = std::min(lastRow, s.h - 1 - off.y);
        }
        parallel_for_(
            Range(0, tiles),
            [&](const Range &range) {
                for (int tile = range.start; tile < range.end; tile++)
                {
                    if (cancel && InterlockedCompareExchange((volatile LONG *)cancel, 0, 0))
                    {
                        stopped = true;
                        return;
                    }
                    int top = 16 * tile;
                    auto &local = stats[tile];
                    int height = std::min(16, s.h - top);
                    Mat raw(height * s.w, 10, CV_32F), votes;
                    for (int t = 0; t < height; t++)
                    {
                        std::array<const uchar *, 8> ptr{};
                        s.rowInfo(top + t, ptr);
                        for (int x = 0; x < s.w; x++)
                            s.pixel(x, top + t, ptr, raw.ptr<float>(t * s.w + x));
                    }
                    m.forest->getVotes(m.features(raw), votes, 0);
                    require(votes.type() == CV_32S && votes.rows == raw.rows + 1 && votes.cols == 2,
                            "Invalid class votes");
                    int pos = votes.at<int>(0, 0) == 1 ? 0 : 1;
                    for (int t = 0; t < height; t++)
                    {
                        std::array<const uchar *, 8> ptr{};
                        bool valid = s.rowInfo(top + t, ptr);
                        for (int x = 0; x < s.w; x++)
                        {
                            int i = t * s.w + x, idx = (top + t) * s.w + x;
                            auto v = votes.ptr<int>(i + 1);
                            int val = (valid && x >= left && x < right)
                                          ? cvRound(255.0 * v[pos] / (v[0] + v[1]))
                                          : 0;
                            score[idx] = uchar(val);
                            mask[idx] = (val >= threshold) ? 255 : 0;
                            local.uncertain += std::abs(val - int(threshold)) <= 20;
                        }
                    }
                }
            },
            std::min(6, tiles));
        require(!stopped.load(), "Cancelled");
        Mat output(s.h, s.w, CV_8U, mask), votes(s.h, s.w, CV_8U, score);
        Rect common(left, firstRow, right - left, lastRow - firstRow + 1);
        auto refined = refineMask(s.im, s.offset, common, votes, threshold, m.route != 8, settings, output);
        if (cancel && InterlockedCompareExchange((volatile LONG *)cancel, 0, 0))
            throw std::runtime_error("Cancelled");
        int count = countNonZero(output), uncertain = 0;
        for (const auto &block : stats)
            uncertain += block.uncertain;
        // 膨胀无效区域得到紧贴边界的一圈有效像素，命中前景时仅置复核标志。
        Mat invalidEdge;
        dilate(refined.excluded, invalidEdge, Mat());
        rectangle(invalidEdge, Rect(0, 0, s.w, s.h), Scalar(255), 1);
        if (countNonZero(output & invalidEdge))
            s.flags |= 2;
        if (uncertain > 0.05 * s.w * s.h || count == 0)
            s.flags |= 4;
        if (refined.weakBackground)
            s.flags |= 8;
        *result = {
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
            s.flags, count, s.maxShift, s.minResponse};
    });
}
