#pragma once
#include <opencv2/ml.hpp>
#include <emmintrin.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

// 对本项目的二分类、有序数值分裂做等价的批量投票。只读索引随模型构建一次。
// 16 个相邻像素共走一棵树；掩码为空的子树无需访问。末尾不足 16 个的像素单独处理。
class ForestVotes
{
    struct Node
    {
        int column, left, right;
        float cut;
    };
    std::vector<Node> nodes;
    std::vector<int> roots;
    std::array<unsigned char, 256> score{};

    void accumulate(int index, const float *columns, size_t stride, __m128i active, __m128i &votes) const
    {
        while (index >= 0)
        {
            const auto &node = nodes.data()[index];
            const auto *values = columns + node.column * stride;
            __m128 cut = _mm_set1_ps(node.cut);
            auto compare = [&](int i) {
                return _mm_castps_si128(_mm_cmple_ps(_mm_loadu_ps(values + i), cut));
            };
            __m128i takeLeft = _mm_packs_epi16(_mm_packs_epi32(compare(0), compare(4)),
                                               _mm_packs_epi32(compare(8), compare(12)));
            __m128i left = _mm_and_si128(active, takeLeft);
            __m128i right = _mm_andnot_si128(takeLeft, active);
            if (!_mm_movemask_epi8(left))
            {
                index = node.right;
                continue;
            }
            if (_mm_movemask_epi8(right))
                accumulate(node.right, columns, stride, right, votes);
            index = node.left;
            active = left;
        }
        // 叶节点 -2 表示前景；有效掩码字节为 -1，减法即加一票。
        if (index == -2)
            votes = _mm_sub_epi8(votes, active);
    }

  public:
    void prepare(const cv::ml::RTrees &forest, const cv::FileNode &metadata)
    {
        std::vector<int> types, columns, classes;
        metadata["var_type"] >> types;
        metadata["var_idx"] >> columns;
        metadata["class_labels"] >> classes;
        const int n = forest.getVarCount();
        if (classes != std::vector<int>{0, 1} || int(columns.size()) != n || int(types.size()) != n + 1 ||
            forest.getRoots().empty() || forest.getRoots().size() > 255 || forest.getMaxDepth() > 32)
            return;
        for (int i = 0; i < n; i++)
            if (types[i] != 0 || columns[i] != i)
                return;
        const auto &tree = forest.getNodes();
        const auto &splits = forest.getSplits();
        std::vector<Node> compiled(tree.size());
        auto child = [&](int i) { return tree[i].left < 0 ? -1 - tree[i].classIdx : i; };
        for (size_t i = 0; i < tree.size(); i++)
        {
            auto &v = tree[i];
            if (v.left < 0)
            {
                if (v.classIdx != 0 && v.classIdx != 1)
                    return;
                continue;
            }
            if (size_t(v.left) >= tree.size() || size_t(v.right) >= tree.size() ||
                size_t(v.split) >= splits.size())
                return;
            const auto &s = splits[v.split];
            if (s.varIdx < 0 || s.varIdx >= n || s.inversed || s.next >= 0 || !std::isfinite(s.c))
                return;
            compiled[i] = {s.varIdx, child(v.left), child(v.right), s.c};
        }
        for (int r : forest.getRoots())
            roots.push_back(child(r));
        nodes = std::move(compiled);
        for (size_t i = 0; i <= roots.size(); i++)
            score[i] = static_cast<unsigned char>(cvRound(255.0 * i / roots.size()));
    }

    bool available() const
    {
        return !roots.empty();
    }

    void evaluate(const cv::Mat &features, unsigned char *out) const
    {
        cv::Mat columns;
        cv::transpose(features, columns);
        const auto *base = columns.ptr<float>();
        const size_t stride = columns.step1();
        int i = 0;
        for (; i + 16 <= features.rows; i += 16)
        {
            __m128i votes = _mm_setzero_si128();
            for (int root : roots)
                accumulate(root, base + i, stride, _mm_set1_epi8(-1), votes);
            alignas(16) unsigned char counts[16];
            _mm_store_si128(reinterpret_cast<__m128i *>(counts), votes);
            for (int j = 0; j < 16; j++)
                out[i + j] = score[counts[j]];
        }
        for (; i < features.rows; i++)
        {
            int count = 0;
            const float *row = features.ptr<float>(i);
            for (int index : roots)
            {
                while (index >= 0)
                {
                    const auto &v = nodes[index];
                    index = row[v.column] <= v.cut ? v.left : v.right;
                }
                count += index == -2;
            }
            out[i] = score[count];
        }
    }
};
