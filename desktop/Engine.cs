using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32.SafeHandles;

namespace PrintInspector;

/// <summary>与 C++ Plane 保持字段顺序和位宽一致。Stride 和 Size 的单位均为字节。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct Plane
{
    public IntPtr Data;
    public long Size;
    public int Width, Height, Stride, Band;
}

[StructLayout(LayoutKind.Sequential)]
internal struct Report
{
    public double Millis;
    public int Flags, Count;
    public float Shift, Response;
    // 一帧可同时触发多个复核原因，保留所有置位项供操作员定位。
    public readonly string Status
    {
        get
        {
            if (Flags == 0)
                return "分割完成";

            var reasons = new[]
            {
                (Flags & 1) != 0 ? "波段对齐" : null,
                (Flags & 2) != 0 ? "图案触边" : null,
                (Flags & 4) != 0 ? "前景得分" : null,
                (Flags & 8) != 0 ? "底布参照不足" : null,
                (Flags & 16) != 0 ? "暗区与台面难区分" : null
            };
            return "待复核 · " + string.Join(" / ", reasons.Where(reason => reason != null));
        }
    }
}

/// <summary>持有 ms_load 返回的模型句柄，由 SafeHandle 在用毕或异常退出时释放。</summary>
internal sealed class ModelHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public ModelHandle() : base(true)
    {
    }

    protected override bool ReleaseHandle()
    {
        Engine.ms_free(handle);
        return true;
    }
}

/// <summary>一次运行的不可变参数快照；同批次复用，导出时随掩膜记录。面积单位为原图像素。</summary>
internal sealed record MaskSettings(int NoiseArea, int HoleArea, int HoleAllowance, byte FabricThreshold = 128,
                                    int FabricNoiseArea = 96, int FabricProtection = 32)
{
    internal static readonly MaskSettings Default = new(96, 80, 46);
}

internal sealed record Output(BitmapSource White, BitmapSource Mask, BitmapSource Score, BitmapSource Contour,
                              Report Report, byte Threshold, MaskSettings Settings, double ReadMilliseconds, int[] OffsetsXY, bool Adaptive, string WhitePath);
internal static class Engine
{
    public static readonly string[] Bands = ["UV", "B", "G", "AM", "R", "FR", "IR", "W"];
    // 相机批量文件名按波长标识通道，开头的拍摄编号可以与样品目录不同。
    private static readonly string[] CameraBands = ["405", "457", "527", "600", "660", "730", "860", "w"];
    private static readonly Regex CameraName = new(@"^.+-(405|457|527|600|660|730|860|w)-[0-9]+\.bmp$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    private static string ResolveBand(string folder, string band)
    {
        string named = Path.Combine(folder, band + ".bmp");
        if (File.Exists(named))
            return named;
        string wavelength = CameraBands[Array.IndexOf(Bands, band)];
        var matches = Directory.EnumerateFiles(folder, "*.bmp").Where(path =>
            CameraName.Match(Path.GetFileName(path)).Groups[1].Value.Equals(wavelength, StringComparison.OrdinalIgnoreCase))
            .Take(2).ToArray();
        return matches.Length switch
        {
            1 => matches[0],
            0 => throw new IOException($"缺少 {band} 波段（{wavelength}）"),
            _ => throw new IOException($"{band} 波段（{wavelength}）有多个文件")
        };
    }

    internal static bool ContainsBands(string folder) =>
        Directory.EnumerateFiles(folder, "*.bmp").Any(path =>
            Bands.Contains(Path.GetFileNameWithoutExtension(path), StringComparer.OrdinalIgnoreCase) ||
            CameraName.IsMatch(Path.GetFileName(path)));

    const string Lib = "PrintVision";
    [DllImport(Lib)]
    static extern int ms_version();
    [DllImport(Lib, CharSet = CharSet.Unicode, ExactSpelling = true)]
    static extern int ms_load(string path, out ModelHandle handle);
    [DllImport(Lib)]
    internal static extern void ms_free(IntPtr handle);
    [DllImport(Lib)]
    static extern IntPtr ms_error();
    [DllImport(Lib)]
    static extern int ms_run_registered(ModelHandle model, [In] Plane[] planes, int count, byte threshold,
                             [Out] byte[] mask, [Out] byte[] score, long capacity, out Report report,
                             IntPtr cancel, int noiseArea, int holeArea, int holeAllowance, [Out] int[] offsetsXY);
    [DllImport(Lib)]
    static extern int ms_run_fabric_configured([In] Plane[] planes, int count, byte threshold, [Out] byte[] mask,
                                    [Out] byte[] score, long capacity, out Report report, IntPtr cancel, [Out] int[] offsetsXY,
                                    int noiseArea, int protectionMargin);

    internal static ModelHandle Load(string path)
    {
        if (ms_version() != 1)
            throw new IOException("DLL 接口版本不匹配");
        Check(ms_load(path, out var h));
        return h;
    }

    static void Check(int n)
    {
        if (n != 0)
            throw new IOException(Marshal.PtrToStringUTF8(ms_error()));
    }

    // 轮廓只有透明、半透明填充、实线三种颜色，索引图每像素只需一个字节。
    static BitmapPalette ContourPalette() => new(new[]
    {
        Color.FromArgb(0, 0, 0, 0), Color.FromArgb(38, 0, 244, 145), Color.FromArgb(255, 0, 244, 145)
    });

    /// <summary>读取灰度、索引色或 24 位彩色 BMP；每个照明波段输出一张单通道帧。关闭文件前加载并冻结像素，供工作线程和 UI 共用。</summary>
    internal static BitmapSource Read(string folder, string band, bool averageRgb = false)
    {
        return ReadFrame(ResolveBand(folder, band), averageRgb);
    }

    private static BitmapSource ReadFrame(string path, bool averageRgb)
    {
        using var f = File.OpenRead(path);
        var src = new BmpBitmapDecoder(f, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad)
                      .Frames[0];
        if (src.Format == PixelFormats.Bgr24 || src.Format == PixelFormats.Rgb24)
            return ColorToGray(src, averageRgb);
        if (src.Format.BitsPerPixel != 8)
            throw new IOException($"{Path.GetFileName(path)} 的像素格式 {src.Format} 不受支持，需要 8 位或 24 位 BMP");
        // 线性灰阶调色板的索引就是像素值；彩色或非线性调色板仍交给 WIC 转换。
        BitmapSource gray = src;
        if (src.Format != PixelFormats.Gray8 && !(src.Format == PixelFormats.Indexed8 &&
            src.Palette is { } palette && palette.Colors.Select((c, i) =>
                c.A == 255 && c.R == i && c.G == i && c.B == i).All(identity => identity)))
            gray = new FormatConvertedBitmap(src, PixelFormats.Gray8, null, 0);
        gray.Freeze();
        return gray;
    }

    // 采集装置的无压缩 BGR24 BMP 可逐块读入最终 Gray8 平面。这样每帧省去彩色 WPF
    // 位图、灰度显示位图和一次全图复制。其它合法编码沿用 WIC 的调色板及格式检查。
    internal static byte[]? ReadFabricBmp(string folder, string band, out int width, out int height) =>
        ReadFabricFile(ResolveBand(folder, band), out width, out height, null);

    private static byte[]? ReadFabricFile(string path, out int width, out int height, byte[]? reuse)
    {
        width = height = 0;
        using var input = File.OpenRead(path);
        string name = Path.GetFileName(path);
        Span<byte> header = stackalloc byte[54];
        if (input.Length < header.Length)
            return null;
        input.ReadExactly(header);
        uint dib = BinaryPrimitives.ReadUInt32LittleEndian(header[14..]);
        if (header[0] != 'B' || header[1] != 'M' || dib < 40 ||
            BinaryPrimitives.ReadUInt16LittleEndian(header[26..]) != 1 ||
            BinaryPrimitives.ReadUInt16LittleEndian(header[28..]) != 24 ||
            BinaryPrimitives.ReadUInt32LittleEndian(header[30..]) != 0)
            return null;

        width = BinaryPrimitives.ReadInt32LittleEndian(header[18..]);
        int signedHeight = BinaryPrimitives.ReadInt32LittleEndian(header[22..]);
        long offset = BinaryPrimitives.ReadUInt32LittleEndian(header[10..]);
        long rows = Math.Abs((long)signedHeight);
        if (width < 1 || rows < 1 || width * rows > 100_000_000 || offset < 14L + dib)
            throw new IOException($"{name} 的尺寸或像素起始位置无效");
        height = (int)rows;
        int stride = checked((width * 3 + 3) & ~3);
        if (offset + rows * stride > input.Length)
            throw new IOException($"{name} 的像素数据不完整");
        input.Position = offset;
        int blockRows = Math.Min(height, Math.Max(1, 1048576 / stride));
        var block = new byte[blockRows * stride];
        int pixels = checked(width * height);
        var gray = reuse?.Length == pixels ? reuse : new byte[pixels];
        for (int first = 0; first < height; first += blockRows)
        {
            int count = Math.Min(height - first, blockRows);
            input.ReadExactly(block.AsSpan(0, count * stride));
            for (int line = 0; line < count; line++)
            {
                // BMP 正高度按自底向上排列；逐行跳过对齐填充，输出统一采用 W 图坐标。
                int y = signedHeight > 0 ? height - 1 - (first + line) : first + line;
                int dst = y * width;
                for (int x = 0, j = line * stride; x < width; ++x, j += 3)
                    gray[dst + x] = (byte)((block[j] + block[j + 1] + block[j + 2] + 1) / 3);
            }
        }
        return gray;
    }

    // 森林沿用训练时的 BT.601 亮度，自适应布底沿用工作台的 RGB 均值；每次仅读取一条带。
    // 固定点系数与评估脚本 Pillow 的 RGB → L 一致；三分量相等时精确保留原灰度。
    private static BitmapSource ColorToGray(BitmapSource src, bool averageRgb)
    {
        int w = src.PixelWidth, h = src.PixelHeight, stride = checked(w * 3);
        int block = Math.Min(32, h);
        var rgb = new byte[checked(stride * block)];
        var gray = new byte[checked(w * h)];
        int red = src.Format == PixelFormats.Bgr24 ? 2 : 0;
        int blue = 2 - red;
        for (int y = 0; y < h; y += block)
        {
            int rows = Math.Min(block, h - y);
            src.CopyPixels(new System.Windows.Int32Rect(0, y, w, rows), rgb, stride, 0);
            for (int i = 0, j = 0; i < w * rows; ++i, j += 3)
                gray[y * w + i] = averageRgb
                    ? (byte)((rgb[j] + rgb[j + 1] + rgb[j + 2] + 1) / 3)
                    : (byte)((19595 * rgb[j + red] + 38470 * rgb[j + 1] +
                              7471 * rgb[j + blue] + 32768) >> 16);
        }
        return Bitmap(w, h, PixelFormats.Gray8, gray, w);
    }

    // 与 DLL 取样方向一致：W 坐标 (x,y) 取原波段 (x+dx,y+dy)。视野外填零。
    internal static BitmapSource InWhiteCoordinates(BitmapSource band, int dx, int dy)
    {
        if (dx == 0 && dy == 0)
            return band;
        int w = band.PixelWidth, h = band.PixelHeight;
        var pixels = new byte[checked(w * h)];
        var common = new System.Windows.Int32Rect(Math.Max(0, dx), Math.Max(0, dy), w - Math.Abs(dx), h - Math.Abs(dy));
        int start = Math.Max(0, -dy) * w + Math.Max(0, -dx);
        band.CopyPixels(common, pixels, w, start);
        return Bitmap(w, h, PixelFormats.Gray8, pixels, w);
    }

    static BitmapSource Bitmap(int w, int h, PixelFormat format, byte[] bytes, int stride, BitmapPalette? palette = null)
    {
        var bitmap = BitmapSource.Create(w, h, 96, 96, format, palette, bytes, stride);
        bitmap.Freeze();
        return bitmap;
    }

    /// <summary>处理一个样品，返回 W 坐标系的原图、掩膜、原始票率以及只用于显示的轮廓层。</summary>
    internal static Output Run(string dir, ModelHandle? model, byte threshold, CancellationToken token, MaskSettings? settings = null)
    {
        // Plane 仅借用像素地址。八个数组在本次原生调用结束前都必须固定，异常路径也需释放。
        settings ??= MaskSettings.Default;
        var pins = new List<GCHandle>();
        var planes = new Plane[8];
        // 彩色采集按组复用一幅解码工作图；八帧写入独立原生内存，推理结束立即释放。
        // 连续运行大图时可避免托管大对象堆堆积上一组的数百 MiB 输入。
        var nativeFrames = IntPtr.Zero;
        byte[]? decodeBuffer = null;
        int w = 0, h = 0;
        var cancel = Marshal.AllocHGlobal(4);
        Marshal.WriteInt32(cancel, 0);
        try
        {
            // 取消回调只能在标志仍有效时写入；using 的作用域先于 finally 中的 FreeHGlobal 结束。
            using var registration = token.Register(() => Marshal.WriteInt32(cancel, 1));
            long readStart = Stopwatch.GetTimestamp();
            BitmapSource? white = null;
            string[] paths = Bands.Select(band => ResolveBand(dir, band)).ToArray();
            for (int b = 0; b < 8; b++)
            {
                token.ThrowIfCancellationRequested();
                int width = 0, height = 0;
                byte[]? a = model == null ? ReadFabricFile(paths[b], out width, out height, decodeBuffer) : null;
                BitmapSource? im = null;
                if (a == null)
                {
                    im = ReadFrame(paths[b], model == null);
                    width = im.PixelWidth;
                    height = im.PixelHeight;
                    a = new byte[checked(width * height)];
                    im.CopyPixels(a, width, 0);
                }
                if (b == 0)
                {
                    w = width;
                    h = height;
                }
                if (width != w || height != h)
                    throw new IOException("波段尺寸不一致");
                IntPtr address;
                if (im == null)
                {
                    decodeBuffer = a;
                    if (nativeFrames == IntPtr.Zero)
                        nativeFrames = Marshal.AllocHGlobal(checked((nint)((long)a.Length * planes.Length)));
                    address = IntPtr.Add(nativeFrames, checked(b * a.Length));
                    Marshal.Copy(a, 0, address, a.Length);
                }
                else
                {
                    var pin = GCHandle.Alloc(a, GCHandleType.Pinned);
                    pins.Add(pin);
                    address = pin.AddrOfPinnedObject();
                }
                planes[b] = new()
                {
                    Data = address,
                    Size = a.Length,
                    Width = w,
                    Height = h,
                    Stride = w,
                    Band = b
                };
                if (b == 7)
                    white = im ?? Bitmap(w, h, PixelFormats.Gray8, a, w);
            }
            double readMilliseconds = Stopwatch.GetElapsedTime(readStart).TotalMilliseconds;
            byte[] m = new byte[checked(w * h)], s = new byte[m.Length];
            var offsetsXY = new int[16];
            Report report;
            var rc = model == null ? ms_run_fabric_configured(planes, 8, threshold, m, s, m.Length, out report, cancel, offsetsXY,
                                       settings.FabricNoiseArea, settings.FabricProtection)
                                   : ms_run_registered(model, planes, 8, threshold, m, s, m.Length, out report, cancel,
                                       settings.NoiseArea, settings.HoleArea, settings.HoleAllowance, offsetsXY);
            token.ThrowIfCancellationRequested();
            Check(rc);
            // 原生函数已经归还八帧指针，生成预览时可以解除固定。
            foreach (var pin in pins)
                pin.Free();
            pins.Clear();
            Marshal.FreeHGlobal(nativeFrames);
            nativeFrames = IntPtr.Zero;
            // 四邻域区分轮廓和填充；三种调色板索引只用于显示。
            var contour = new byte[m.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = y * w + x;
                    if (m[i] != 0 && (x == 0 || y == 0 || x == w - 1 || y == h - 1 || m[i - 1] == 0 ||
                                      m[i + 1] == 0 || m[i - w] == 0 || m[i + w] == 0))
                    {
                        contour[i] = 2;
                    }
                    else if (m[i] != 0)
                    {
                        contour[i] = 1;
                    }
                }
            return new(white!, Bitmap(w, h, PixelFormats.Gray8, m, w), Bitmap(w, h, PixelFormats.Gray8, s, w),
                       Bitmap(w, h, PixelFormats.Indexed8, contour, w, ContourPalette()), report, threshold, settings,
                       readMilliseconds, offsetsXY, model == null, paths[7]);
        }
        finally
        {
            foreach (var p in pins)
                p.Free();
            Marshal.FreeHGlobal(nativeFrames);
            Marshal.FreeHGlobal(cancel);
        }
    }

    // RenderTargetBitmap 需要单线程单元；让磁盘写入、整幅渲染与 PNG 编码离开界面线程。
    internal static Task SaveAsync(string dir, Output result) => ExportAsync(() => Save(dir, result));

    internal static Task SavePngAsync(string path, Output result, bool transparent) =>
        ExportAsync(() => WritePng(path, transparent ? WhiteCutout(result) : result.Mask));

    private static Task ExportAsync(Action export)
    {
        var finished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            try
            {
                try
                {
                    export();
                }
                finally
                {
                    // 主动关闭此导出线程的 WPF 调度器，使离屏渲染资源在批次中及时释放。
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
                finished.SetResult();
            }
            catch (Exception ex)
            {
                finished.SetException(ex);
            }
        })
        { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        return finished.Task;
    }

    // 白光原图保持相机的 RGB 或灰阶。只在导出时逐条解码，使用同坐标最终掩膜写入非预乘 Alpha。
    private static BitmapSource WhiteCutout(Output result)
    {
        using var stream = File.OpenRead(result.WhitePath);
        var white = new BmpBitmapDecoder(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.None)
                    .Frames[0];
        int width = result.Mask.PixelWidth, height = result.Mask.PixelHeight;
        if (white.PixelWidth != width || white.PixelHeight != height)
            throw new IOException("白光原图尺寸与掩膜不一致");
        var source = new FormatConvertedBitmap(white, PixelFormats.Bgra32, null, 0);
        var target = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
        const int stripHeight = 64;
        int stride = checked(width * 4);
        var rgb = new byte[checked(stride * Math.Min(stripHeight, height))];
        var alpha = new byte[checked(width * Math.Min(stripHeight, height))];
        for (int y = 0; y < height; y += stripHeight)
        {
            int rows = Math.Min(stripHeight, height - y);
            var rect = new System.Windows.Int32Rect(0, y, width, rows);
            source.CopyPixels(rect, rgb, stride, 0);
            result.Mask.CopyPixels(rect, alpha, width, 0);
            for (int i = 0, p = 0; i < width * rows; ++i, p += 4)
            {
                if (alpha[i] == 0)
                    rgb[p] = rgb[p + 1] = rgb[p + 2] = 0;
                rgb[p + 3] = alpha[i];
            }
            target.WritePixels(rect, rgb, stride, 0);
        }
        return target;
    }

    private static void WritePng(string path, BitmapSource frame)
    {
        var target = Path.GetFullPath(path);
        var temp = target + "." + Guid.NewGuid() + ".tmp";
        try
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(frame));
            using (var file = new FileStream(temp, FileMode.CreateNew))
                encoder.Save(file);
            File.Move(temp, target, true);
        }
        finally
        {
            if (File.Exists(temp))
                File.Delete(temp);
        }
    }

    /// <summary>按相机像素尺寸导出；叠加底图始终使用 W，观察窗口的缩放和选定波段均不参与导出。</summary>
    internal static void Save(string dir, Output result)
    {
        Directory.CreateDirectory(dir);
        WritePng(Path.Combine(dir, "mask.png"), result.Mask);
        WritePng(Path.Combine(dir, "score.png"), result.Score);
        var vis = new DrawingVisual();
        using (var ctx = vis.RenderOpen())
        {
            var bounds = new System.Windows.Rect(0, 0, result.White.PixelWidth, result.White.PixelHeight);
            ctx.DrawImage(result.White, bounds);
            ctx.DrawImage(result.Contour, bounds);
        }
        var img = new RenderTargetBitmap(result.White.PixelWidth, result.White.PixelHeight, 96, 96,
                                         PixelFormats.Pbgra32);
        img.Render(vis);
        WritePng(Path.Combine(dir, "overlay.png"), img);
        File.WriteAllText(Path.Combine(dir, "status.txt"), result.Report.Status + Environment.NewLine +
                                                               result.Report.Millis.ToString("F1") + " ms" + Environment.NewLine +
                          (result.Adaptive ? $"自适应布底：残差阈值 {result.Threshold}，去噪面积 {result.Settings.FabricNoiseArea}，小点保护分差 {result.Settings.FabricProtection}" : $"分类阈值 {result.Threshold}，去噪面积 {result.Settings.NoiseArea}，" +
                          $"补洞面积 {result.Settings.HoleArea}，补洞容差 {result.Settings.HoleAllowance}"));
    }
}
