using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
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
                (Flags & 8) != 0 ? "底布参照不足" : null
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
internal sealed record MaskSettings(int NoiseArea, int HoleArea, int HoleAllowance)
{
    internal static readonly MaskSettings Default = new(96, 80, 46);
}

internal sealed record Output(BitmapSource White, BitmapSource Mask, BitmapSource Score, BitmapSource Contour,
                              Report Report, byte Threshold, MaskSettings Settings, double ReadMilliseconds, int[] OffsetsXY);
internal static class Engine
{
    public static readonly string[] Bands = ["UV", "B", "G", "AM", "R", "FR", "IR", "W"];
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

    /// <summary>读取八位灰度帧；线性灰阶索引可直接取值，其他调色板先转换。关闭文件前加载并冻结像素，供工作线程和 UI 共用。</summary>
    internal static BitmapSource Read(string folder, string band)
    {
        using var f = File.OpenRead(Path.Combine(folder, band + ".bmp"));
        var src = new BmpBitmapDecoder(f, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad)
                      .Frames[0];
        if (src.Format.BitsPerPixel != 8)
            throw new IOException("需要 8 位灰度图 " + band);
        // 线性灰阶调色板的索引就是像素值；彩色或非线性调色板仍交给 WIC 转换。
        BitmapSource gray = src;
        if (src.Format != PixelFormats.Gray8 && !(src.Format == PixelFormats.Indexed8 &&
            src.Palette is { } palette && palette.Colors.Select((c, i) =>
                c.A == 255 && c.R == i && c.G == i && c.B == i).All(identity => identity)))
            gray = new FormatConvertedBitmap(src, PixelFormats.Gray8, null, 0);
        gray.Freeze();
        return gray;
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
    internal static Output Run(string dir, ModelHandle model, byte threshold, CancellationToken token, MaskSettings? settings = null)
    {
        // Plane 仅借用像素地址。八个数组在本次原生调用结束前都必须固定，异常路径也需释放。
        settings ??= MaskSettings.Default;
        var pins = new List<GCHandle>();
        var planes = new Plane[8];
        int w = 0, h = 0;
        var cancel = Marshal.AllocHGlobal(4);
        Marshal.WriteInt32(cancel, 0);
        try
        {
            // 取消回调只能在标志仍有效时写入；using 的作用域先于 finally 中的 FreeHGlobal 结束。
            using var registration = token.Register(() => Marshal.WriteInt32(cancel, 1));
            long readStart = Stopwatch.GetTimestamp();
            BitmapSource? white = null;
            for (int b = 0; b < 8; b++)
            {
                token.ThrowIfCancellationRequested();
                var im = Read(dir, Bands[b]);
                if (b == 0)
                {
                    w = im.PixelWidth;
                    h = im.PixelHeight;
                }
                if (im.PixelWidth != w || im.PixelHeight != h)
                    throw new IOException("波段尺寸不一致");
                byte[] a = new byte[checked(w * h)];
                im.CopyPixels(a, w, 0);
                var pin = GCHandle.Alloc(a, GCHandleType.Pinned);
                pins.Add(pin);
                planes[b] = new()
                {
                    Data = pin.AddrOfPinnedObject(),
                    Size = a.Length,
                    Width = w,
                    Height = h,
                    Stride = w,
                    Band = b
                };
                if (b == 7)
                    white = im;
            }
            double readMilliseconds = Stopwatch.GetElapsedTime(readStart).TotalMilliseconds;
            byte[] m = new byte[checked(w * h)], s = new byte[m.Length];
            var offsetsXY = new int[16];
            var rc = ms_run_registered(model, planes, 8, threshold, m, s, m.Length, out var report, cancel,
                                       settings.NoiseArea, settings.HoleArea, settings.HoleAllowance, offsetsXY);
            token.ThrowIfCancellationRequested();
            Check(rc);
            // 原生函数已经归还八帧指针，生成预览时可以解除固定。
            foreach (var pin in pins)
                pin.Free();
            pins.Clear();
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
                       readMilliseconds, offsetsXY);
        }
        finally
        {
            foreach (var p in pins)
                p.Free();
            Marshal.FreeHGlobal(cancel);
        }
    }

    // RenderTargetBitmap 需要单线程单元；让磁盘写入、整幅渲染与 PNG 编码离开界面线程。
    internal static Task SaveAsync(string dir, Output result)
    {
        var finished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            try
            {
                try
                {
                    Save(dir, result);
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

    /// <summary>按相机像素尺寸导出；叠加底图始终使用 W，观察窗口的缩放和选定波段均不参与导出。</summary>
    internal static void Save(string dir, Output result)
    {
        Directory.CreateDirectory(dir);
        void Png(string name, BitmapSource frame)
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(frame));
            var target = Path.Combine(dir, name);
            var temp = target + "." + Guid.NewGuid() + ".tmp";
            try
            {
                using (var f = File.Create(temp)) encoder.Save(f);
                File.Move(temp, target, true);
            }
            finally
            {
                if (File.Exists(temp))
                    File.Delete(temp);
            }
        }
        Png("mask.png", result.Mask);
        Png("score.png", result.Score);
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
        Png("overlay.png", img);
        File.WriteAllText(Path.Combine(dir, "status.txt"), result.Report.Status + Environment.NewLine +
                                                               result.Report.Millis.ToString("F1") + " ms" + Environment.NewLine +
                          $"分类阈值 {result.Threshold}，去噪面积 {result.Settings.NoiseArea}，" +
                          $"补洞面积 {result.Settings.HoleArea}，补洞容差 {result.Settings.HoleAllowance}");
    }
}
