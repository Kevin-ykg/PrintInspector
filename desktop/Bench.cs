using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace PrintInspector;

internal static class Bench
{
    // 用真实解码、原生推理和 WPF 离屏渲染覆盖操作路径。
    // 该计时不包含屏幕呈现和磁盘导出。
    internal static void VerifyWindow(string root, string output)
    {
        var window = new MainWindow();
        window.OpenFolder(Path.GetFullPath(root));
        var groups = (ListBox)window.FindName("Groups");
        groups.SelectedIndex = 0;
        var run = (Button)window.FindName("Single");
        run.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        var status = (TextBlock)window.FindName("Status");
        // 验证按钮会异步启动推理，保持消息循环才能等到 UI 线程完成预览更新。
        var frame = new System.Windows.Threading.DispatcherFrame();
        var watch = Stopwatch.StartNew();
        var timer = new System.Windows.Threading.DispatcherTimer();
        timer.Interval = TimeSpan.FromMilliseconds(50);
        timer.Tick += (_, _) =>
        {
            if (run.IsEnabled || watch.Elapsed.TotalSeconds > 120)
                frame.Continue = false;
        };
        timer.Start();
        System.Windows.Threading.Dispatcher.PushFrame(frame);
        timer.Stop();
        if (!status.Text.Contains("到预览"))
            throw new IOException("UI smoke test: " + status.Text);
        var band = (ComboBox)window.FindName("Band");
        band.SelectedItem = "IR";
        band.SelectedItem = "W";
        var layer = (ComboBox)window.FindName("Layer");
        layer.SelectedIndex = 1;
        layer.SelectedIndex = 2;
        layer.SelectedIndex = 0;
        var content = (FrameworkElement)window.Content;
        content.Measure(new Size(1280, 800));
        content.Arrange(new Rect(0, 0, 1280, 800));
        content.UpdateLayout();
        VerifyBandCoordinates();
        VerifyNavigation(window);
        var bitmap = new RenderTargetBitmap(1280, 800, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(content);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using (var stream = File.Create(output)) encoder.Save(stream);
        window.Close();
    }

    // 检查位移正负号、左右行步长和越界填零；实拍图库可能恰好全部取整为零。
    private static void VerifyBandCoordinates()
    {
        const int w = 13, h = 11;
        var input = Enumerable.Range(1, w * h).Select(i => (byte)i).ToArray();
        var src = BitmapSource.Create(w, h, 96, 96, PixelFormats.Gray8, null, input, w);
        foreach (var (dx, dy) in new[] { (2, -3), (-2, 3), (0, 0) })
        {
            var bytes = new byte[w * h];
            Engine.InWhiteCoordinates(src, dx, dy).CopyPixels(bytes, w, 0);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    int sx = x + dx, sy = y + dy;
                    int expected = sx >= 0 && sx < w && sy >= 0 && sy < h ? input[sy * w + sx] : 0;
                    if (bytes[y * w + x] != expected)
                        throw new IOException("波段预览与 W 坐标不一致");
                }
        }
    }

    // 用完成布局后的两个画框及叠加层父控件检查坐标变换。
    private static void VerifyNavigation(MainWindow window)
    {
        var left = (Canvas)window.FindName("OriginalPane");
        var right = (Canvas)window.FindName("ResultPane");
        var image = (Image)window.FindName("Original");
        var result = (Grid)window.FindName("ResultContent");
        if (!left.ClipToBounds || !right.ClipToBounds || left.ActualWidth <= 0 || right.ActualWidth <= 0)
            throw new IOException("Image viewports have invalid clipping or size");

        // 以窗口内容为共同坐标系，核对操作栏移动后两个画框的阅读顺序。
        var root = (FrameworkElement)window.Content;
        var groups = (ListBox)window.FindName("Groups");
        if (left.TranslatePoint(new Point(left.ActualWidth, 0), root).X > right.TranslatePoint(new Point(), root).X ||
            right.TranslatePoint(new Point(right.ActualWidth, 0), root).X > groups.TranslatePoint(new Point(), root).X)
            throw new IOException("原图、结果、操作栏的排列顺序异常");

        var pixel = new Point(image.Width * 0.43, image.Height * 0.47);
        Point leftStart = image.RenderTransform.Transform(pixel);
        Point rightStart = result.RenderTransform.Transform(pixel);
        double initialScale = image.RenderTransform.Value.M11;
        window.ZoomAt(right, rightStart, 2);
        if ((result.RenderTransform.Transform(pixel) - rightStart).Length > 1e-6 ||
            (image.RenderTransform.Transform(pixel) - leftStart).Length > 1e-6 ||
            image.RenderTransform.Value.M11 <= initialScale)
            throw new IOException("Pointer anchor or linked zoom failed");
        var delta = new Vector(31, -18);
        window.PanBy(delta);
        if ((image.RenderTransform.Transform(pixel) - leftStart - delta).Length > 1e-6 ||
            (result.RenderTransform.Transform(pixel) - rightStart - delta).Length > 1e-6)
            throw new IOException("Linked pan failed");
        window.PanBy(-delta);
        window.ZoomAt(left, leftStart, -2);
        if (Math.Abs(result.RenderTransform.Value.M11 - initialScale) > 1e-6)
            throw new IOException("Zoom-out did not restore the original scale");
        var wheel = new System.Windows.Input.MouseWheelEventArgs(
            System.Windows.Input.Mouse.PrimaryDevice, Environment.TickCount, 120)
        { RoutedEvent = System.Windows.Input.Mouse.MouseWheelEvent };
        left.RaiseEvent(wheel);
        if (!wheel.Handled || result.RenderTransform.Value.M11 <= initialScale)
            throw new IOException("Wheel routing failed");
        window.ResetView();
    }

    internal static void Run(string root, string output)
    {
        var models = Path.Combine(AppContext.BaseDirectory, "models");
        var cfg =
            JsonSerializer.Deserialize<Config>(File.ReadAllText(Path.Combine(models, "settings.json")))!;
        var window = new MainWindow();
        var lines = new List<string> {
            "route,pass,group,native_ms,read_to_render_ms,private_mib,peak_private_mib,flags,read_ms,preview_and_interop_ms"
        };
        var process = Process.GetCurrentProcess();
        long peak = 0;
        object gate = new();
        // 峰值来自进程私有内存的定时采样，包含托管堆、原生缓冲区和 WPF 渲染资源。
        using var sampler = new Timer(
            _ =>
            {
                lock (gate)
                {
                    process.Refresh();
                    peak = Math.Max(peak, process.PrivateMemorySize64);
                }
            },
            null, 0, 30);
        if (Directory.Exists(Path.Combine(root, "没有膜片")))
            root = Path.Combine(root, "没有膜片");
        foreach (var route in new[] { cfg.pca, 8 })
        {
            lock (gate)
            {
                peak = 0;
            }
            using var handle = Engine.Load(Path.Combine(models, $"rf{route}.yml"));
            int passes = route == 8 ? 2 : 1;
            for (int pass = 1; pass <= passes; pass++)
            {
                var ids = Directory.GetDirectories(root).Where(d => File.Exists(Path.Combine(d, "W.bmp"))).Select(d => new DirectoryInfo(d).Name);
                foreach (var id in ids)
                {
                    var sw = Stopwatch.StartNew();
                    var o = Engine.Run(Path.Combine(root, id.ToString()), handle,
                                       cfg.thresholds[route.ToString()], CancellationToken.None);
                    var visual = new DrawingVisual();
                    using (var ctx = visual.RenderOpen())
                    {
                        ctx.DrawImage(o.White, new Rect(0, 0, 480, 402));
                        ctx.DrawImage(o.White, new Rect(480, 0, 480, 402));
                        ctx.DrawImage(o.Contour, new Rect(480, 0, 480, 402));
                    }
                    var target = new RenderTargetBitmap(960, 402, 96, 96, PixelFormats.Pbgra32);
                    target.Render(visual);
                    var elapsed = sw.Elapsed.TotalMilliseconds;
                    // 差值含固定缓冲区之外的封送、轮廓生成、位图构造及离屏绘制。
                    var previewAndInterop = elapsed - o.ReadMilliseconds - o.Report.Millis;
                    lock (gate)
                    {
                        process.Refresh();
                        lines.Add(FormattableString.Invariant(
                            $"{route},{pass},{id},{o.Report.Millis:F2},{elapsed:F2},{process.PrivateMemorySize64 / 1048576.0:F1},{peak / 1048576.0:F1},{o.Report.Flags},{o.ReadMilliseconds:F2},{previewAndInterop:F2}"));
                    }
                    File.WriteAllLines(output, lines);
                }
            }
        }
        window.Close();
    }
}
