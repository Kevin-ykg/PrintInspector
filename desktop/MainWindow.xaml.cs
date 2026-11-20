using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace PrintInspector;

internal sealed record Config(int route, int pca, Dictionary<string, byte> thresholds);
public partial class MainWindow : Window
{
    string root = "", shown = "";
    Output? current;
    BitmapSource? source;
    CancellationTokenSource? cancel;
    Config? config;
    bool busy;
    MaskSettings maskSettings = MaskSettings.Default;
    static string Models => Path.Combine(AppContext.BaseDirectory, "models");
    public MainWindow()
    {
        InitializeComponent();
        Band.ItemsSource = Engine.Bands;
        Band.SelectedIndex = 7;
        Layer.ItemsSource = new[] { "轮廓叠加", "二值掩膜", "前景得分" };
        Layer.SelectedIndex = 0;
        Route.ItemsSource = new[] { "八波段随机森林", "PCA 随机森林" };
        Route.SelectedIndex = 0;
        Route.SelectionChanged += (_, _) => ClearResult();
        try
        {
            config =
                JsonSerializer.Deserialize<Config>(File.ReadAllText(Path.Combine(Models, "settings.json")));
        }
        catch (Exception e)
        {
            Status.Text = "模型配置未就绪 · " + e.Message;
        }
        // 显式传入样品目录时优先使用；从 IDE 启动时沿输出路径向上寻找默认无膜片图库。
        Loaded += (_, _) =>
        {
            var supplied = Environment.GetCommandLineArgs().Skip(1).FirstOrDefault();
            if (supplied != null && Directory.Exists(supplied))
                OpenFolder(Path.GetFullPath(supplied));
            else
            {
                for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
                {
                    var data = Path.Combine(dir.FullName, "锐视光电", "没有膜片");
                    if (Directory.Exists(data))
                    {
                        OpenFolder(data);
                        break;
                    }
                }
            }
        };
        // 推理尚未交回模型句柄时，只发取消信号，等本次收尾后再允许关闭窗口。
        Closing += (_, e) =>
        {
            if (busy)
            {
                cancel?.Cancel();
                e.Cancel = true;
            }
        };
    }

    // 样品或模型切换后，旧掩膜与当前选择已不对应，必须同步清空预览和导出状态。
    void ClearResult()
    {
        current = null;
        ResultView.Source = Edges.Source = null;
        Export.IsEnabled = false;
    }

    void BrowseClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog();
        if (dlg.ShowDialog() != true)
            return;
        OpenFolder(dlg.FolderName);
    }

    /// <summary>支持图库根目录、膜片分组目录及单个样品目录。列表中的编号对应实际文件夹名。</summary>
    internal void OpenFolder(string folder)
    {
        root = Directory.Exists(Path.Combine(folder, "没有膜片")) ? Path.Combine(folder, "没有膜片") : folder;
        var dirs = Directory.GetDirectories(root).Where(d => Directory.EnumerateFiles(d, "*.bmp").Any()).ToList();
        if (Directory.EnumerateFiles(root, "*.bmp").Any())
        {
            dirs = [root];
            root = Path.GetDirectoryName(root)!;
        }
        // 打开图库时恢复生产默认，防止膜片试验的参数带入下一批无膜片样品。
        SetMaskSettings(MaskSettings.Default);
        Groups.ItemsSource = dirs.Select(Path.GetFileName)
                                 .OrderBy(s => int.TryParse(s, out int i) ? i : int.MaxValue)
                                 .ThenBy(s => s)
                                 .ToArray();
        Dataset.Text = "样品目录  " + root;
        Groups.SelectedIndex = 0;
        if (Groups.Items.Count == 0)
            Status.Text = "目录中没有样品组";
    }

    void GroupChanged(object sender, RoutedEventArgs e)
    {
        ResetView();
        ClearResult();
        shown = Groups.SelectedItem is string name ? Path.Combine(root, name) : "";
        ReadBand();
    }

    void ReadBand()
    {
        try
        {
            source = Directory.Exists(shown) ? Engine.Read(shown, Band.SelectedItem as string ?? "W") : null;
            Render();
        }
        catch (Exception e)
        {
            source = null;
            Render();
            Status.Text = e.Message;
        }
    }

    void ViewChanged(object sender, RoutedEventArgs e)
    {
        if (!IsInitialized || Edges == null)
            return;
        if (sender == Band)
            ReadBand();
        else
            Render();
    }

    void Render()
    {
        Original.Source = source;
        // 分数图直接显示森林投票；只有轮廓模式才把二值结果绘到当前波段底图上。
        ResultView.Source = Layer.SelectedIndex switch
        {
            1 => current?.Mask,
            2 => current?.Score,
            _ => current == null ? null : source
        };
        Edges.Source = Layer.SelectedIndex == 0 ? current?.Contour : null;
        if (source != null)
        {
            Original.Width = ResultContent.Width = source.PixelWidth;
            Original.Height = ResultContent.Height = source.PixelHeight;
        }
        ApplyView();
    }

    // 两个画框共用归一化图像中心和倍率，缩放锚点始终对应同一原图像素。
    double zoom = 1;
    Point center = new(0.5, 0.5);
    Canvas? dragging;
    Point lastPointer;

    internal void ResetView()
    {
        zoom = 1;
        center = new Point(0.5, 0.5);
        ApplyView();
    }

    double PixelsToDip()
    {
        if (source == null || ResultPane == null)
            return 0;
        return zoom * Math.Min(Math.Min(OriginalPane.ActualWidth, ResultPane.ActualWidth) / source.PixelWidth,
                               Math.Min(OriginalPane.ActualHeight, ResultPane.ActualHeight) / source.PixelHeight);
    }

    void ApplyView()
    {
        double scale = PixelsToDip();
        if (scale <= 0 || source == null)
            return;
        MatrixTransform Transform(Canvas pane) => new(scale, 0, 0, scale,
            pane.ActualWidth * 0.5 - center.X * source.PixelWidth * scale,
            pane.ActualHeight * 0.5 - center.Y * source.PixelHeight * scale);
        Original.RenderTransform = Transform(OriginalPane);
        ResultContent.RenderTransform = Transform(ResultPane);
    }

    void PaneSizeChanged(object sender, SizeChangedEventArgs e) => ApplyView();

    // 将指针由画框 DIP 反算到图像坐标，更新倍率后再反求中心，使笔画留在指针下。
    internal void ZoomAt(Canvas pane, Point pointer, double steps)
    {
        double before = PixelsToDip();
        if (before <= 0 || source == null)
            return;
        zoom = Math.Clamp(zoom * Math.Pow(1.25, steps), 0.25, 40);
        double delta = 1 / before - 1 / PixelsToDip();
        center = new Point(center.X + (pointer.X - pane.ActualWidth / 2) * delta / source.PixelWidth,
                           center.Y + (pointer.Y - pane.ActualHeight / 2) * delta / source.PixelHeight);
        ApplyView();
    }

    internal void PanBy(Vector movement)
    {
        double scale = PixelsToDip();
        if (scale <= 0 || source == null)
            return;
        center = new Point(Math.Clamp(center.X - movement.X / (scale * source.PixelWidth), 0, 1),
                           Math.Clamp(center.Y - movement.Y / (scale * source.PixelHeight), 0, 1));
        ApplyView();
    }

    void PaneWheel(object sender, MouseWheelEventArgs e)
    {
        var pane = (Canvas)sender;
        if (source == null)
            return;
        ZoomAt(pane, e.GetPosition(pane), e.Delta / 120.0);
        e.Handled = true;
    }

    void PaneDown(object sender, MouseButtonEventArgs e)
    {
        if (source == null)
            return;
        var pane = (Canvas)sender;
        if (e.ClickCount == 2)
            ResetView();
        else if (pane.CaptureMouse())
        {
            dragging = pane;
            lastPointer = e.GetPosition(pane);
            pane.Cursor = Cursors.SizeAll;
        }
        e.Handled = true;
    }

    void PaneMove(object sender, MouseEventArgs e)
    {
        if (dragging == null || sender != dragging)
            return;
        if (e.LeftButton != MouseButtonState.Pressed)
        {
            dragging.ReleaseMouseCapture();
            return;
        }
        Point pointer = e.GetPosition(dragging);
        PanBy(pointer - lastPointer);
        lastPointer = pointer;
    }

    void PaneUp(object sender, MouseButtonEventArgs e)
    {
        if (dragging != null && dragging == sender)
            dragging.ReleaseMouseCapture();
    }

    void PaneLostCapture(object sender, MouseEventArgs e)
    {
        ((Canvas)sender).ClearValue(CursorProperty);
        dragging = null;
    }

    // 参数是当前图库的临时设置；按钮标出是否偏离默认，便于批量操作前核对。
    void SetMaskSettings(MaskSettings settings)
    {
        maskSettings = settings;
        Tune.Content = settings == MaskSettings.Default ? "掩膜参数…" : "掩膜参数 · 已调整";
        Tune.ToolTip = $"去噪面积 {settings.NoiseArea}，补洞面积 {settings.HoleArea}，补洞容差 {settings.HoleAllowance}";
    }

    async void TuneClick(object sender, RoutedEventArgs e)
    {
        if (busy)
            return;
        bool canRun = config != null && Groups.SelectedItem != null;
        var dialog = new MaskSettingsWindow(maskSettings, canRun) { Owner = this };
        if (dialog.ShowDialog() != true)
            return;

        SetMaskSettings(dialog.Settings);
        ClearResult();
        if (canRun)
            await Run(null);
    }

    async void SingleClick(object sender, RoutedEventArgs e) => await Run(null);

    async void BatchClick(object sender, RoutedEventArgs e)
    {
        if (busy || config == null || Groups.SelectedItem == null)
            return;

        var dialog = new OpenFolderDialog { Title = "选择批量结果目录" };
        if (dialog.ShowDialog() == true)
        {
            string folder = "分割-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff");
            await Run(Path.Combine(dialog.FolderName, folder));
        }
    }

    void StopClick(object sender, RoutedEventArgs e) => cancel?.Cancel();

    /// <summary>在给定目录逐组导出批量结果。目录为空时只预览当前样品。</summary>
    async Task Run(string? dest)
    {
        if (busy || config == null || Groups.SelectedItem == null)
            return;

        bool all = dest != null;
        // 开始时固定处理顺序。一批只加载一次模型，每完成一组即导出，取消时已完成的组仍可查看。
        var settings = maskSettings;
        var names = all ? Groups.Items.Cast<string>().ToArray() : new[] { (string)Groups.SelectedItem };
        using var cts = new CancellationTokenSource();
        cancel = cts;
        busy = true;
        Tune.IsEnabled = Browse.IsEnabled = Route.IsEnabled = Groups.IsEnabled = Single.IsEnabled = Batch.IsEnabled = false;
        Stop.IsEnabled = true;
        int ok = 0;
        try
        {
            int route = Route.SelectedIndex == 1 ? config.pca : 8;
            using var model = Engine.Load(Path.Combine(Models, $"rf{route}.yml"));
            foreach (var name in names)
            {
                cts.Token.ThrowIfCancellationRequested();
                Groups.SelectedItem = name;
                ClearResult();
                Status.Text = $"处理 {name} · {ok}/{names.Length}";
                string path = Path.Combine(root, name);
                try
                {
                    var sw = Stopwatch.StartNew();
                    // 解码和 DLL 推理在工作线程运行；冻结的位图可在 await 返回后交给 WPF。
                    current = await Task.Run(
                        () => Engine.Run(path, model, config.thresholds[route.ToString()], cts.Token, settings));
                    Render();
                    Export.IsEnabled = true;
                    Status.Text =
                        $"{name} · {current.Report.Status} · DLL {current.Report.Millis:F0} ms · 到预览 {sw.ElapsedMilliseconds} ms";
                    if (dest != null)
                        Engine.Save(Path.Combine(dest, name), current);
                    ok++;
                }
                // 单组错误立即呈现，批量则按编号记录并继续；显式取消交给外层统一收尾。
                catch (Exception ex) when (all && ex is not OperationCanceledException)
                {
                    Directory.CreateDirectory(dest!);
                    File.AppendAllText(Path.Combine(dest!, "errors.txt"),
                                       name + " " + ex.Message + Environment.NewLine);
                }
            }
            if (all)
                Status.Text = $"完成 {ok}/{names.Length} 组 · {dest}";
        }
        catch (OperationCanceledException)
        {
            Status.Text = $"已取消 · 完成 {ok} 组";
        }
        catch (Exception e)
        {
            Status.Text = e.Message;
        }
        finally
        {
            busy = false;
            cancel = null;
            Tune.IsEnabled = Browse.IsEnabled = Route.IsEnabled = Groups.IsEnabled = Single.IsEnabled = Batch.IsEnabled = true;
            Stop.IsEnabled = false;
        }
    }

    void ExportClick(object sender, RoutedEventArgs e)
    {
        if (current == null)
            return;
        var dlg = new OpenFolderDialog();
        if (dlg.ShowDialog() != true)
            return;
        try
        {
            var dest = Path.Combine(dlg.FolderName, Path.GetFileName(shown) + "-" +
                                                        DateTime.Now.ToString("yyyyMMdd-HHmmss-fff"));
            Engine.Save(dest, current);
            Status.Text = "已导出 " + dest;
        }
        catch (Exception ex)
        {
            Status.Text = ex.Message;
        }
    }
}
