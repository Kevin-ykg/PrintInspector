using System.Windows;
using System.Windows.Controls;

namespace PrintInspector;

public partial class MaskSettingsWindow : Window
{
    internal MaskSettings Settings { get; private set; }

    private readonly bool adaptive;

    internal MaskSettingsWindow(MaskSettings current, bool canRun, bool adaptive)
    {
        InitializeComponent();
        Settings = current;
        this.adaptive = adaptive;
        FabricSettings.Visibility = adaptive ? Visibility.Visible : Visibility.Collapsed;
        ForestSettings.Visibility = adaptive ? Visibility.Collapsed : Visibility.Visible;
        if (!canRun)
            ApplyButton.Content = "应用";
        ShowValues(current);
        Noise.TextChanged += ClearValidationMessage;
        Holes.TextChanged += ClearValidationMessage;
        Allowance.TextChanged += ClearValidationMessage;
        FabricThreshold.TextChanged += ClearValidationMessage;
        FabricArea.TextChanged += ClearValidationMessage;
        FabricProtection.TextChanged += ClearValidationMessage;
    }

    // 仅更新编辑框。取消或关闭窗口时，正在使用的设置和上一张结果都应保留。
    void ShowValues(MaskSettings value)
    {
        Noise.Text = value.NoiseArea.ToString();
        Holes.Text = value.HoleArea.ToString();
        Allowance.Text = value.HoleAllowance.ToString();
        FabricThreshold.Text = value.FabricThreshold.ToString();
        FabricArea.Text = value.FabricNoiseArea.ToString();
        FabricProtection.Text = value.FabricProtection.ToString();
        ErrorText.Text = "";
    }

    void ClearValidationMessage(object sender, TextChangedEventArgs e) => ErrorText.Text = "";

    void RestoreClick(object sender, RoutedEventArgs e) => ShowValues(MaskSettings.Default);

    bool ReadValue(TextBox box, string name, int maximum, out int value, int minimum = 0)
    {
        if (int.TryParse(box.Text, out value) && value >= minimum && value <= maximum)
            return true;

        ErrorText.Text = $"{name}需填写 {minimum}～{maximum} 的整数。";
        box.Focus();
        box.SelectAll();
        return false;
    }

    void ApplyClick(object sender, RoutedEventArgs e)
    {
        // 全部字段通过校验后才提交整组参数，防止一次运行混用新旧值。
        if (adaptive)
        {
            if (!ReadValue(FabricThreshold, "前景阈值", 254, out int threshold, 1) ||
                !ReadValue(FabricArea, "去噪面积", 8192, out int area) ||
                !ReadValue(FabricProtection, "小点保护分差", 127, out int protection, 1))
                return;
            Settings = Settings with
            {
                FabricThreshold = (byte)threshold,
                FabricNoiseArea = area,
                FabricProtection = protection
            };
        }
        else
        {
            if (!ReadValue(Noise, "去噪面积", 1024, out int noise) ||
                !ReadValue(Holes, "补洞面积", 8192, out int holes) ||
                !ReadValue(Allowance, "补洞证据容差", 255, out int allowance))
                return;
            Settings = Settings with { NoiseArea = noise, HoleArea = holes, HoleAllowance = allowance };
        }
        DialogResult = true;
    }
}
