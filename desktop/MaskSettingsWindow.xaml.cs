using System.Windows;
using System.Windows.Controls;

namespace PrintInspector;

public partial class MaskSettingsWindow : Window
{
    internal MaskSettings Settings { get; private set; }

    internal MaskSettingsWindow(MaskSettings current, bool canRun)
    {
        InitializeComponent();
        Settings = current;
        if (!canRun)
            ApplyButton.Content = "应用";
        ShowValues(current);
        Noise.TextChanged += ClearValidationMessage;
        Holes.TextChanged += ClearValidationMessage;
        Allowance.TextChanged += ClearValidationMessage;
    }

    // 仅更新编辑框。取消或关闭窗口时，正在使用的设置和上一张结果都应保留。
    void ShowValues(MaskSettings value)
    {
        Noise.Text = value.NoiseArea.ToString();
        Holes.Text = value.HoleArea.ToString();
        Allowance.Text = value.HoleAllowance.ToString();
        ErrorText.Text = "";
    }

    void ClearValidationMessage(object sender, TextChangedEventArgs e) => ErrorText.Text = "";

    void RestoreClick(object sender, RoutedEventArgs e) => ShowValues(MaskSettings.Default);

    bool ReadValue(TextBox box, string name, int maximum, out int value)
    {
        if (int.TryParse(box.Text, out value) && value >= 0 && value <= maximum)
            return true;

        ErrorText.Text = $"{name}需填写 0～{maximum} 的整数。";
        box.Focus();
        box.SelectAll();
        return false;
    }

    void ApplyClick(object sender, RoutedEventArgs e)
    {
        // 全部字段通过校验后才提交整组参数，防止一次运行混用新旧值。
        if (!ReadValue(Noise, "去噪面积", 1024, out int noise) ||
            !ReadValue(Holes, "补洞面积", 8192, out int holes) ||
            !ReadValue(Allowance, "补洞证据容差", 255, out int allowance))
            return;

        Settings = new MaskSettings(noise, holes, allowance);
        DialogResult = true;
    }
}
