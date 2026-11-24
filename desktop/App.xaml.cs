using System.Windows;

namespace PrintInspector;

public partial class App : Application
{
    // 自动验证与正常启动共用 WPF 资源；命令行分支用退出码和错误文件报告结果。
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        if (e.Args.Length == 3 && e.Args[0] is "--benchmark" or "--fabric-benchmark" or "--ui-smoke" or "--fabric-smoke" or "--verify-fabric-export")
        {
            try
            {
                if (e.Args[0] is "--ui-smoke" or "--fabric-smoke")
                    Bench.VerifyWindow(e.Args[1], e.Args[2], e.Args[0] == "--fabric-smoke");
                else if (e.Args[0] == "--verify-fabric-export")
                    Bench.VerifyPngExport(e.Args[1], e.Args[2]);
                else
                    Bench.Run(e.Args[1], e.Args[2], e.Args[0] == "--fabric-benchmark");
                Shutdown(0);
            }
            catch (Exception ex)
            {
                System.IO.File.WriteAllText(e.Args[2] + ".error", ex.ToString());
                Shutdown(1);
            }
        }
        else
            new MainWindow().Show();
    }
}
