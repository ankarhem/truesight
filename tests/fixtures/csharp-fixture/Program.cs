using System;
using AuthService = FixtureApp.Services.AuthService;
using User = FixtureApp.Models.User;

namespace FixtureApp;

/// <summary>
/// Main entry point for the fixture application.
/// </summary>
public class Program
{
    /// <summary>
    /// Application status enumeration.
    /// </summary>
    public enum AppStatus
    {
        Running,
        Stopped,
        Error
    }

    /// <summary>
    /// Main entry point.
    /// </summary>
    public static void Main(string[] args)
    {
        var status = AppStatus.Running;
        Console.WriteLine($"Application status: {status}");
    }

    /// <summary>
    /// Validates command line arguments.
    /// </summary>
    public static bool ValidateArgs(string[] args)
    {
        return args != null && args.Length > 0;
    }

    /// <summary>
    /// Parses configuration from arguments.
    /// </summary>
    public static string ParseConfig(string[] args)
    {
        if (args == null || args.Length == 0)
        {
            return "default";
        }
        return string.Join(",", args);
    }
}
