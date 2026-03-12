using System;

namespace FixtureApp.Services;

/// <summary>
/// Service for handling user authentication.
/// </summary>
public class AuthService
{
    private string? _token;

    /// <summary>
    /// Authenticates a user with credentials.
    /// </summary>
    public bool Login(string email, string password)
    {
        if (string.IsNullOrEmpty(email) || string.IsNullOrEmpty(password))
        {
            return false;
        }
        _token = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes($"{email}:{password}"));
        return true;
    }

    /// <summary>
    /// Logs out the current user.
    /// </summary>
    public void Logout()
    {
        _token = null;
    }

    /// <summary>
    /// Checks if a user is currently authenticated.
    /// </summary>
    public bool IsAuthenticated()
    {
        return !string.IsNullOrEmpty(_token);
    }

    /// <summary>
    /// Validates an email address format.
    /// </summary>
    public static bool ValidateEmail(string email)
    {
        return !string.IsNullOrEmpty(email) && email.Contains('@') && email.Contains('.');
    }
}

/// <summary>
/// Interface for authentication providers.
/// </summary>
public interface IAuthProvider
{
    bool Login(string email, string password);
    void Logout();
    bool IsAuthenticated();
}
