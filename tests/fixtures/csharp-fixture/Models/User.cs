using System;

namespace FixtureApp.Models;

/// <summary>
/// Represents a user in the system.
/// </summary>
public class User
{
    /// <summary>
    /// Gets or sets the user identifier.
    /// </summary>
    public long Id { get; set; }

    /// <summary>
    /// Gets or sets the user name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the user email.
    /// </summary>
    public string Email { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the user status.
    /// </summary>
    public UserStatus Status { get; set; }

    /// <summary>
    /// Returns a formatted display string.
    /// </summary>
    public string GetDisplayName()
    {
        return $"{Name} <{Email}>";
    }
}

/// <summary>
/// Possible user status values.
/// </summary>
public enum UserStatus
{
    Active,
    Inactive,
    Suspended
}
