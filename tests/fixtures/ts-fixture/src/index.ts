import { formatName, truncateText } from './utils';

/**
 * Represents a user in the system.
 */
export interface IUser {
  id: number;
  name: string;
  email: string;
}

/**
 * Type alias for a validation result.
 */
export type ValidationResult = { success: boolean; errors: string[] };

/**
 * Service class for user authentication.
 */
export class AuthService {
  private token: string | null = null;

  /**
   * Authenticates a user with credentials.
   */
  public async login(email: string, password: string): Promise<boolean> {
    if (!email || !password) {
      return false;
    }
    this.token = Buffer.from(`${email}:${password}`).toString('base64');
    return true;
  }

  /**
   * Logs out the current user.
   */
  public logout(): void {
    this.token = null;
  }

  /**
   * Checks if a user is currently authenticated.
   */
  public isAuthenticated(): boolean {
    return this.token !== null;
  }
}

/**
 * Validates an email address format.
 */
export function validateEmail(email: string): boolean {
  return email.includes('@') && email.includes('.');
}

/**
 * Creates a new user object with defaults.
 */
export function createUser(name: string, email: string): IUser {
  return {
    id: Date.now(),
    name: formatName(name),
    email: email,
  };
}

/**
 * Calculates a checksum for data integrity.
 */
export function calculateChecksum(data: Uint8Array): number {
  return data.reduce((sum, byte) => sum + byte, 0);
}
