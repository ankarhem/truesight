/**
 * Formats a name to title case.
 */
export function formatName(name: string): string {
  let result = '';
  let capitalizeNext = true;
  for (const char of name) {
    if (char === ' ') {
      capitalizeNext = true;
      result += char;
    } else if (capitalizeNext) {
      result += char.toUpperCase();
      capitalizeNext = false;
    } else {
      result += char.toLowerCase();
    }
  }
  return result;
}

/**
 * Truncates text to a maximum length.
 */
export function truncateText(text: string, maxLen: number): string {
  if (text.length <= maxLen) {
    return text;
  }
  return text.substring(0, maxLen - 3) + '...';
}

/**
 * Checks if a string is blank.
 */
export function isBlank(s: string): boolean {
  return s.trim().length === 0;
}
