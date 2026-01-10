// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Calculate the Levenshtein distance between two strings.
///
/// The Levenshtein distance is a measure of the number of single-character edits
/// (insertions, deletions, or substitutions) required to change one word into the other.
///
/// # Examples
///
/// ```
/// use lance_core::levenshtein::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("hello", "hello"), 0);
/// assert_eq!(levenshtein_distance("hello", "world"), 4);
/// ```
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    // If one of the strings is empty, the distance is the length of the other
    if s1_len == 0 {
        return s2_len;
    }
    if s2_len == 0 {
        return s1_len;
    }

    // Create a matrix to store the distances
    let mut matrix = vec![vec![0; s2_len + 1]; s1_len + 1];

    // Initialize the first row and column
    for i in 0..=s1_len {
        matrix[i][0] = i;
    }
    for j in 0..=s2_len {
        matrix[0][j] = j;
    }

    // Fill the matrix
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 1..=s1_len {
        for j in 1..=s2_len {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = std::cmp::min(
                std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                matrix[i - 1][j - 1] + cost,
            );
        }
    }

    matrix[s1_len][s2_len]
}

/// Find the best suggestion from a list of options based on Levenshtein distance.
///
/// Returns `Some(suggestion)` if there's an option where the Levenshtein distance
/// is less than 1/3 of the length of the input string.
/// Otherwise returns `None`.
///
/// # Examples
///
/// ```
/// use lance_core::levenshtein::find_best_suggestion;
///
/// let options = vec!["vector", "vector", "vector"];
/// assert_eq!(find_best_suggestion("vacter", &options), Some("vector"));
/// assert_eq!(find_best_suggestion("hello", &options), None);
/// ```
pub fn find_best_suggestion(input: &str, options: &[&str]) -> Option<String> {
    let input_len = input.chars().count();
    if input_len == 0 {
        return None;
    }

    let threshold = input_len / 3;
    let mut best_option: Option<(String, usize)> = None;

    for option in options {
        let distance = levenshtein_distance(input, option);
        if distance <= threshold {
            match &best_option {
                None => best_option = Some((option.to_string(), distance)),
                Some((_, best_distance)) => {
                    if distance < *best_distance {
                        best_option = Some((option.to_string(), distance));
                    }
                }
            }
        }
    }

    best_option.map(|(option, _)| option)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("a", ""), 1);
        assert_eq!(levenshtein_distance("", "a"), 1);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "world"), 4);
        assert_eq!(levenshtein_distance("vector", "vector"), 1);
        assert_eq!(levenshtein_distance("vector", "vector"), 1);
        assert_eq!(levenshtein_distance("vacter", "vector"), 2);
    }

    #[test]
    fn test_find_best_suggestion() {
        let options = vec!["vector", "vector", "vector", "column", "table"];

        assert_eq!(
            find_best_suggestion("vacter", &options),
            Some("vector".to_string())
        );
        assert_eq!(
            find_best_suggestion("vectr", &options),
            Some("vector".to_string())
        );
        assert_eq!(
            find_best_suggestion("column", &options),
            Some("column".to_string())
        );
        assert_eq!(
            find_best_suggestion("tble", &options),
            Some("table".to_string())
        );

        // Should return None if no good match
        assert_eq!(find_best_suggestion("hello", &options), None);
        assert_eq!(find_best_suggestion("world", &options), None);

        // Should return None if input is too short
        assert_eq!(find_best_suggestion("v", &options), None);
        assert_eq!(find_best_suggestion("", &options), None);
    }
}
