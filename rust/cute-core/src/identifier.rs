//! ASCII identifier classification + camelCase / snake_case sub-splitting.
//!
//! Mirrors `patterns.py:54` `split_identifier`. Implemented as a hand-rolled
//! state machine instead of regex because the Python pattern uses lookahead
//! `(?=...)` which the `regex` crate doesn't support.
//!
//! Behavior matches `_SUBPART_REGEX.findall` semantics: chars that don't
//! belong to ASCII letter / digit classes are silently skipped (matching
//! Python `re.findall` returning only matches), with the Python fallback of
//! "if no parts emitted, return the whole chunk" preserved.

/// True iff `s` matches the conservative ASCII identifier shape:
/// `^[A-Za-z_][A-Za-z0-9_]*$`. Mirrors `patterns.py:49`.
pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Split camelCase / PascalCase / snake_case / SCREAMING_CASE / digit runs.
///
/// `''.join(split_identifier(x))` reconstructs `x` minus underscores.
/// Examples (mirroring `patterns.py:54` doctests):
///
/// - `"myVar"` → `["my", "Var"]`
/// - `"HTTPRequestParser"` → `["HTTP", "Request", "Parser"]`
/// - `"MAX_BUFFER_SIZE"` → `["MAX", "BUFFER", "SIZE"]`
/// - `"get_user_id_42"` → `["get", "user", "id", "42"]`
/// - `""` → `[]`
pub fn split_identifier(ident: &str) -> Vec<String> {
    if ident.is_empty() {
        return Vec::new();
    }
    let mut parts: Vec<String> = Vec::new();
    for chunk in ident.split('_') {
        if chunk.is_empty() {
            continue;
        }
        split_chunk_into(chunk, &mut parts);
    }
    parts
}

/// In-place variant for callers (frequency counting) that want to avoid
/// allocating a fresh `Vec<String>` per identifier.
pub fn extend_split_identifier(ident: &str, out: &mut Vec<String>) {
    if ident.is_empty() {
        return;
    }
    for chunk in ident.split('_') {
        if chunk.is_empty() {
            continue;
        }
        split_chunk_into(chunk, out);
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Class {
    Other,
    Lower,
    Upper,
    Digit,
}

#[inline]
fn classify(b: u8) -> Class {
    match b {
        b'a'..=b'z' => Class::Lower,
        b'A'..=b'Z' => Class::Upper,
        b'0'..=b'9' => Class::Digit,
        _ => Class::Other,
    }
}

fn split_chunk_into(chunk: &str, out: &mut Vec<String>) {
    let bytes = chunk.as_bytes();
    let n = bytes.len();
    let initial_len = out.len();

    let mut i = 0usize;
    while i < n {
        let b = bytes[i];
        if !b.is_ascii() {
            // Non-ASCII char: walk past the full UTF-8 codepoint without
            // emitting (mirrors Python regex skipping non-matches).
            let ch = chunk[i..].chars().next().expect("char at byte boundary");
            i += ch.len_utf8();
            continue;
        }
        let c0 = classify(b);
        if matches!(c0, Class::Other) {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i + 1;
        match c0 {
            Class::Lower => {
                while end < n && classify(bytes[end]) == Class::Lower {
                    end += 1;
                }
            }
            Class::Upper => {
                while end < n && classify(bytes[end]) == Class::Upper {
                    end += 1;
                }
                // Acronym handling: an upper run of length > 1 followed by a
                // lower starts a new piece at the last upper char.
                // "HTTPRequest" → "HTTP", then "Request" starts at 'R'.
                if end < n && classify(bytes[end]) == Class::Lower && end - start > 1 {
                    end -= 1;
                }
                // Pascal pattern `[A-Z]?[a-z]+`: a single upper followed by
                // lowers absorbs them all.
                if end < n && classify(bytes[end]) == Class::Lower && end - start == 1 {
                    while end < n && classify(bytes[end]) == Class::Lower {
                        end += 1;
                    }
                }
            }
            Class::Digit => {
                while end < n && classify(bytes[end]) == Class::Digit {
                    end += 1;
                }
            }
            Class::Other => unreachable!(),
        }
        out.push(chunk[start..end].to_string());
        i = end;
    }

    // Python fallback: if findall yielded nothing, the whole chunk is kept.
    if out.len() == initial_len {
        out.push(chunk.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_empty() {
        assert_eq!(split_identifier(""), Vec::<String>::new());
        assert!(!is_identifier(""));
    }

    #[test]
    fn camel_case() {
        assert_eq!(split_identifier("myVar"), vec!["my", "Var"]);
    }

    #[test]
    fn pascal_with_acronym() {
        assert_eq!(
            split_identifier("HTTPRequestParser"),
            vec!["HTTP", "Request", "Parser"]
        );
    }

    #[test]
    fn screaming_snake_case() {
        assert_eq!(
            split_identifier("MAX_BUFFER_SIZE"),
            vec!["MAX", "BUFFER", "SIZE"]
        );
    }

    #[test]
    fn snake_with_digits() {
        assert_eq!(
            split_identifier("get_user_id_42"),
            vec!["get", "user", "id", "42"]
        );
    }

    #[test]
    fn is_identifier_basics() {
        assert!(is_identifier("foo"));
        assert!(is_identifier("_foo"));
        assert!(is_identifier("foo_bar_42"));
        assert!(is_identifier("X"));
        assert!(!is_identifier("42foo"));
        assert!(!is_identifier("foo-bar"));
        assert!(!is_identifier("héllo"));
        assert!(!is_identifier(" "));
    }

    #[test]
    fn non_ascii_returns_chunk_as_fallback() {
        // Non-ASCII chunks have no regex match in Python; the Python
        // fallback returns the whole chunk. We mirror that.
        assert_eq!(split_identifier("αβγ"), vec!["αβγ"]);
    }

    #[test]
    fn leading_trailing_underscores() {
        // Python: split_identifier("__init__") splits to ["", "", "init", "", ""]
        // (split by "_"), then empty chunks are skipped, so result is ["init"].
        assert_eq!(split_identifier("__init__"), vec!["init"]);
    }

    #[test]
    fn single_letter_runs() {
        assert_eq!(split_identifier("aB"), vec!["a", "B"]);
        assert_eq!(split_identifier("Ab"), vec!["Ab"]);
    }
}
