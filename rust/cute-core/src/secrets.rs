//! Corpus secret-pattern scanning. Mirrors `corpus.py:40` `has_secret`.
//!
//! The Python implementation runs each of 10 regexes sequentially; we
//! collapse them into a single `regex::RegexSet`, which evaluates all
//! patterns simultaneously with a single pass over the input.

use once_cell::sync::Lazy;
use regex::RegexSet;

/// Names parallel to `SECRET_PATTERNS` in Python — kept in the same order
/// so that the index returned by `RegexSet::matches` corresponds to the
/// Python pattern name.
pub const SECRET_NAMES: &[&str] = &[
    "aws_access_key",
    "openai_api_key",
    "anthropic_api_key",
    "github_pat",
    "github_oauth",
    "github_app",
    "google_api",
    "slack_token",
    "private_key_pem",
    "jwt",
];

const SECRET_PATTERNS: &[&str] = &[
    r"AKIA[0-9A-Z]{16}",
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
    r"sk-ant-[A-Za-z0-9_\-]{50,}",
    r"ghp_[A-Za-z0-9]{36}",
    r"gho_[A-Za-z0-9]{36}",
    r"(?:ghu|ghs)_[A-Za-z0-9]{36}",
    r"AIza[0-9A-Za-z_\-]{35}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
];

static SECRET_SET: Lazy<RegexSet> =
    Lazy::new(|| RegexSet::new(SECRET_PATTERNS).expect("SECRET_PATTERNS valid"));

/// Return the name of the first matching secret pattern, or `None`.
///
/// Note: Python returns the FIRST match in pattern order. `RegexSet`
/// reports all matches; we pick the lowest index to match Python.
pub fn has_secret(text: &str) -> Option<&'static str> {
    let m = SECRET_SET.matches(text);
    m.into_iter().next().map(|i| SECRET_NAMES[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_aws_key() {
        let s = "use AKIAIOSFODNN7EXAMPLE today";
        assert_eq!(has_secret(s), Some("aws_access_key"));
    }

    #[test]
    fn detects_github_pat() {
        let s = "token=ghp_abcdefghijklmnopqrstuvwxyzABCDEF1234";
        assert_eq!(has_secret(s), Some("github_pat"));
    }

    #[test]
    fn no_match_on_clean_text() {
        assert_eq!(has_secret("hello world"), None);
    }

    #[test]
    fn detects_pem_marker() {
        let s = "-----BEGIN RSA PRIVATE KEY-----";
        assert_eq!(has_secret(s), Some("private_key_pem"));
    }
}
