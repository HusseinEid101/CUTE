//! Private-Use-Area codepoint ranges. Mirrors `pua.py`.

/// BMP PUA: U+E000 .. U+F8FF inclusive.
pub const PUA_BMP_START: u32 = 0xE000;
pub const PUA_BMP_END: u32 = 0xF8FF;

/// Plane 15 PUA: U+F0000 .. U+FFFFD inclusive (FFFFE/FFFFF are non-characters).
pub const PUA_P15_START: u32 = 0xF_0000;
pub const PUA_P15_END: u32 = 0xF_FFFD;

/// Plane 16 PUA: U+100000 .. U+10FFFD inclusive.
pub const PUA_P16_START: u32 = 0x10_0000;
pub const PUA_P16_END: u32 = 0x10_FFFD;

/// True iff `ch` is a single character in any of the 3 PUA ranges.
#[inline]
pub fn is_pua_char(ch: char) -> bool {
    let cp = ch as u32;
    (PUA_BMP_START..=PUA_BMP_END).contains(&cp)
        || (PUA_P15_START..=PUA_P15_END).contains(&cp)
        || (PUA_P16_START..=PUA_P16_END).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_pua_recognized() {
        assert!(is_pua_char('\u{E000}'));
        assert!(is_pua_char('\u{F8FF}'));
        assert!(!is_pua_char('\u{F900}'));
    }

    #[test]
    fn supplementary_pua_recognized() {
        assert!(is_pua_char('\u{F0000}'));
        assert!(is_pua_char('\u{10FFFD}'));
        assert!(!is_pua_char('\u{10FFFE}'));
        assert!(!is_pua_char('\u{10FFFF}'));
    }

    #[test]
    fn ascii_not_pua() {
        for c in ['a', 'A', '0', ' ', '!', '\n'] {
            assert!(!is_pua_char(c), "{c:?} should not be PUA");
        }
    }
}
