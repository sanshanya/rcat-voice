use unicode_segmentation::UnicodeSegmentation;

/// Language detection based on Unicode ranges
pub fn detect_language(text: &str) -> Language {
    let has_chinese = text.chars().any(|c| {
        ('\u{4E00}'..='\u{9FFF}').contains(&c) // CJK Unified Ideographs
    });

    if has_chinese {
        Language::Chinese
    } else {
        Language::English
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Language {
    Chinese,
    English,
}

/// Split text into sentences using Unicode-aware rules
pub fn split_sentences(text: &str) -> Vec<String> {
    let lang = detect_language(text);

    let mut sentences = Vec::new();
    let mut current = String::new();

    let graphemes = text.graphemes(true).collect::<Vec<&str>>();

    for (_i, grapheme) in graphemes.iter().enumerate() {
        current.push_str(grapheme);

        let is_sentence_end = match lang {
            Language::Chinese => {
                // Chinese sentence endings - simpler logic
                matches!(*grapheme, "。" | "！" | "？")
            }
            Language::English => {
                // English sentence endings with basic abbreviation handling
                if matches!(*grapheme, "." | "!" | "?") {
                    // Check for common abbreviations
                    !current.trim_end().ends_with("Dr.")
                        && !current.trim_end().ends_with("Mr.")
                        && !current.trim_end().ends_with("Mrs.")
                        && !current.trim_end().ends_with("Ms.")
                        && !current.trim_end().ends_with("Prof.")
                        && !current.trim_end().ends_with("Sr.")
                        && !current.trim_end().ends_with("Jr.")
                } else {
                    false
                }
            }
        };

        if is_sentence_end {
            sentences.push(current.trim().to_string());
            current.clear();
        }
    }

    // Add remaining text
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("Hello world"), Language::English);
        assert_eq!(detect_language("你好世界"), Language::Chinese);
        assert_eq!(detect_language("Hello 你好"), Language::Chinese); // Mixed, prefers Chinese
    }

    #[test]
    fn test_split_english() {
        let text = "Hello world. How are you? I am fine!";
        let result = split_sentences(text);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "Hello world.");
        assert_eq!(result[1], "How are you?");
        assert_eq!(result[2], "I am fine!");
    }

    #[test]
    fn test_split_chinese() {
        let text = "你好。这是测试。结束了！";
        let result = split_sentences(text);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "你好。");
        assert_eq!(result[1], "这是测试。");
        assert_eq!(result[2], "结束了！");
    }

    #[test]
    fn test_abbreviations() {
        let text = "Dr. Smith works here. He is great.";
        let result = split_sentences(text);
        assert!(result.len() <= 2); // Should not split at "Dr."
    }
}
