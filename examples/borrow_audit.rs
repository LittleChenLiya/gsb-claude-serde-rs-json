//! Example of using the borrow audit.
//!
//! This example demonstrates two string cases:
//! 1. A plain string (no escapes) that is borrowed from the input.
//! 2. A string with escapes that is copied.
//!
//! Run with:
//!   cargo borrow-audit

use serde::Deserialize;
use serde_json::borrow_audit::BorrowKind;
use serde_json::Deserializer;

fn main() {
    // Example 1: A plain string (no escapes) that is borrowed from the input.
    let data = r#"{"name": "John", "city": "New York"}"#;
    let mut de = Deserializer::from_str(data);
    let _: serde_json::Value = Deserialize::deserialize(&mut de).unwrap();

    println!("=== Example 1: Plain strings (borrowed) ===");
    println!("Input: {}", data);
    let audit = de.borrow_audit();
    for event in audit.events() {
        println!(
            "  {}: {}..{} ({})",
            event.path, event.range.start, event.range.end, event.kind
        );
    }
    println!();

    // Example 2: A string with escapes that is copied.
    let data = r#"{"greeting": "Hello, \"world\"!", "tab": "a\tb"}"#;
    let mut de = Deserializer::from_str(data);
    let _: serde_json::Value = Deserialize::deserialize(&mut de).unwrap();

    println!("=== Example 2: Strings with escapes (copied) ===");
    println!("Input: {}", data);
    let audit = de.borrow_audit();
    for event in audit.events() {
        println!(
            "  {}: {}..{} ({})",
            event.path, event.range.start, event.range.end, event.kind
        );
    }
    println!();

    // Verify the acceptance criteria:
    // - No escaped borrow fields have original slice source.
    // - Fields with escapes go through copy.
    let audit = de.borrow_audit();
    for event in audit.events() {
        if !event.is_key && (event.path == "greeting" || event.path == "tab") {
            assert_eq!(
                event.kind,
                BorrowKind::Copied,
                "escaped string should be copied, not borrowed"
            );
        }
    }
    println!("All assertions passed!");
}
