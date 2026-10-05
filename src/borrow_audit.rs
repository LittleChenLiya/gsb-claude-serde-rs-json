//! Explicit borrow audit for the Deserializer.
//!
//! This module provides a diagnostic tool that records the paths and source
//! ranges of every `visit_borrowed_str`, copied string, and owned read during
//! deserialization. It does not change the behavior or lifetime constraints of
//! deserialization.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

/// The kind of borrow that occurred when a string was visited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    /// The string was borrowed directly from the input slice (no escape
    /// sequences were present).
    ///
    /// This is only possible when deserializing from a `&[u8]` or `&str`.
    Borrowed,
    /// The string was copied into the scratch space because it contained
    /// escape sequences that needed to be unescaped.
    Copied,
    /// The string was read from a `std::io::Read` source and is owned by the
    /// deserializer.
    ///
    /// Reader-based deserializers never borrow from the input because the
    /// `std::io::Read` trait has no non-copying methods.
    Owned,
}

impl fmt::Display for BorrowKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BorrowKind::Borrowed => f.write_str("borrowed"),
            BorrowKind::Copied => f.write_str("copied"),
            BorrowKind::Owned => f.write_str("owned"),
        }
    }
}

/// A segment of a path through a JSON structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// A map key.
    Key(String),
    /// An array index.
    Index(usize),
}

impl fmt::Display for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Key(k) => f.write_str(k),
            PathSegment::Index(i) => write!(f, "[{}]", i),
        }
    }
}

/// A single borrow event recorded during deserialization.
#[derive(Debug, Clone)]
pub struct BorrowEvent {
    /// The path to the string in the JSON structure.
    pub path: String,
    /// The source range (byte offsets) of the string in the input, including
    /// the surrounding quotes.
    pub range: Range<usize>,
    /// The kind of borrow that occurred.
    pub kind: BorrowKind,
    /// Whether the string is a map key (as opposed to a map value or array
    /// element).
    pub is_key: bool,
}

impl fmt::Display for BorrowEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}..{} ({})",
            self.path, self.range.start, self.range.end, self.kind
        )
    }
}

/// An audit of all borrows that occurred during deserialization.
///
/// This is a diagnostic tool that records the paths and source ranges of every
/// `visit_borrowed_str`, copied string, and owned read. It does not change the
/// behavior or lifetime constraints of deserialization.
///
/// # Examples
///
/// ```
/// use serde_json::Deserializer;
/// use serde_json::borrow_audit::BorrowKind;
///
/// let data = r#"{"name": "John", "greeting": "Hello, \"world\"!"}"#;
/// let mut de = Deserializer::from_str(data);
/// let _: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
///
/// let audit = de.borrow_audit();
/// for event in audit.events() {
///     println!("{}", event);
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct BorrowAudit {
    events: Vec<BorrowEvent>,
}

impl BorrowAudit {
    /// Create a new, empty borrow audit.
    pub fn new() -> Self {
        BorrowAudit { events: Vec::new() }
    }

    /// Record a borrow event.
    pub(crate) fn record(
        &mut self,
        path: String,
        start: usize,
        end: usize,
        kind: BorrowKind,
        is_key: bool,
    ) {
        self.events.push(BorrowEvent {
            path,
            range: start..end,
            kind,
            is_key,
        });
    }

    /// Returns a slice of all recorded borrow events.
    pub fn events(&self) -> &[BorrowEvent] {
        &self.events
    }

    /// Returns the number of recorded borrow events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if no borrow events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns an iterator over the recorded borrow events.
    pub fn iter(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events.iter()
    }

    /// Returns an iterator over the borrowed events (strings that were
    /// borrowed directly from the input).
    pub fn borrowed(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events
            .iter()
            .filter(|e| e.kind == BorrowKind::Borrowed)
    }

    /// Returns an iterator over the copied events (strings that were copied
    /// because they contained escape sequences).
    pub fn copied(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events
            .iter()
            .filter(|e| e.kind == BorrowKind::Copied)
    }

    /// Returns an iterator over the owned events (strings that were read from
    /// a `std::io::Read` source and are owned by the deserializer).
    pub fn owned(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events
            .iter()
            .filter(|e| e.kind == BorrowKind::Owned)
    }

    /// Returns an iterator over the key events (map keys).
    pub fn keys(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events.iter().filter(|e| e.is_key)
    }

    /// Returns an iterator over the value events (map values or array
    /// elements).
    pub fn values(&self) -> impl Iterator<Item = &BorrowEvent> {
        self.events.iter().filter(|e| !e.is_key)
    }
}

impl fmt::Display for BorrowAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BorrowAudit ({} events):", self.events.len())?;
        for event in &self.events {
            writeln!(f, "  {}", event)?;
        }
        Ok(())
    }
}
