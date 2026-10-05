//! Explicit audit of string borrowing performed by the deserializer.
//!
//! This module records, during a single ordinary deserialization pass, which
//! JSON strings were actually delivered to the visitor through
//! `visit_borrowed_str` (a true borrow of the input), which were delivered
//! through `visit_str` after being copied into the scratch buffer (escaped
//! strings), and which were read from an `io::Read` input where no borrowing
//! of the input is possible at all. The audit is based solely on what the
//! parser actually did at each call site, never on guesses derived from the
//! target type being deserialized.
//!
//! The audit does not re-run `Deserialize`, does not wrap or replace the
//! visitor (so the number of visitor calls is unchanged), and does not
//! alter the parse result or any lifetime constraints. If deserialization
//! fails, the audit contains only the events recorded up to the failure
//! point and is therefore potentially incomplete.
//!
//! # Cargo commands to verify locally
//!
//! ```text
//! cargo test --test audit
//! cargo test --doc audit
//! ```
//!
//! # Two string examples
//!
//! A string without escapes in a `&[u8]` input is borrowed from the
//! original slice, and the recorded byte range points back into that
//! slice:
//!
//! ```
//! use serde_json::audit::{InputSource, StringKind, StringRole};
//!
//! let json = br#"{"name": "ferris"}"#;
//! let mut de = serde_json::Deserializer::from_slice(json);
//! de.enable_borrow_audit();
//! let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
//!
//! let audit = de.borrow_audit().unwrap();
//! let events = audit.events();
//! assert_eq!(events.len(), 2);
//!
//! // The map key "name" was borrowed from the original slice.
//! assert_eq!(events[0].path, "$.name");
//! assert_eq!(events[0].role, StringRole::Key);
//! assert_eq!(events[0].kind, StringKind::Borrowed);
//! assert_eq!(events[0].source, InputSource::Slice);
//! assert_eq!(&json[events[0].start..events[0].end], b"\"name\"");
//!
//! // The value "ferris" has no escapes, so it was borrowed too.
//! assert_eq!(events[1].path, "$.name");
//! assert_eq!(events[1].role, StringRole::Value);
//! assert_eq!(events[1].kind, StringKind::Borrowed);
//! assert_eq!(&json[events[1].start..events[1].end], b"\"ferris\"");
//! ```
//!
//! A string containing an escape sequence can never be reported as
//! borrowed; it is copied into the scratch buffer instead:
//!
//! ```
//! use serde_json::audit::{StringKind, StringRole};
//!
//! let json = br#"{"greeting": "hi\nthere"}"#;
//! let mut de = serde_json::Deserializer::from_slice(json);
//! de.enable_borrow_audit();
//! let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
//!
//! let audit = de.borrow_audit().unwrap();
//! let events = audit.events();
//! assert_eq!(events.len(), 2);
//! assert_eq!(events[0].kind, StringKind::Borrowed); // key "greeting"
//! assert_eq!(events[1].path, "$.greeting");
//! assert_eq!(events[1].role, StringRole::Value);
//! assert_eq!(events[1].kind, StringKind::Copied); // escaped, never borrowed
//! ```

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::read::Reference;

/// The kind of input the deserializer is reading from.
///
/// This is recorded by the `Deserializer` constructor (`from_slice`,
/// `from_str`, `from_reader` or the generic `new`), not inferred from the
/// target type being deserialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputSource {
    /// `Deserializer::from_slice`: input is a `&[u8]`; unescaped strings
    /// may be borrowed directly from it.
    Slice,
    /// `Deserializer::from_str`: input is a `&str`; unescaped strings may
    /// be borrowed directly from it.
    Str,
    /// `Deserializer::from_reader`: input is an `io::Read`. Nothing can be
    /// borrowed from the input; every string is copied out of the stream.
    Reader,
    /// `Deserializer::new` with a custom `Read` implementation.
    Custom,
}

impl InputSource {
    fn as_str(self) -> &'static str {
        match self {
            InputSource::Slice => "slice",
            InputSource::Str => "str",
            InputSource::Reader => "reader",
            InputSource::Custom => "custom",
        }
    }
}

/// Whether a recorded string appeared as a map key or as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringRole {
    /// The string was parsed while deserializing a map key.
    Key,
    /// The string was parsed while deserializing a value (including array
    /// elements and top-level values).
    Value,
}

impl StringRole {
    fn as_str(self) -> &'static str {
        match self {
            StringRole::Key => "key",
            StringRole::Value => "value",
        }
    }
}

/// How a string was actually delivered to the visitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringKind {
    /// The visitor received the string through `visit_borrowed_str` (or
    /// `visit_borrowed_bytes`): a genuine borrow of the input. Only
    /// possible for slice/str inputs and only when the string contains no
    /// escape sequences.
    Borrowed,
    /// The visitor received the string through `visit_str` (or
    /// `visit_bytes`) after it was copied into the scratch buffer. This is
    /// always the case for strings containing escape sequences and for
    /// anything read from an `io::Read` input.
    Copied,
}

impl StringKind {
    fn as_str(self) -> &'static str {
        match self {
            StringKind::Borrowed => "borrowed",
            StringKind::Copied => "copied",
        }
    }
}

/// One recorded string read performed by the deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringEvent {
    /// JSON path of the string, e.g. `$.user.name` or `$.tags[2]`. Map
    /// keys are reported at their own path (the same path their value
    /// will later be reported at).
    pub path: String,
    /// Whether the string was a map key or a value.
    pub role: StringRole,
    /// Whether the visitor got a genuine borrow of the input or a copy.
    pub kind: StringKind,
    /// The kind of input the deserializer was reading from.
    pub source: InputSource,
    /// Byte offset of the opening `"` of the string token in the input
    /// (for `Reader` inputs: in the byte stream consumed so far).
    pub start: usize,
    /// Byte offset one past the closing `"` of the string token. For
    /// `Slice`/`Str` sources, `&input[start..end]` is the raw JSON string
    /// token including quotes; for a `Borrowed` event the borrowed `&str`
    /// is exactly `&input[start + 1..end - 1]`.
    pub end: usize,
}

impl StringEvent {
    /// Byte range of the raw JSON string token (including quotes) in the
    /// input.
    pub fn range(&self) -> (usize, usize) {
        (self.start, self.end)
    }
}

#[derive(Debug, Clone)]
enum PathSegment {
    Key(String),
    Index(usize),
}

fn is_simple_key(key: &str) -> bool {
    let mut chars = key.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn push_key_segment(out: &mut String, key: &str) {
    if is_simple_key(key) {
        out.push('.');
        out.push_str(key);
    } else {
        out.push('[');
        // Debug formatting of str produces a quoted, escaped string.
        let _ = write!(out, "{:?}", key);
        out.push(']');
    }
}

fn render_path(segments: &[PathSegment]) -> String {
    let mut out = String::from("$");
    for segment in segments {
        match segment {
            PathSegment::Key(key) => push_key_segment(&mut out, key),
            PathSegment::Index(index) => {
                let _ = write!(out, "[{}]", index);
            }
        }
    }
    out
}

/// A record of every string the deserializer read, collected during a
/// single deserialization pass.
///
/// Obtain one by calling
/// [`Deserializer::enable_borrow_audit`][crate::Deserializer] before
/// deserializing, then inspect it through
/// [`Deserializer::borrow_audit`][crate::Deserializer] or
/// [`Deserializer::take_borrow_audit`][crate::Deserializer].
#[derive(Debug, Clone)]
pub struct BorrowAudit {
    source: InputSource,
    events: Vec<StringEvent>,
    segments: Vec<PathSegment>,
    in_key: bool,
    pending_key: Option<String>,
    pending_kind: Option<StringKind>,
}

impl BorrowAudit {
    pub(crate) fn new(source: InputSource) -> Self {
        BorrowAudit {
            source,
            events: Vec::new(),
            segments: Vec::new(),
            in_key: false,
            pending_key: None,
            pending_kind: None,
        }
    }

    /// The kind of input the deserializer read from.
    pub fn source(&self) -> InputSource {
        self.source
    }

    /// All recorded string events, in the order the strings were parsed.
    pub fn events(&self) -> &[StringEvent] {
        &self.events
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no strings were recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate over only the events where the visitor received a genuine
    /// borrow of the input (`visit_borrowed_str` / `visit_borrowed_bytes`).
    pub fn borrowed(&self) -> impl Iterator<Item = &StringEvent> {
        self.events
            .iter()
            .filter(|event| event.kind == StringKind::Borrowed)
    }

    /// Iterate over only the events where the visitor received a copied
    /// string (`visit_str` / `visit_bytes`): escaped strings and anything
    /// read from a reader.
    pub fn copied(&self) -> impl Iterator<Item = &StringEvent> {
        self.events
            .iter()
            .filter(|event| event.kind == StringKind::Copied)
    }

    /// Render a human-readable report, one line per recorded string.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "input source: {}", self.source.as_str());
        for event in &self.events {
            let _ = writeln!(
                out,
                "{} {} {} [{}..{}] ({})",
                event.kind.as_str(),
                event.role.as_str(),
                event.path,
                event.start,
                event.end,
                event.source.as_str(),
            );
        }
        out
    }

    // -- crate-internal recording hooks ------------------------------------

    fn note(&mut self, kind: StringKind) {
        self.pending_kind = Some(kind);
    }

    fn commit(&mut self, start: usize, end: usize) {
        if let Some(kind) = self.pending_kind.take() {
            let (role, path) = if self.in_key {
                let mut segments = self.segments.clone();
                let key = self.pending_key.as_deref().unwrap_or("?").to_owned();
                segments.push(PathSegment::Key(key));
                (StringRole::Key, render_path(&segments))
            } else {
                (StringRole::Value, render_path(&self.segments))
            };
            self.events.push(StringEvent {
                path,
                role,
                kind,
                source: self.source,
                start,
                end,
            });
        }
    }

    fn begin_key(&mut self) {
        self.in_key = true;
        self.pending_key = None;
    }

    fn end_key(&mut self) {
        self.in_key = false;
        let key = self.pending_key.take().unwrap_or_else(|| String::from("?"));
        self.segments.push(PathSegment::Key(key));
    }

    fn end_value(&mut self) {
        self.segments.pop();
    }

    fn begin_index(&mut self, index: usize) {
        self.segments.push(PathSegment::Index(index));
    }

    fn end_container_child(&mut self) {
        self.segments.pop();
    }
}

// -- free helpers so call sites only borrow the `audit` field --------------

pub(crate) fn note_str(slot: &mut Option<BorrowAudit>, reference: &Reference<'_, '_, str>) {
    if let Some(audit) = slot {
        let kind = match reference {
            Reference::Borrowed(_) => StringKind::Borrowed,
            Reference::Copied(_) => StringKind::Copied,
        };
        audit.note(kind);
        if audit.in_key {
            let text: &str = match reference {
                Reference::Borrowed(s) => s,
                Reference::Copied(s) => s,
            };
            audit.pending_key = Some(text.to_owned());
        }
    }
}

pub(crate) fn note_bytes(slot: &mut Option<BorrowAudit>, reference: &Reference<'_, '_, [u8]>) {
    if let Some(audit) = slot {
        let kind = match reference {
            Reference::Borrowed(_) => StringKind::Borrowed,
            Reference::Copied(_) => StringKind::Copied,
        };
        audit.note(kind);
    }
}

pub(crate) fn commit(slot: &mut Option<BorrowAudit>, start: usize, end: usize) {
    if let Some(audit) = slot {
        audit.commit(start, end);
    }
}

pub(crate) fn begin_key(slot: &mut Option<BorrowAudit>) {
    if let Some(audit) = slot {
        audit.begin_key();
    }
}

pub(crate) fn end_key(slot: &mut Option<BorrowAudit>) {
    if let Some(audit) = slot {
        audit.end_key();
    }
}

pub(crate) fn end_value(slot: &mut Option<BorrowAudit>) {
    if let Some(audit) = slot {
        audit.end_value();
    }
}

pub(crate) fn begin_index(slot: &mut Option<BorrowAudit>, index: usize) {
    if let Some(audit) = slot {
        audit.begin_index(index);
    }
}

pub(crate) fn end_container_child(slot: &mut Option<BorrowAudit>) {
    if let Some(audit) = slot {
        audit.end_container_child();
    }
}
