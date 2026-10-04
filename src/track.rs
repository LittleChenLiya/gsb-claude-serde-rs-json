//! Track JSON Pointer paths and byte ranges while serializing.
//!
//! This module provides [`PointerSerializer`], an explicit adapter around the
//! ordinary serde_json [`Serializer`] that records, during a *single* JSON
//! serialization pass, the [RFC 6901] JSON Pointer of every value in the
//! output together with the byte range it occupies in the produced bytes.
//!
//! The index is built straight from the byte stream the serializer actually
//! emits (a counting [`io::Write`] wrapper plus a tracking [`Formatter`]); the
//! input is never serialized twice and the output is never re-parsed. Because
//! recording is driven by the real output:
//!
//! * ranges point at the actual bytes, including the surrounding quotes of
//!   escaped strings and the full span of containers (with pretty indent);
//! * array indices are assigned in the order elements are actually emitted,
//!   never inferred from a Rust field layout;
//! * fields renamed by a hand-written `Serialize` impl or by
//!   `serde(rename)` are indexed under the names that were written;
//! * flattened and skipped fields (`skip`, `skip_serializing_if`) and fields
//!   dropped by custom `serialize` impls are simply absent from the index.
//!
//! The default serde_json serializer is unchanged; nothing here is active
//! unless [`PointerSerializer`] or one of the `*_tracked` helpers is used.
//!
//! # Range semantics
//!
//! Each [`PointerEntry`] stores a half-open range `start..end` into the
//! serialized output, so `output[start..end]` is the complete JSON encoding of
//! the referenced value:
//!
//! | Referenced value              | Covered bytes |
//! |-------------------------------|---------------|
//! | null / bool / number          | the literal token (`null`, `true`, `-2.5`) |
//! | string                        | opening quote through closing quote, all escapes included |
//! | array                         | `[` through `]`, including elements, commas and internal indentation |
//! | object                        | `{` through `}`, including keys, colons, commas and indentation |
//!
//! The range never includes the separator or pretty indentation that belongs
//! to the parent container, so every slice is a standalone JSON document.
//! Object keys themselves are not indexed (only their values); the key text
//! is taken from the emitted bytes and encoded per RFC 6901 (`~` to `~0`,
//! `/` to `~1`).
//!
//! # Examples
//!
//! ```
//! use serde_json::track::to_vec_tracked;
//!
//! // {"a/b":["猫",2]}
//! let value = serde_json::json!({"a/b": ["猫", 2]});
//! let (bytes, index) = to_vec_tracked(&value).unwrap();
//!
//! assert_eq!(index.slice("/a~1b/0", &bytes).unwrap(), b"\"\xE7\x8C\xAB\"");
//! assert_eq!(index.slice("/a~1b", &bytes).unwrap(), b"[\"\xE7\x8C\xAB\",2]");
//! assert_eq!(index.slice("", &bytes).unwrap(), &bytes[..]);
//! assert!(index.complete());
//! ```
//!
//! # Failure behavior
//!
//! If serialization fails (a serde error returned by the value's `Serialize`
//! impl, or an I/O error from the underlying writer, including a streaming
//! writer that fails mid-output), the serializer returns the original error
//! and [`PointerIndex::complete`] is `false`. Entries for values fully
//! written before the failure remain available as a prefix fragment; a value
//! whose bytes were interrupted is never recorded. The adapter never
//! presents a partial index as complete.
//!
//! [RFC 6901]: https://www.rfc-editor.org/rfc/rfc6901

use crate::io;
use crate::ser::{CharEscape, CompactFormatter, Formatter, PrettyFormatter, Serializer};
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::char;
use core::str;
use serde::ser::Serialize;

/// Mutable recording state shared by the counting writer and the formatter.
struct TrackState {
    /// Bytes successfully written so far.
    offset: usize,
    /// Set the first time any error aborts serialization.
    failed: bool,
    /// Containers currently open, outermost first.
    frames: Vec<Frame>,
    /// Values fully recorded so far.
    entries: Vec<PointerEntry>,
    /// State of the string currently being written, if any.
    pending: Option<PendingString>,
    /// While `Some`, every successfully written byte belongs to the object
    /// key token currently being emitted and is mirrored here. Set by
    /// `begin_object_key`, consumed by `end_object_key`.
    key_capture: Option<Vec<u8>>,
}

struct PendingString {
    start: usize,
    pointer: String,
    /// The string is an object key: accumulate decoded bytes instead of
    /// producing an entry.
    is_key: bool,
}

struct Frame {
    kind: FrameKind,
    /// RFC 6901 pointer of the container itself (`""` at the root).
    prefix: String,
    /// Offset of the opening bracket/brace.
    start: usize,
    /// Next array index to hand out (arrays only).
    array_index: u64,
    phase: Phase,
    /// Decoded text of the most recently written key (objects only).
    key_buffer: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameKind {
    Array,
    Object,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Next value inside this frame is an array element.
    ArrayValue,
    /// Next string inside this frame is an object key.
    Key,
    /// Key has been written; next value is the corresponding object value.
    Value,
}

/// A JSON Pointer and the byte range its value occupies in the output.
///
/// See the [module documentation](self) for exact range semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointerEntry {
    /// RFC 6901 JSON Pointer (`""` for the root value, e.g. `"/a~1b/0"`).
    pub pointer: String,
    /// Inclusive start byte offset in the serialized output.
    pub start: usize,
    /// Exclusive end byte offset in the serialized output.
    pub end: usize,
}

impl PointerEntry {
    /// Slice this value's complete JSON encoding out of `output`.
    ///
    /// Returns `None` when the range does not lie within `output` (for
    /// example a fragment index paired with a truncated buffer).
    pub fn get<'a>(&self, output: &'a [u8]) -> Option<&'a [u8]> {
        output.get(self.start..self.end)
    }
}

/// JSON Pointer / byte-range index produced by a [`PointerSerializer`].
#[derive(Clone, Debug)]
pub struct PointerIndex {
    entries: Vec<PointerEntry>,
    complete: bool,
}

impl PointerIndex {
    /// Whether the whole value was serialized without error.
    ///
    /// A `false` value marks a prefix fragment: it contains only entries
    /// whose bytes were complete when serialization failed, and must not be
    /// interpreted as a full index.
    pub fn complete(&self) -> bool {
        self.complete
    }

    /// Recorded entries in completion order: nested values precede their
    /// container, and the root value is last.
    pub fn entries(&self) -> &[PointerEntry] {
        &self.entries
    }

    /// Number of recorded values.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no value was recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Byte range of `pointer`, e.g. `range("/a~1b/0")`.
    pub fn range(&self, pointer: &str) -> Option<core::ops::Range<usize>> {
        self.entries
            .iter()
            .find(|entry| entry.pointer == pointer)
            .map(|entry| entry.start..entry.end)
    }

    /// Slice `pointer`'s complete JSON encoding out of `output`.
    pub fn slice<'a>(&self, pointer: &str, output: &'a [u8]) -> Option<&'a [u8]> {
        self.range(pointer).and_then(|range| output.get(range))
    }
}

/// `io::Write` decorator that counts bytes written through it.
struct CountingWriter<W> {
    inner: W,
    state: Rc<RefCell<TrackState>>,
}

impl<W> io::Write for CountingWriter<W>
where
    W: io::Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = match self.inner.write(buf) {
            Ok(n) => n,
            Err(err) => return Err(err),
        };
        let mut st = self.state.borrow_mut();
        st.offset += n;
        if let Some(capture) = st.key_capture.as_mut() {
            capture.extend_from_slice(&buf[..n]);
        }
        Ok(n)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        if let Err(err) = self.inner.write_all(buf) {
            return Err(err);
        }
        let mut st = self.state.borrow_mut();
        st.offset += buf.len();
        if let Some(capture) = st.key_capture.as_mut() {
            capture.extend_from_slice(buf);
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// [`Formatter`] decorator that records ranges while delegating all actual
/// formatting to an inner compact/pretty formatter.
struct TrackFormatter<F> {
    inner: F,
    state: Rc<RefCell<TrackState>>,
}

impl<F> TrackFormatter<F> {
    /// Begin a non-string scalar: reserve the pending slot and return the
    /// start offset.
    fn begin_scalar(&self) {
        let mut st = self.state.borrow_mut();
        let start = st.offset;
        let (pointer, is_key) = next_value_pointer(&mut st);
        // All map keys are emitted inside begin_string/end_string by
        // serde_json's MapKeySerializer; a scalar hook with the frame in
        // `Key` phase is therefore unreachable in practice. Skip it if it
        // ever occurs rather than recording a bogus entry.
        if !is_key {
            st.pending = Some(PendingString {
                start,
                pointer,
                is_key: false,
            });
        }
    }
}

/// Compute the pointer of the next value, consuming one array index or
/// preparing the object-key slot as appropriate. Returns the container's
/// own pointer with `is_key = true` when the next emitted value is a key.
fn next_value_pointer(st: &mut TrackState) -> (String, bool) {
    let Some(frame) = st.frames.last_mut() else {
        return (String::new(), false);
    };
    match frame.phase {
        Phase::ArrayValue => {
            let mut pointer = frame.prefix.clone();
            pointer.push('/');
            push_u64(&mut pointer, frame.array_index);
            frame.array_index += 1;
            (pointer, false)
        }
        Phase::Value => {
            let mut pointer = frame.prefix.clone();
            pointer.push('/');
            pointer.push_str(&escape_token(&frame.key_buffer));
            (pointer, false)
        }
        Phase::Key => {
            frame.key_buffer.clear();
            (frame.prefix.clone(), true)
        }
    }
}

/// RFC 6901 reference-token escaping: `~` -> `~0`, `/` -> `~1`.
fn escape_token(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for ch in key.chars() {
        match ch {
            '~' => out.push_str("~0"),
            '/' => out.push_str("~1"),
            _ => out.push(ch),
        }
    }
    out
}

fn push_u64(out: &mut String, value: u64) {
    let mut buffer = itoa::Buffer::new();
    out.push_str(buffer.format(value));
}

/// Decode the text of a JSON key token as it appeared in the output.
///
/// `token` is a quoted JSON string (serde_json's `MapKeySerializer` always
/// emits even non-string keys through `begin_string`/`end_string`). The slice
/// is taken from the real output bytes, so escaping is reversed exactly as
/// emitted; malformed tokens fall back to lossy raw text.
fn decode_json_key(token: &[u8]) -> String {
    let bs = char::from_u32(92).unwrap();
    let inner = match token
        .strip_prefix(b"\"")
        .and_then(|t| t.strip_suffix(b"\""))
    {
        Some(inner) => inner,
        None => return String::from_utf8_lossy(token).into_owned(),
    };
    let text = match str::from_utf8(inner) {
        Ok(text) => text,
        Err(_) => return String::from_utf8_lossy(inner).into_owned(),
    };
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != bs {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some(ch) if ch == bs => out.push(bs),
            Some('/') => out.push('/'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000C}'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if hex.len() == 4 {
                    if let Ok(value) = u32::from_str_radix(&hex, 16) {
                        if (0xD800..=0xDBFF).contains(&value) {
                            let mut low = String::new();
                            if chars.next() == Some(bs) && chars.next() == Some('u') {
                                low.extend(chars.by_ref().take(4));
                            }
                            if let Ok(lo) = u32::from_str_radix(&low, 16) {
                                if let Some(c) = char::from_u32(
                                    0x10000 + ((value - 0xD800) << 10) + (lo - 0xDC00),
                                ) {
                                    out.push(c);
                                    continue;
                                }
                            }
                        }
                        if let Some(c) = char::from_u32(value) {
                            out.push(c);
                            continue;
                        }
                    }
                }
                out.push(bs);
                out.push('u');
                out.push_str(&hex);
            }
            Some(other) => {
                out.push(bs);
                out.push(other);
            }
            None => out.push(bs),
        }
    }
    out
}

impl<F> TrackFormatter<F>
where
    F: Formatter,
{
    /// Run `body` (which writes one non-string scalar through the inner
    /// formatter), recording the scalar's range on success.
    fn scalar<W>(
        &mut self,
        writer: &mut W,
        body: impl FnOnce(&mut F, &mut W) -> io::Result<()>,
    ) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.begin_scalar();
        let result = body(&mut self.inner, writer);
        self.finish_scalar(result)
    }

    fn finish_scalar(&self, result: io::Result<()>) -> io::Result<()> {
        let mut st = self.state.borrow_mut();
        match result {
            Ok(()) => {
                if let Some(pending) = st.pending.take() {
                    let end = st.offset;
                    st.entries.push(PointerEntry {
                        pointer: pending.pointer,
                        start: pending.start,
                        end,
                    });
                }
                Ok(())
            }
            Err(err) => {
                st.pending = None;
                st.failed = true;
                Err(err)
            }
        }
    }
}

macro_rules! scalar_fn {
    ($method:ident, $arg_ty:ty) => {
        #[inline]
        fn $method<W>(&mut self, writer: &mut W, value: $arg_ty) -> io::Result<()>
        where
            W: ?Sized + io::Write,
        {
            self.scalar(writer, |f, w| f.$method(w, value))
        }
    };
}

impl<F> Formatter for TrackFormatter<F>
where
    F: Formatter,
{
    #[inline]
    fn write_null<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.scalar(writer, |f, w| f.write_null(w))
    }

    scalar_fn!(write_bool, bool);
    scalar_fn!(write_i8, i8);
    scalar_fn!(write_i16, i16);
    scalar_fn!(write_i32, i32);
    scalar_fn!(write_i64, i64);
    scalar_fn!(write_i128, i128);
    scalar_fn!(write_u8, u8);
    scalar_fn!(write_u16, u16);
    scalar_fn!(write_u32, u32);
    scalar_fn!(write_u64, u64);
    scalar_fn!(write_u128, u128);
    scalar_fn!(write_f32, f32);
    scalar_fn!(write_f64, f64);

    #[inline]
    fn write_number_str<W>(&mut self, writer: &mut W, value: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.scalar(writer, |f, w| f.write_number_str(w, value))
    }

    #[inline]
    fn write_raw_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.scalar(writer, |f, w| f.write_raw_fragment(w, fragment))
    }

    #[inline]
    fn begin_string<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        {
            let mut st = self.state.borrow_mut();
            let start = st.offset;
            let (pointer, is_key) = next_value_pointer(&mut st);
            st.pending = Some(PendingString {
                start,
                pointer,
                is_key,
            });
        }
        let result = self.inner.begin_string(writer);
        if result.is_err() {
            let mut st = self.state.borrow_mut();
            st.pending = None;
            st.key_capture = None;
            st.failed = true;
        }
        result
    }

    #[inline]
    fn end_string<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let result = self.inner.end_string(writer);
        let mut st = self.state.borrow_mut();
        match result {
            Ok(()) => {
                if let Some(pending) = st.pending.take() {
                    if !pending.is_key {
                        let end = st.offset;
                        st.entries.push(PointerEntry {
                            pointer: pending.pointer,
                            start: pending.start,
                            end,
                        });
                    }
                }
                Ok(())
            }
            Err(err) => {
                st.pending = None;
                st.failed = true;
                Err(err)
            }
        }
    }

    #[inline]
    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let result = self.inner.write_string_fragment(writer, fragment);
        if result.is_err() {
            self.state.borrow_mut().failed = true;
        }
        result
    }

    #[inline]
    fn write_char_escape<W>(&mut self, writer: &mut W, char_escape: CharEscape) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let result = self.inner.write_char_escape(writer, char_escape);
        if result.is_err() {
            self.state.borrow_mut().failed = true;
        }
        result
    }

    fn write_byte_array<W>(&mut self, writer: &mut W, value: &[u8]) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        // Reproduce the default Formatter implementation so every callback
        // (brackets, separators, integers) goes through our tracked hooks
        // instead of bypassing them.
        tri!(self.begin_array(writer));
        let mut first = true;
        for &byte in value {
            tri!(self.begin_array_value(writer, first));
            tri!(self.write_u8(writer, byte));
            tri!(self.end_array_value(writer));
            first = false;
        }
        self.end_array(writer)
    }

    #[inline]
    fn begin_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        {
            let mut st = self.state.borrow_mut();
            let start = st.offset;
            let (pointer, is_key) = next_value_pointer(&mut st);
            debug_assert!(!is_key, "an array cannot be an object key");
            st.frames.push(Frame {
                kind: FrameKind::Array,
                prefix: pointer,
                start,
                array_index: 0,
                phase: Phase::ArrayValue,
                key_buffer: String::new(),
            });
        }
        let result = self.inner.begin_array(writer);
        if result.is_err() {
            let mut st = self.state.borrow_mut();
            st.frames.pop();
            st.failed = true;
        }
        result
    }

    #[inline]
    fn end_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let result = self.inner.end_array(writer);
        let mut st = self.state.borrow_mut();
        match result {
            Ok(()) => {
                if let Some(frame) = st.frames.pop() {
                    debug_assert_eq!(frame.kind, FrameKind::Array);
                    let end = st.offset;
                    st.entries.push(PointerEntry {
                        pointer: frame.prefix,
                        start: frame.start,
                        end,
                    });
                }
                Ok(())
            }
            Err(err) => {
                st.failed = true;
                Err(err)
            }
        }
    }

    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.inner.begin_array_value(writer, first)
    }

    #[inline]
    fn end_array_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        Ok(())
    }

    #[inline]
    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        {
            let mut st = self.state.borrow_mut();
            let start = st.offset;
            let (pointer, is_key) = next_value_pointer(&mut st);
            debug_assert!(!is_key, "an object cannot be an object key");
            st.frames.push(Frame {
                kind: FrameKind::Object,
                prefix: pointer,
                start,
                array_index: 0,
                phase: Phase::Key,
                key_buffer: String::new(),
            });
        }
        let result = self.inner.begin_object(writer);
        if result.is_err() {
            let mut st = self.state.borrow_mut();
            st.frames.pop();
            st.failed = true;
        }
        result
    }

    #[inline]
    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let result = self.inner.end_object(writer);
        let mut st = self.state.borrow_mut();
        match result {
            Ok(()) => {
                if let Some(frame) = st.frames.pop() {
                    debug_assert_eq!(frame.kind, FrameKind::Object);
                    let end = st.offset;
                    st.entries.push(PointerEntry {
                        pointer: frame.prefix,
                        start: frame.start,
                        end,
                    });
                }
                Ok(())
            }
            Err(err) => {
                st.failed = true;
                Err(err)
            }
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        {
            let mut st = self.state.borrow_mut();
            if let Some(frame) = st.frames.last_mut() {
                frame.phase = Phase::Key;
                frame.key_buffer.clear();
            }
        }
        let result = self.inner.begin_object_key(writer, first);
        if result.is_ok() {
            // The separator/newline/indent are now written; every byte
            // emitted before `end_object_key` is the key token itself.
            self.state.borrow_mut().key_capture = Some(Vec::new());
        } else {
            self.state.borrow_mut().failed = true;
        }
        result
    }

    #[inline]
    fn end_object_key<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        // The key token has finished; stop mirroring *before* the inner call
        // emits the (optional) post-key separator.
        let token = self.state.borrow_mut().key_capture.take();
        let result = self.inner.end_object_key(writer);
        let mut st = self.state.borrow_mut();
        if result.is_ok() {
            let key = token.map_or_else(String::new, |bytes| decode_json_key(&bytes));
            if let Some(frame) = st.frames.last_mut() {
                frame.key_buffer = key;
                frame.phase = Phase::Value;
            }
        } else {
            st.failed = true;
        }
        result
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.inner.begin_object_value(writer)
    }

    #[inline]
    fn end_object_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        Ok(())
    }
}

fn new_state() -> Rc<RefCell<TrackState>> {
    Rc::new(RefCell::new(TrackState {
        offset: 0,
        failed: false,
        frames: Vec::new(),
        entries: Vec::new(),
        pending: None,
        key_capture: None,
    }))
}

fn snapshot(state: &Rc<RefCell<TrackState>>) -> PointerIndex {
    let st = state.borrow();
    PointerIndex {
        entries: st.entries.clone(),
        complete: !st.failed && st.frames.is_empty() && st.pending.is_none(),
    }
}

/// Serialize JSON while recording each value's JSON Pointer and byte range.
///
/// This is an explicit adapter: it wraps a normal serde_json
/// [`Serializer`] and does not change the serializer's output or behavior.
/// Construct it over any [`io::Write`] target (a `Vec<u8>`, a file, a
/// socket...), call [`serialize`](PointerSerializer::serialize), then call
/// [`index`](PointerSerializer::index) to obtain the [`PointerIndex`].
///
/// Compact output is used; use [`pretty`](PointerSerializer::pretty) or
/// [`with_formatter`](PointerSerializer::with_formatter) for pretty printing
/// or a custom [`Formatter`].
pub struct PointerSerializer<W, F = CompactFormatter> {
    ser: Serializer<CountingWriter<W>, TrackFormatter<F>>,
    state: Rc<RefCell<TrackState>>,
}

impl<W> PointerSerializer<W, CompactFormatter>
where
    W: io::Write,
{
    /// Create a tracking serializer writing compact JSON to `writer`.
    pub fn new(writer: W) -> Self {
        Self::with_formatter(writer, CompactFormatter)
    }
}

impl<'a, W> PointerSerializer<W, PrettyFormatter<'a>>
where
    W: io::Write,
{
    /// Create a tracking serializer writing pretty-printed JSON to `writer`.
    pub fn pretty(writer: W) -> Self {
        Self::with_formatter(writer, PrettyFormatter::new())
    }
}

impl<W, F> PointerSerializer<W, F>
where
    W: io::Write,
    F: Formatter,
{
    /// Create a tracking serializer with a custom base [`Formatter`].
    pub fn with_formatter(writer: W, formatter: F) -> Self {
        let state = new_state();
        let counting = CountingWriter {
            inner: writer,
            state: Rc::clone(&state),
        };
        let track = TrackFormatter {
            inner: formatter,
            state: Rc::clone(&state),
        };
        PointerSerializer {
            ser: Serializer::with_formatter(counting, track),
            state,
        }
    }

    /// Serialize `value`, returning the serde error on failure.
    ///
    /// On failure the index is marked incomplete (see
    /// [`PointerIndex::complete`]); values fully written before the failure
    /// are still available through [`index`](Self::index).
    pub fn serialize<T>(&mut self, value: &T) -> crate::Result<()>
    where
        T: ?Sized + Serialize,
    {
        match value.serialize(&mut self.ser) {
            Ok(()) => Ok(()),
            Err(err) => {
                self.state.borrow_mut().failed = true;
                Err(err)
            }
        }
    }

    /// Snapshot of the index recorded so far.
    ///
    /// After a successful [`serialize`](Self::serialize) the returned index
    /// is [`complete`](PointerIndex::complete); after a failure it is a
    /// prefix fragment.
    pub fn index(&self) -> PointerIndex {
        snapshot(&self.state)
    }

    /// Consume the adapter and return the wrapped writer.
    pub fn into_inner(self) -> W {
        self.ser.into_inner().inner
    }
}

/// Error together with the incomplete index recorded before failure.
///
/// `index.complete()` is always `false`. The entries it holds cover values
/// fully written before the failure (a prefix fragment of the output);
/// values interrupted mid-write are absent.
#[derive(Debug)]
pub struct TrackedError {
    /// The serialization error (a serde error or the underlying I/O error).
    pub error: crate::Error,
    /// Prefix-fragment index; never reported as complete.
    pub index: PointerIndex,
}

impl TrackedError {
    fn new(error: crate::Error, index: PointerIndex) -> Self {
        debug_assert!(!index.complete());
        TrackedError { error, index }
    }
}

/// Serialize `value` to compact JSON bytes, tracking pointer ranges.
pub fn to_vec_tracked<T>(value: &T) -> Result<(Vec<u8>, PointerIndex), TrackedError>
where
    T: ?Sized + Serialize,
{
    let mut ser = PointerSerializer::new(Vec::new());
    match ser.serialize(value) {
        Ok(()) => {
            let index = ser.index();
            Ok((ser.into_inner(), index))
        }
        Err(error) => Err(TrackedError::new(error, ser.index())),
    }
}

/// Pretty-printed variant of [`to_vec_tracked`].
pub fn to_vec_pretty_tracked<T>(value: &T) -> Result<(Vec<u8>, PointerIndex), TrackedError>
where
    T: ?Sized + Serialize,
{
    let mut ser = PointerSerializer::pretty(Vec::new());
    match ser.serialize(value) {
        Ok(()) => {
            let index = ser.index();
            Ok((ser.into_inner(), index))
        }
        Err(error) => Err(TrackedError::new(error, ser.index())),
    }
}

/// Serialize `value` to a compact JSON [`String`], tracking pointer ranges.
pub fn to_string_tracked<T>(value: &T) -> Result<(String, PointerIndex), TrackedError>
where
    T: ?Sized + Serialize,
{
    let (bytes, index) = match to_vec_tracked(value) {
        Ok(ok) => ok,
        Err(err) => return Err(err),
    };
    let string = match String::from_utf8(bytes) {
        Ok(string) => string,
        Err(_) => {
            return Err(TrackedError::new(
                crate::Error::io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "JSON was not valid UTF-8",
                )),
                PointerIndex {
                    entries: Vec::new(),
                    complete: false,
                },
            ));
        }
    };
    Ok((string, index))
}

/// Pretty-printed variant of [`to_string_tracked`].
pub fn to_string_pretty_tracked<T>(value: &T) -> Result<(String, PointerIndex), TrackedError>
where
    T: ?Sized + Serialize,
{
    let (bytes, index) = match to_vec_pretty_tracked(value) {
        Ok(ok) => ok,
        Err(err) => return Err(err),
    };
    let string = match String::from_utf8(bytes) {
        Ok(string) => string,
        Err(_) => {
            return Err(TrackedError::new(
                crate::Error::io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "JSON was not valid UTF-8",
                )),
                PointerIndex {
                    entries: Vec::new(),
                    complete: false,
                },
            ));
        }
    };
    Ok((string, index))
}

/// Serialize `value` to `writer`, tracking ranges as bytes stream out.
///
/// `writer` is borrowed, so on failure the caller keeps it and can inspect
/// the bytes already flushed into it. A failure returns the original error
/// together with a fragment index whose
/// [`complete`](PointerIndex::complete) is `false`.
pub fn to_writer_tracked<W, T>(writer: &mut W, value: &T) -> Result<PointerIndex, TrackedError>
where
    W: ?Sized + io::Write,
    T: ?Sized + Serialize,
{
    let mut ser = PointerSerializer::new(&mut *writer);
    match ser.serialize(value) {
        Ok(()) => Ok(ser.index()),
        Err(error) => Err(TrackedError::new(error, ser.index())),
    }
}

/// Pretty-printed variant of [`to_writer_tracked`].
pub fn to_writer_pretty_tracked<W, T>(
    writer: &mut W,
    value: &T,
) -> Result<PointerIndex, TrackedError>
where
    W: ?Sized + io::Write,
    T: ?Sized + Serialize,
{
    let mut ser = PointerSerializer::pretty(&mut *writer);
    match ser.serialize(value) {
        Ok(()) => Ok(ser.index()),
        Err(error) => Err(TrackedError::new(error, ser.index())),
    }
}

impl core::fmt::Display for TrackedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for TrackedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
