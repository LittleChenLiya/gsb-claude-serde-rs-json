//! Serializer adapter that records the byte range of every JSON value it
//! writes, in the same pass that produces the output.
//!
//! The [`Locator`] wraps an ordinary [`Serializer`] and feeds every
//! `serialize_*` call through to it unchanged, while bookkeeping the byte
//! offset at which each value starts and ends. The result is a
//! [`LocatorIndex`]: a list of ([JSON Pointer], [`Span`]) pairs that a
//! displayer can use to highlight the source bytes of any field without
//! serializing the data again or re-parsing the output.
//!
//! # When to use this
//!
//! Use a `Locator` when you need to point at the exact bytes of a serialized
//! document — for example, a "locate this field" feature in an editor, a
//! diff view, or an error reporter that wants to underline the offending
//! value. Because the spans are captured while the document is written, they
//! reflect the *real* output: escaped strings include their quotes,
//! containers include their brackets, pretty indentation is counted, and
//! custom `Serialize` impls, `flatten`, and skipped fields are all handled
//! according to what the serializer actually emits.
//!
//! If you only need the JSON text, use [`to_string`](crate::to_string) or
//! [`Serializer`] directly — the default serializer is unchanged.
//!
//! # Example
//!
//! ```
//! use serde_json::locator::Locator;
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Doc {
//!     #[serde(rename = "a/b")]
//!     a_b: (String, u32),
//! }
//!
//! let doc = Doc { a_b: ("猫".to_owned(), 2) };
//! let mut locator = Locator::new(Vec::new());
//! doc.serialize(&mut locator).unwrap();
//! let (out, index) = locator.into_parts();
//! let out_str = std::str::from_utf8(&out).unwrap();
//!
//! assert_eq!(out_str, r#"{"a/b":["猫",2]}"#);
//! assert!(index.is_complete());
//!
//! // The string at /a~1b/0 is the complete JSON string, quotes included.
//! let span = index.get("/a~1b/0").unwrap();
//! let span_str = std::str::from_utf8(&out[span.start..span.end]).unwrap();
//! assert_eq!(span_str, r#""猫""#);
//!
//! // The parent array span is accurate too.
//! let parent = index.get("/a~1b").unwrap();
//! let parent_str = std::str::from_utf8(&out[parent.start..parent.end]).unwrap();
//! assert_eq!(parent_str, r#"["猫",2]"#);
//! ```
//!
//! # Span semantics
//!
//! - A span is a half-open byte range `[start, end)` into the output bytes.
//!   Byte offsets are counted after UTF-8 encoding, so multi-byte characters
//!   (such as `猫`) count as multiple bytes.
//! - Every value gets a span: scalars include their surrounding quotes
//!   (`"猫"`), containers include their brackets (`[...]`, `{...}`), and
//!   object keys are *not* indexed (a JSON Pointer identifies values, not
//!   key names).
//! - Array indices are the actual indices produced while serializing,
//!   starting at zero.
//! - Object members are indexed by their logical key, escaped per [RFC 6901]
//!   (`~` becomes `~0`, `/` becomes `~1`). The key is the *logical* key the
//!   `Serialize` impl provides, which matches the decoded JSON string written
//!   to the output.
//! - Pretty-printing indentation and newlines are part of the output bytes
//!   and are included in container spans.
//! - A skipped field (`#[serde(skip)]`) is never serialized, so it never
//!   appears in the index.
//!
//! # Failure and partial indexes
//!
//! Serialization can fail — a custom `Serialize` impl may return an error, or
//! the underlying writer may refuse bytes (a full disk, a closed stream).
//! When that happens the `Locator` still holds every span recorded up to the
//! failure, but the index is explicitly marked *incomplete*
//! ([`LocatorIndex::is_complete`] returns `false`). The index is a fragment of
//! what was written, not a complete map; callers must not treat it as such.
//!
//! ```
//! use serde_json::locator::Locator;
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Broken {
//!     ok: u32,
//!     bad: Failing,
//! }
//!
//! struct Failing;
//!
//! impl Serialize for Failing {
//!     fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
//!     where S: serde::Serializer {
//!         Err(serde::ser::Error::custom("nope"))
//!     }
//! }
//!
//! let mut locator = Locator::new(Vec::new());
//! let result = Broken { ok: 1, bad: Failing }.serialize(&mut locator);
//! assert!(result.is_err());
//!
//! let index = locator.index();
//! assert!(!index.is_complete());
//! // The field written before the failure is still there.
//! assert!(index.get("/ok").is_some());
//! ```
//!
//! [JSON Pointer]: https://datatracker.ietf.org/doc/html/rfc6901
//! [RFC 6901]: https://datatracker.ietf.org/doc/html/rfc6901

use crate::error::{Error, ErrorCode, Result};
use crate::io;
use crate::ser::{CompactFormatter, Formatter, Serializer};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{self, Display, Write as _};
use serde::ser::{self, Impossible, Serialize};

/// A half-open byte range `[start, end)` into the serialized output.
///
/// The range covers the complete JSON value it refers to: a string span
/// includes its surrounding quotes, and a container span includes its
/// brackets. See the [module documentation](self) for the full span
/// semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// The byte offset of the first byte of the value (inclusive).
    pub start: usize,
    /// The byte offset one past the last byte of the value (exclusive).
    pub end: usize,
}

impl Span {
    /// The length in bytes of the spanned output.
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the span covers zero bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// The index produced by a [`Locator`]: every JSON Pointer and the byte span
/// of the value it identifies.
///
/// Entries are stored in serialization order. The index is only complete when
/// serialization finished without error; see [`is_complete`](Self::is_complete).
#[derive(Debug, Clone)]
pub struct LocatorIndex {
    entries: Vec<(String, Span)>,
    complete: bool,
}

impl LocatorIndex {
    /// Returns `true` if the index covers the entire serialization.
    ///
    /// After a successful `serialize` call this is `true`. If serialization
    /// failed partway through, the index is a fragment and this returns
    /// `false`; the entries that are present still refer to bytes actually
    /// written.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// All `(pointer, span)` pairs in serialization order.
    ///
    /// A pointer is an RFC 6901 JSON Pointer (such as `/a~1b/0`), and the span
    /// is the byte range of the value at that pointer.
    #[inline]
    pub fn entries(&self) -> &[(String, Span)] {
        &self.entries
    }

    /// The number of indexed values.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no values were indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up the byte span for an RFC 6901 JSON Pointer.
    ///
    /// Returns `None` if no value was serialized at that pointer (for example
    /// because the field was skipped or does not exist).
    pub fn get(&self, pointer: &str) -> Option<Span> {
        self.entries
            .iter()
            .find(|(p, _)| p == pointer)
            .map(|(_, span)| *span)
    }

    /// Iterates over `(pointer, span)` pairs in serialization order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Span)> {
        self.entries.iter().map(|(p, span)| (p.as_str(), *span))
    }
}

/// A writer wrapper that counts the bytes written through it.
///
/// This is the only bookkeeping the [`Locator`] needs: every span is derived
/// from the byte offset before and after delegating to the real serializer.
pub(crate) struct CountingWriter<W> {
    inner: W,
    count: usize,
}

impl<W> CountingWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        CountingWriter { inner, count: 0 }
    }

    #[inline]
    pub(crate) fn bytes_written(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: io::Write> io::Write for CountingWriter<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = tri!(self.inner.write(buf));
        self.count += n;
        Ok(n)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// The path from the document root to the value currently being serialized.
enum Segment {
    Key(String),
    Index(usize),
}

/// Marks the locator's index as incomplete and returns the error.
///
/// Every fallible operation goes through this so that a failed serialization
/// never leaves the index claiming to be complete.
macro_rules! attempt {
    ($self:expr, $e:expr) => {
        match $e {
            Ok(val) => val,
            Err(err) => {
                $self.index.complete = false;
                return Err(err);
            }
        }
    };
}

/// An explicit Serializer adapter that records the byte range of every output
/// field during the same JSON serialization pass.
///
/// `Locator` wraps a serde_json [`Serializer`] and delegates every operation
/// to it unchanged, so the output bytes are exactly what the default
/// serializer would produce. At the same time it records, for each value, the
/// JSON Pointer and the byte span of the value in the output.
///
/// Construct one with [`Locator::new`] (compact) or [`Locator::pretty`]
/// (pretty-printed), then serialize a value with
/// [`serialize`](Locator::serialize) or by passing `&mut locator` to
/// [`Serialize::serialize`]. Finally call [`into_parts`](Locator::into_parts)
/// to retrieve the writer and the [`LocatorIndex`].
pub struct Locator<W, F = CompactFormatter> {
    inner: Serializer<CountingWriter<W>, F>,
    path: Vec<Segment>,
    index: LocatorIndex,
}

impl<W: io::Write> Locator<W, CompactFormatter> {
    /// Creates a new compact JSON locator.
    #[inline]
    pub fn new(writer: W) -> Self {
        Locator::with_formatter(writer, CompactFormatter)
    }
}

impl<'a, W: io::Write> Locator<W, crate::ser::PrettyFormatter<'a>> {
    /// Creates a new pretty-printed JSON locator.
    #[inline]
    pub fn pretty(writer: W) -> Self {
        Locator::with_formatter(writer, crate::ser::PrettyFormatter::new())
    }
}

impl<W: io::Write, F: Formatter> Locator<W, F> {
    /// Creates a new locator whose output will be written to the writer
    /// specified, using the given formatter.
    #[inline]
    pub fn with_formatter(writer: W, formatter: F) -> Self {
        Locator {
            inner: Serializer::with_formatter(CountingWriter::new(writer), formatter),
            path: Vec::new(),
            index: LocatorIndex {
                entries: Vec::new(),
                complete: true,
            },
        }
    }

    /// Serialize the given data structure, recording spans as it is written.
    ///
    /// This is a convenience equivalent to `value.serialize(&mut *self)`.
    /// On error the locator still holds the spans recorded up to the failure;
    /// see [`is_complete`](Locator::is_complete).
    #[inline]
    pub fn serialize<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    /// Returns the index of spans recorded so far.
    #[inline]
    pub fn index(&self) -> &LocatorIndex {
        &self.index
    }

    /// Returns `true` if serialization has completed without error.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.index.complete
    }

    /// Unwrap the `Writer` and the recorded [`LocatorIndex`].
    #[inline]
    pub fn into_parts(self) -> (W, LocatorIndex) {
        let writer = self.inner.into_inner().into_inner();
        (writer, self.index)
    }

    #[inline]
    fn bytes(&self) -> usize {
        self.inner.writer.bytes_written()
    }

    fn current_pointer(&self) -> String {
        let mut s = String::new();
        for seg in &self.path {
            s.push('/');
            match seg {
                Segment::Key(k) => push_escaped_pointer(&mut s, k),
                Segment::Index(i) => {
                    let mut buf = itoa::Buffer::new();
                    s.push_str(buf.format(*i));
                }
            }
        }
        s
    }

    fn record(&mut self, pointer: String, start: usize, end: usize) {
        self.index.entries.push((pointer, Span { start, end }));
    }

    /// Writes the `{"variant":` wrapper for an externally tagged enum and
    /// pushes the variant key onto the path. Returns the byte offset where
    /// the inner value begins.
    fn begin_variant(&mut self, variant: &str) -> Result<usize> {
        attempt!(self, self.inner.begin_object());
        attempt!(self, self.inner.begin_object_key(true));
        attempt!(
            self,
            ser::Serializer::serialize_str(&mut self.inner, variant)
        );
        attempt!(self, self.inner.end_object_key());
        attempt!(self, self.inner.begin_object_value());
        let inner_start = self.bytes();
        self.path.push(Segment::Key(String::from(variant)));
        Ok(inner_start)
    }

    /// Begins recording an opaque struct (`Number` or `RawValue`) that is
    /// serialized as a single verbatim string. The whole struct is one field
    /// whose value is the raw JSON fragment.
    #[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
    fn opaque_struct(&mut self, kind: OpaqueKind) -> Result<LocatorCompound<'_, W, F>> {
        let start = self.bytes();
        let pointer = self.current_pointer();
        Ok(LocatorCompound {
            loc: self,
            start,
            pointer,
            state: State::First,
            kind: CompoundKind::Opaque(kind),
            index: 0,
            pending_key: None,
        })
    }
}

/// Escapes a logical key for use in an RFC 6901 JSON Pointer.
fn push_escaped_pointer(s: &mut String, key: &str) {
    for ch in key.chars() {
        match ch {
            '~' => s.push_str("~0"),
            '/' => s.push_str("~1"),
            c => s.push(c),
        }
    }
}

/// The state of a compound value (array or object) being serialized.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    First,
    Rest,
}
/// The kind of compound a [`LocatorCompound`] is wrapping.
enum CompoundKind {
    Seq,
    Map,
    TupleVariant { inner_start: usize },
    StructVariant { inner_start: usize },
    #[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
    Opaque(OpaqueKind),
}

/// A compound value being serialized through a [`Locator`].
///
/// This is the `SerializeSeq`/`SerializeMap`/... type returned by the
/// [`Serializer`] impl. It mirrors the bookkeeping of serde_json's internal
/// compound type while recording spans.
pub struct LocatorCompound<'a, W, F> {
    loc: &'a mut Locator<W, F>,
    start: usize,
    pointer: String,
    state: State,
    kind: CompoundKind,
    index: usize,
    pending_key: Option<String>,
}

impl<'a, W, F> ser::Serializer for &'a mut Locator<W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    type SerializeSeq = LocatorCompound<'a, W, F>;
    type SerializeTuple = LocatorCompound<'a, W, F>;
    type SerializeTupleStruct = LocatorCompound<'a, W, F>;
    type SerializeTupleVariant = LocatorCompound<'a, W, F>;
    type SerializeMap = LocatorCompound<'a, W, F>;
    type SerializeStruct = LocatorCompound<'a, W, F>;
    type SerializeStructVariant = LocatorCompound<'a, W, F>;

    #[inline]
    fn serialize_bool(self, value: bool) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_bool(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_i8(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_i16(self, value: i16) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_i16(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_i32(self, value: i32) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_i32(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_i64(self, value: i64) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_i64(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_i128(self, value: i128) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_i128(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_u8(self, value: u8) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_u8(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_u16(self, value: u16) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_u16(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_u32(self, value: u32) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_u32(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_u64(self, value: u64) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_u64(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_u128(self, value: u128) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_u128(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_f32(self, value: f32) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_f32(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_f64(self, value: f64) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_f64(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_char(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_str(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_bytes(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_unit(self) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_unit());
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_unit_struct(_name));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        let start = self.bytes();
        attempt!(
            self,
            self.inner.serialize_unit_variant(_name, _variant_index, variant)
        );
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        // Newtype structs are transparent: the value keeps the same pointer.
        let start = self.bytes();
        attempt!(self, value.serialize(&mut *self));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        let start = self.bytes();
        let pointer = self.current_pointer();
        attempt!(self, self.begin_variant(variant));
        let result = value.serialize(&mut *self);
        self.path.pop();
        attempt!(self, result);
        attempt!(
            self,
            self.inner.end_object_value()
        );
        attempt!(self, self.inner.end_object());
        let end = self.bytes();
        self.record(pointer, start, end);
        Ok(())
    }

    #[inline]
    fn serialize_none(self) -> Result<()> {
        let start = self.bytes();
        attempt!(self, self.inner.serialize_none());
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    #[inline]
    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        // `Some` is transparent: the value keeps the same pointer.
        let start = self.bytes();
        attempt!(self, value.serialize(&mut *self));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        let start = self.bytes();
        attempt!(self, self.inner.begin_array());
        let _ = len;
        let pointer = self.current_pointer();
        Ok(LocatorCompound {
            loc: self,
            start,
            pointer,
            state: State::First,
            kind: CompoundKind::Seq,
            index: 0,
            pending_key: None,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        let start = self.bytes();
        let pointer = self.current_pointer();
        let _ = len;
        attempt!(self, self.begin_variant(variant));
        let inner_start = self.bytes();
        attempt!(self, self.inner.begin_array());
        Ok(LocatorCompound {
            loc: self,
            start,
            pointer,
            state: State::First,
            kind: CompoundKind::TupleVariant { inner_start },
            index: 0,
            pending_key: None,
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        let start = self.bytes();
        attempt!(self, self.inner.begin_object());
        let _ = len;
        let pointer = self.current_pointer();
        Ok(LocatorCompound {
            loc: self,
            start,
            pointer,
            state: State::First,
            kind: CompoundKind::Map,
            index: 0,
            pending_key: None,
        })
    }

    fn serialize_struct(self, name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        #[cfg(feature = "arbitrary_precision")]
        if name == crate::number::TOKEN {
            return self.opaque_struct(OpaqueKind::Number);
        }
        #[cfg(feature = "raw_value")]
        if name == crate::raw::TOKEN {
            return self.opaque_struct(OpaqueKind::RawValue);
        }
        let _ = name;
        self.serialize_map(Some(len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        let start = self.bytes();
        let pointer = self.current_pointer();
        let _ = len;
        attempt!(self, self.begin_variant(variant));
        let inner_start = self.bytes();
        attempt!(self, self.inner.begin_object());
        Ok(LocatorCompound {
            loc: self,
            start,
            pointer,
            state: State::First,
            kind: CompoundKind::StructVariant { inner_start },
            index: 0,
            pending_key: None,
        })
    }

    fn collect_str<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Display,
    {
        let start = self.bytes();
        attempt!(self, self.inner.collect_str(value));
        let end = self.bytes();
        self.record(self.current_pointer(), start, end);
        Ok(())
    }
}

impl<'a, W: io::Write, F: Formatter> LocatorCompound<'a, W, F> {
    // (compound trait methods are in their own impls below)
}

impl<'a, W, F> ser::SerializeSeq for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        match self.kind {
            CompoundKind::Seq | CompoundKind::TupleVariant { .. } => {}
            _ => unreachable!("serialize_element on a non-sequence compound"),
        }
        let first = self.state == State::First;
        attempt!(
            self.loc,
            self.loc.inner.begin_array_value(first)
        );
        self.state = State::Rest;
        let idx = self.index;
        self.index += 1;
        self.loc.path.push(Segment::Index(idx));
        let result = value.serialize(&mut *self.loc);
        self.loc.path.pop();
        attempt!(self.loc, result);
        attempt!(
            self.loc,
            self.loc.inner.end_array_value()
        );
        Ok(())
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeTuple for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeTupleStruct for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeTupleVariant for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeMap for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_key<T>(&mut self, key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        match self.kind {
            CompoundKind::Map | CompoundKind::StructVariant { .. } => {}
            _ => unreachable!("serialize_key on a non-map compound"),
        }
        let first = self.state == State::First;
        attempt!(
            self.loc,
            self.loc.inner.begin_object_key(first)
        );
        self.state = State::Rest;

        // Capture the logical key for the JSON Pointer.
        let mut key_string = String::new();
        attempt!(self.loc, key.serialize(&mut KeyCapture { key: &mut key_string }));

        // Write the real key through the real serializer, so escaping and
        // custom formatters are exactly what the default serializer does.
        attempt!(self.loc, self.loc.inner.serialize_map_key(key));

        attempt!(
            self.loc,
            self.loc.inner.end_object_key()
        );
        self.pending_key = Some(key_string);
        Ok(())
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        attempt!(
            self.loc,
            self.loc.inner.begin_object_value()
        );
        let key = self
            .pending_key
            .take()
            .expect("serialize_value called without a preceding serialize_key");
        self.loc.path.push(Segment::Key(key));
        let result = value.serialize(&mut *self.loc);
        self.loc.path.pop();
        attempt!(self.loc, result);
        attempt!(
            self.loc,
            self.loc.inner.end_object_value()
        );
        Ok(())
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeStruct for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        #[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
        if let CompoundKind::Opaque(kind) = self.kind {
            let expected = match kind {
                #[cfg(feature = "arbitrary_precision")]
                OpaqueKind::Number => crate::number::TOKEN,
                #[cfg(feature = "raw_value")]
                OpaqueKind::RawValue => crate::raw::TOKEN,
            };
            if key != expected {
                return Err(match kind {
                    #[cfg(feature = "arbitrary_precision")]
                    OpaqueKind::Number => Error::syntax(ErrorCode::InvalidNumber, 0, 0),
                    #[cfg(feature = "raw_value")]
                    OpaqueKind::RawValue => ser::Error::custom("expected RawValue"),
                });
            }
            let emitter = TokenEmitter {
                loc: &mut *self.loc,
                kind,
            };
            attempt!(self.loc, value.serialize(emitter));
            return Ok(());
        }
        ser::SerializeMap::serialize_entry(self, key, value)
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W, F> ser::SerializeStructVariant for LocatorCompound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        ser::SerializeStruct::serialize_field(self, key, value)
    }

    fn end(self) -> Result<()> {
        self.finish()
    }
}

impl<'a, W: io::Write, F: Formatter> LocatorCompound<'a, W, F> {
    /// Closes the compound and records its span.
    fn finish(self) -> Result<()> {
        let LocatorCompound {
            loc,
            start,
            pointer,
            kind,
            ..
        } = self;
        match kind {
            CompoundKind::Seq => {
                attempt!(loc, loc.inner.end_array());
            }
            CompoundKind::Map => {
                attempt!(loc, loc.inner.end_object());
            }
            CompoundKind::TupleVariant { inner_start } => {
                attempt!(loc, loc.inner.end_array());
                let inner_end = loc.bytes();
                loc.record(loc.current_pointer(), inner_start, inner_end);
                attempt!(loc, loc.inner.end_object_value());
                loc.path.pop();
                attempt!(loc, loc.inner.end_object());
            }
            CompoundKind::StructVariant { inner_start } => {
                attempt!(loc, loc.inner.end_object());
                let inner_end = loc.bytes();
                loc.record(loc.current_pointer(), inner_start, inner_end);
                attempt!(loc, loc.inner.end_object_value());
                loc.path.pop();
                attempt!(loc, loc.inner.end_object());
            }
            #[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
            CompoundKind::Opaque(_) => {
                let end = loc.bytes();
                loc.record(pointer, start, end);
                return Ok(());
            }
        }
        let end = loc.bytes();
        loc.record(pointer, start, end);
        Ok(())
    }
}

/// Captures the logical key of a map entry for the JSON Pointer.
///
/// JSON object keys are string-like scalars. This serializer records the
/// logical key value (e.g. `a/b` or `true`) without writing anything; the
/// actual output is written separately through the real serializer. The set
/// of accepted key types deliberately mirrors serde_json's internal
/// `MapKeySerializer`.
struct KeyCapture<'a> {
    key: &'a mut String,
}

impl<'a> ser::Serializer for &mut KeyCapture<'a> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Impossible<(), Error>;
    type SerializeTuple = Impossible<(), Error>;
    type SerializeTupleStruct = Impossible<(), Error>;
    type SerializeTupleVariant = Impossible<(), Error>;
    type SerializeMap = Impossible<(), Error>;
    type SerializeStruct = Impossible<(), Error>;
    type SerializeStructVariant = Impossible<(), Error>;

    #[inline]
    fn serialize_str(self, value: &str) -> Result<()> {
        self.key.push_str(value);
        Ok(())
    }

    #[inline]
    fn serialize_bool(self, value: bool) -> Result<()> {
        self.key.push_str(if value { "true" } else { "false" });
        Ok(())
    }

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_i16(self, value: i16) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_i32(self, value: i32) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_i64(self, value: i64) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_i128(self, value: i128) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_u8(self, value: u8) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_u16(self, value: u16) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_u32(self, value: u32) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_u64(self, value: u64) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_u128(self, value: u128) -> Result<()> {
        let mut buf = itoa::Buffer::new();
        self.key.push_str(buf.format(value));
        Ok(())
    }

    #[inline]
    fn serialize_f32(self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::syntax(ErrorCode::FloatKeyMustBeFinite, 0, 0));
        }
        let mut buf = zmij::Buffer::new();
        self.key.push_str(buf.format_finite(value));
        Ok(())
    }

    #[inline]
    fn serialize_f64(self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::syntax(ErrorCode::FloatKeyMustBeFinite, 0, 0));
        }
        let mut buf = zmij::Buffer::new();
        self.key.push_str(buf.format_finite(value));
        Ok(())
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<()> {
        self.key.push(value);
        Ok(())
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.key.push_str(variant);
        Ok(())
    }

    #[inline]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    #[inline]
    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn collect_str<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Display,
    {
        struct Adapter<'a>(&'a mut String);
        impl<'a> fmt::Write for Adapter<'a> {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0.push_str(s);
                Ok(())
            }
        }
        write!(Adapter(self.key), "{}", value).expect("writing to a String cannot fail");
        Ok(())
    }

    #[inline]
    fn serialize_bytes(self, _value: &[u8]) -> Result<()> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_unit(self) -> Result<()> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_none(self) -> Result<()> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }

    #[inline]
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(Error::syntax(ErrorCode::KeyMustBeAString, 0, 0))
    }
}

/// The kind of opaque (raw) value being serialized through a locator.
#[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
#[derive(Clone, Copy)]
enum OpaqueKind {
    #[cfg(feature = "arbitrary_precision")]
    Number,
    #[cfg(feature = "raw_value")]
    RawValue,
}

/// Serializer for the opaque payload of a `$serde_json::private::*` struct.
///
/// `Number` (arbitrary_precision) and `RawValue` (raw_value) serialize a
/// single string field that is written verbatim as a number or raw JSON
/// fragment, without quotes. This serializer mirrors that behavior so the
/// locator records the opaque value as one leaf span.
#[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
struct TokenEmitter<'a, W, F> {
    loc: &'a mut Locator<W, F>,
    kind: OpaqueKind,
}

#[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
impl<'a, W, F> TokenEmitter<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    fn bad(&self) -> Error {
        match self.kind {
            #[cfg(feature = "arbitrary_precision")]
            OpaqueKind::Number => Error::syntax(ErrorCode::InvalidNumber, 0, 0),
            #[cfg(feature = "raw_value")]
            OpaqueKind::RawValue => ser::Error::custom("expected RawValue"),
        }
    }
}

#[cfg(any(feature = "arbitrary_precision", feature = "raw_value"))]
impl<'a, W, F> ser::Serializer for TokenEmitter<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Impossible<(), Error>;
    type SerializeTuple = Impossible<(), Error>;
    type SerializeTupleStruct = Impossible<(), Error>;
    type SerializeTupleVariant = Impossible<(), Error>;
    type SerializeMap = Impossible<(), Error>;
    type SerializeStruct = Impossible<(), Error>;
    type SerializeStructVariant = Impossible<(), Error>;

    fn serialize_str(self, value: &str) -> Result<()> {
        match self.kind {
            #[cfg(feature = "arbitrary_precision")]
            OpaqueKind::Number => self.loc.inner.write_number_str(value),
            #[cfg(feature = "raw_value")]
            OpaqueKind::RawValue => self.loc.inner.write_raw_fragment(value),
        }
    }

    fn serialize_bool(self, _v: bool) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_i8(self, _v: i8) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_i16(self, _v: i16) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_i32(self, _v: i32) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_i64(self, _v: i64) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_i128(self, _v: i128) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_u8(self, _v: u8) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_u16(self, _v: u16) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_u32(self, _v: u32) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_u64(self, _v: u64) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_u128(self, _v: u128) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_f32(self, _v: f32) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_f64(self, _v: f64) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_char(self, _v: char) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_unit(self) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(self.bad())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(self.bad())
    }

    fn serialize_none(self) -> Result<()> {
        Err(self.bad())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(self.bad())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(self.bad())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(self.bad())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(self.bad())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(self.bad())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(self.bad())
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(self.bad())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(self.bad())
    }

    fn collect_str<T>(self, _value: &T) -> Result<()>
    where
        T: ?Sized + Display,
    {
        Err(self.bad())
    }
}
