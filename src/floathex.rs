//! Explicit float field protocol adapter (`floathex`).
//!
//! This module implements an opt-in, path-based protocol adaptation for
//! `f64` fields. Fields whose path is *registered* in a [`FloatHexProtocol`]
//! are encoded as a JSON string containing the fixed 16-character lowercase
//! hex bit pattern of the `f64`; every other field keeps the ordinary JSON
//! format. Deserialization accepts the protocol only on the very same
//! registered paths, so negative zero and NaN payloads survive a round trip
//! bit-for-bit without any decimal intermediary.
//!
//! The plain [`serde_json::Serializer`][crate::Serializer],
//! [`serde_json::Deserializer`][crate::Deserializer] and the top-level
//! `to_string` / `from_str` functions are completely unaffected; the
//! adaptation only applies to values that go through a [`FloatHexProtocol`]
//! instance.
//!
//! # Byte format of the protocol
//!
//! A registered `f64` is serialized as a JSON string of **exactly 16 ASCII
//! characters**, each one a lowercase hex digit (`0-9` or `a-f`). The 16
//! digits are the hexadecimal representation of the 64-bit IEEE-754
//! binary64 bit pattern of the value (i.e. `f64::to_bits()`), written
//! most-significant nibble first: two hex digits per byte, big-endian byte
//! order. Examples:
//!
//! - `1.0` -> bits `0x3ff0000000000000` -> `"3ff0000000000000"`
//! - `-0.0` -> bits `0x8000000000000000` -> `"8000000000000000"`
//! - quiet NaN with payload 1 -> bits `0x7ff8000000000001` ->
//!   `"7ff8000000000001"`
//!
//! Decoding parses the 16 hex digits directly into a `u64` and calls
//! `f64::from_bits`; no decimal floating point conversion is ever involved,
//! so the sign of zero and every NaN payload bit are preserved exactly.
//! Uppercase digits, other lengths and non-hex characters are rejected, and
//! the rejection error is labeled with the JSON path of the offending field
//! (for example `invalid float hex at $.samples.3: ...`).
//!
//! # Paths
//!
//! Paths are dot-separated sequences of *actual Serde field names* — that
//! is, the names after `#[serde(rename = "...")]` and friends have been
//! applied. Sequence elements (including nested lists) contribute their
//! decimal index as a path segment, and map keys contribute their string
//! form. The wildcard segment `*` matches exactly one arbitrary segment,
//! which is how list elements are usually registered:
//!
//! - `value` — the field `value` of the root struct
//! - `outer.inner.value` — a field three levels deep
//! - `matrix.*.*` — every element of a nested `Vec<Vec<f64>>`
//! - `lookup.*` — every value of a `HashMap<String, f64>` under `lookup`
//!
//! Enum variant names are not part of the path; fields inside a variant are
//! addressed by their own names as usual.
//!
//! # Example
//!
//! ```
//! use serde::{Deserialize, Serialize};
//! use serde_json::floathex::FloatHexProtocol;
//!
//! #[derive(Serialize, Deserialize)]
//! struct Reading {
//!     name: String,
//!     value: f64,
//! }
//!
//! let proto = FloatHexProtocol::new(["value"]).unwrap();
//! let reading = Reading { name: "sensor".to_owned(), value: -0.0 };
//!
//! let json = proto.to_string(&reading).unwrap();
//! assert_eq!(json, r#"{"name":"sensor","value":"8000000000000000"}"#);
//!
//! let back: Reading = proto.from_str(&json).unwrap();
//! assert_eq!(back.value.to_bits(), (-0.0f64).to_bits());
//! ```
//!
//! # Verifying with cargo
//!
//! The protocol is exercised by a dedicated integration test target:
//!
//! `cargo test --test floathex`
//!
//! A quick check that the ordinary serializer/deserializer behavior is
//! untouched can be run with:
//!
//! `cargo test --lib`

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt::{self, Display};

use serde::de::{
    self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor,
};
use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::Result;

/// Error returned when a [`FloatHexProtocol`] path pattern is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolError {
    msg: String,
}

impl Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProtocolError {}

/// A path-based protocol adapter that encodes registered `f64` fields as
/// fixed 16-character lowercase hex bit-pattern strings.
///
/// See the [module level documentation](self) for the byte format and the
/// path syntax.
#[derive(Debug, Clone)]
pub struct FloatHexProtocol {
    patterns: Vec<Pattern>,
}

type Pattern = Vec<PatSeg>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum PatSeg {
    /// Matches exactly one arbitrary segment.
    Any,
    /// Matches a field name / map key / decimal sequence index.
    Key(String),
}

impl FloatHexProtocol {
    /// Registers the given dot-separated path patterns.
    ///
    /// Each pattern is a non-empty sequence of dot-separated segments. A
    /// segment is either a literal Serde field name (after `rename`), a
    /// decimal sequence index, or the single-segment wildcard `*`. Empty
    /// patterns and empty segments are rejected.
    pub fn new<I, S>(paths: I) -> core::result::Result<Self, ProtocolError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut patterns = Vec::new();
        for path in paths {
            let path = path.as_ref();
            if path.is_empty() {
                return Err(ProtocolError {
                    msg: "floathex path pattern must not be empty".to_owned(),
                });
            }
            let mut segments = Vec::new();
            for seg in path.split('.') {
                if seg.is_empty() {
                    return Err(ProtocolError {
                        msg: alloc::format!(
                            "floathex path pattern {:?} contains an empty segment",
                            path
                        ),
                    });
                }
                segments.push(if seg == "*" {
                    PatSeg::Any
                } else {
                    PatSeg::Key(seg.to_owned())
                });
            }
            patterns.push(segments);
        }
        Ok(FloatHexProtocol { patterns })
    }

    /// Serializes a value to a JSON string, encoding `f64` fields on the
    /// registered paths as 16-character lowercase hex bit-pattern strings.
    pub fn to_string<T>(&self, value: &T) -> Result<String>
    where
        T: ?Sized + Serialize,
    {
        let vec = tri!(self.to_vec(value));
        // serde_json only emits valid UTF-8.
        Ok(String::from_utf8(vec).expect("serde_json output is valid UTF-8"))
    }

    /// Serializes a value to a JSON byte vector; see [`FloatHexProtocol::to_string`].
    pub fn to_vec<T>(&self, value: &T) -> Result<Vec<u8>>
    where
        T: ?Sized + Serialize,
    {
        let mut writer = Vec::with_capacity(128);
        {
            let mut ser = crate::ser::Serializer::new(&mut writer);
            let path = Path::Root;
            tri!(value.serialize(SerWrap {
                ser: &mut ser,
                path: &path,
                proto: &self.patterns,
            }));
        }
        Ok(writer)
    }

    /// Deserializes a value from a JSON string, accepting the hex
    /// bit-pattern strings only on the registered paths.
    pub fn from_str<'a, T>(&self, s: &'a str) -> Result<T>
    where
        T: Deserialize<'a>,
    {
        self.from_slice(s.as_bytes())
    }

    /// Deserializes a value from a JSON byte slice; see
    /// [`FloatHexProtocol::from_str`].
    pub fn from_slice<'a, T>(&self, bytes: &'a [u8]) -> Result<T>
    where
        T: Deserialize<'a>,
    {
        let mut de = crate::de::Deserializer::from_slice(bytes);
        let path = Path::Root;
        let value = tri!(T::deserialize(DeWrap {
            de: &mut de,
            path: &path,
            proto: &self.patterns,
        }));
        // Make sure the whole stream has been consumed.
        tri!(de.end());
        Ok(value)
    }
}

/// Linked list of path segments from the root to the value currently being
/// serialized or deserialized.
enum Path<'a> {
    Root,
    Seq {
        parent: &'a Path<'a>,
        index: usize,
    },
    Struct {
        parent: &'a Path<'a>,
        key: &'static str,
    },
    Map {
        parent: &'a Path<'a>,
        key: &'a str,
    },
}

impl<'a> Path<'a> {
    fn parent(&self) -> Option<&'a Path<'a>> {
        match self {
            Path::Root => None,
            Path::Seq { parent, .. } | Path::Struct { parent, .. } | Path::Map { parent, .. } => {
                Some(parent)
            }
        }
    }
}

impl Display for Path<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Path::Root => f.write_str("$"),
            Path::Seq { parent, index } => {
                tri!(parent.fmt(f));
                write!(f, ".{}", index)
            }
            Path::Struct { parent, key } | Path::Map { parent, key } => {
                tri!(parent.fmt(f));
                write!(f, ".{}", key)
            }
        }
    }
}

/// Returns true if any registered pattern exactly matches the path.
fn matches(patterns: &[Pattern], path: &Path) -> bool {
    patterns.iter().any(|p| pattern_matches(p, path))
}

fn pattern_matches(pattern: &[PatSeg], path: &Path) -> bool {
    let mut node = path;
    for seg in pattern.iter().rev() {
        let (parent, ok) = match node {
            Path::Root => return false,
            Path::Seq { parent, index } => {
                let ok = match seg {
                    PatSeg::Any => true,
                    PatSeg::Key(k) => {
                        let mut buf = itoa::Buffer::new();
                        k.as_str() == buf.format(*index)
                    }
                };
                (parent, ok)
            }
            Path::Struct { parent, key } | Path::Map { parent, key } => {
                let ok = match seg {
                    PatSeg::Any => true,
                    PatSeg::Key(k) => k == key,
                };
                (parent, ok)
            }
        };
        if !ok {
            return false;
        }
        node = parent;
    }
    node.parent().is_none()
}

/// Writes the 16-character lowercase hex bit pattern of `bits` into `out`,
/// most-significant nibble first.
fn write_hex(bits: u64, out: &mut [u8; 16]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut i = 0;
    while i < 16 {
        let shift = 60 - 4 * i as u32;
        out[i] = HEX[((bits >> shift) & 0xf) as usize];
        i += 1;
    }
}

/// Parses exactly 16 lowercase hex digits into the raw `u64` bit pattern.
/// No decimal floating point conversion is involved.
fn parse_hex_bits(s: &str) -> core::result::Result<u64, &'static str> {
    let bytes = s.as_bytes();
    if bytes.len() != 16 {
        return Err("expected exactly 16 hex digits");
    }
    let mut bits = 0u64;
    for &c in bytes {
        let digit = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => return Err("uppercase hex digits are not allowed"),
            _ => return Err("expected only lowercase hex digits 0-9a-f"),
        };
        bits = bits << 4 | u64::from(digit);
    }
    Ok(bits)
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serializer wrapper that reroutes registered `f64` values to their hex
/// bit-pattern string form.
struct SerWrap<'a, S> {
    ser: S,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

/// `Serialize` wrapper that applies [`SerWrap`] to the serializer before
/// serializing the wrapped value.
struct Capture<'a, 'b, T: ?Sized> {
    value: &'b T,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

impl<T: ?Sized + Serialize> Serialize for Capture<'_, '_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        self.value.serialize(SerWrap {
            ser: serializer,
            path: self.path,
            proto: self.proto,
        })
    }
}

impl<'a, S> Serializer for SerWrap<'a, S>
where
    S: Serializer,
{
    type Ok = S::Ok;
    type Error = S::Error;

    type SerializeSeq = SeqSer<'a, S::SerializeSeq>;
    type SerializeTuple = SeqSer<'a, S::SerializeTuple>;
    type SerializeTupleStruct = SeqSer<'a, S::SerializeTupleStruct>;
    type SerializeTupleVariant = SeqSer<'a, S::SerializeTupleVariant>;
    type SerializeMap = MapSer<'a, S::SerializeMap>;
    type SerializeStruct = StructSer<'a, S::SerializeStruct>;
    type SerializeStructVariant = StructSer<'a, S::SerializeStructVariant>;

    fn serialize_f64(self, v: f64) -> core::result::Result<Self::Ok, Self::Error> {
        if matches(self.proto, self.path) {
            let mut buf = [0u8; 16];
            write_hex(v.to_bits(), &mut buf);
            // The buffer is pure ASCII by construction.
            let s = core::str::from_utf8(&buf).expect("hex output is ASCII");
            self.ser.serialize_str(s)
        } else {
            self.ser.serialize_f64(v)
        }
    }

    fn serialize_bool(self, v: bool) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_bool(v)
    }

    fn serialize_i8(self, v: i8) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_i8(v)
    }

    fn serialize_i16(self, v: i16) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_i16(v)
    }

    fn serialize_i32(self, v: i32) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_i32(v)
    }

    fn serialize_i64(self, v: i64) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_i64(v)
    }

    fn serialize_i128(self, v: i128) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_i128(v)
    }

    fn serialize_u8(self, v: u8) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_u8(v)
    }

    fn serialize_u16(self, v: u16) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_u16(v)
    }

    fn serialize_u32(self, v: u32) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_u32(v)
    }

    fn serialize_u64(self, v: u64) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_u64(v)
    }

    fn serialize_u128(self, v: u128) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_u128(v)
    }

    fn serialize_f32(self, v: f32) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_f32(v)
    }

    fn serialize_char(self, v: char) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_char(v)
    }

    fn serialize_str(self, v: &str) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_str(v)
    }

    fn serialize_bytes(self, v: &[u8]) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_bytes(v)
    }

    fn serialize_none(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_none()
    }

    fn serialize_some<T: ?Sized + Serialize>(
        self,
        value: &T,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_some(&Capture {
            value,
            path: self.path,
            proto: self.proto,
        })
    }

    fn serialize_unit(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_unit()
    }

    fn serialize_unit_struct(
        self,
        name: &'static str,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_unit_struct(name)
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser
            .serialize_unit_variant(name, variant_index, variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_newtype_struct(
            name,
            &Capture {
                value,
                path: self.path,
                proto: self.proto,
            },
        )
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.serialize_newtype_variant(
            name,
            variant_index,
            variant,
            &Capture {
                value,
                path: self.path,
                proto: self.proto,
            },
        )
    }

    fn serialize_seq(
        self,
        len: Option<usize>,
    ) -> core::result::Result<Self::SerializeSeq, Self::Error> {
        let inner = tri!(self.ser.serialize_seq(len));
        Ok(SeqSer {
            inner,
            path: self.path,
            proto: self.proto,
            index: 0,
        })
    }

    fn serialize_tuple(
        self,
        len: usize,
    ) -> core::result::Result<Self::SerializeTuple, Self::Error> {
        let inner = tri!(self.ser.serialize_tuple(len));
        Ok(SeqSer {
            inner,
            path: self.path,
            proto: self.proto,
            index: 0,
        })
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> core::result::Result<Self::SerializeTupleStruct, Self::Error> {
        let inner = tri!(self.ser.serialize_tuple_struct(name, len));
        Ok(SeqSer {
            inner,
            path: self.path,
            proto: self.proto,
            index: 0,
        })
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> core::result::Result<Self::SerializeTupleVariant, Self::Error> {
        let inner = tri!(self
            .ser
            .serialize_tuple_variant(name, variant_index, variant, len));
        Ok(SeqSer {
            inner,
            path: self.path,
            proto: self.proto,
            index: 0,
        })
    }

    fn serialize_map(
        self,
        len: Option<usize>,
    ) -> core::result::Result<Self::SerializeMap, Self::Error> {
        let inner = tri!(self.ser.serialize_map(len));
        Ok(MapSer {
            inner,
            path: self.path,
            proto: self.proto,
            key: None,
        })
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> core::result::Result<Self::SerializeStruct, Self::Error> {
        let inner = tri!(self.ser.serialize_struct(name, len));
        Ok(StructSer {
            inner,
            path: self.path,
            proto: self.proto,
        })
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> core::result::Result<Self::SerializeStructVariant, Self::Error> {
        let inner = tri!(self
            .ser
            .serialize_struct_variant(name, variant_index, variant, len));
        Ok(StructSer {
            inner,
            path: self.path,
            proto: self.proto,
        })
    }

    fn collect_str<T: ?Sized + Display>(
        self,
        value: &T,
    ) -> core::result::Result<Self::Ok, Self::Error> {
        self.ser.collect_str(value)
    }

    fn is_human_readable(&self) -> bool {
        self.ser.is_human_readable()
    }
}

/// Compound wrapper for sequences, tuples, tuple structs and tuple variants.
/// Elements are addressed by their decimal index.
struct SeqSer<'a, C> {
    inner: C,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
    index: usize,
}

impl<'a, C> SeqSer<'a, C> {
    fn next_link(&mut self) -> Path<'a> {
        let index = self.index;
        self.index += 1;
        Path::Seq {
            parent: self.path,
            index,
        }
    }
}

impl<C: SerializeSeq> SerializeSeq for SeqSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_element<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = self.next_link();
        self.inner.serialize_element(&Capture {
            value,
            path: &link,
            proto: self.proto,
        })
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

impl<C: SerializeTuple> SerializeTuple for SeqSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_element<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = self.next_link();
        self.inner.serialize_element(&Capture {
            value,
            path: &link,
            proto: self.proto,
        })
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

impl<C: SerializeTupleStruct> SerializeTupleStruct for SeqSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = self.next_link();
        self.inner.serialize_field(&Capture {
            value,
            path: &link,
            proto: self.proto,
        })
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

impl<C: SerializeTupleVariant> SerializeTupleVariant for SeqSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = self.next_link();
        self.inner.serialize_field(&Capture {
            value,
            path: &link,
            proto: self.proto,
        })
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

/// Compound wrapper for maps. Keys contribute their string form to the path.
struct MapSer<'a, C> {
    inner: C,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
    key: Option<String>,
}

impl<C: SerializeMap> SerializeMap for MapSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_key<T: ?Sized + Serialize>(
        &mut self,
        key: &T,
    ) -> core::result::Result<(), Self::Error> {
        self.key = key.serialize(KeyToString).ok();
        self.inner.serialize_key(key)
    }

    fn serialize_value<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = Path::Map {
            parent: self.path,
            key: self.key.as_deref().unwrap_or("?"),
        };
        self.inner.serialize_value(&Capture {
            value,
            path: &link,
            proto: self.proto,
        })
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

/// Compound wrapper for structs and struct variants. Fields contribute their
/// actual Serde (post-rename) names to the path.
struct StructSer<'a, C> {
    inner: C,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

impl<C: SerializeStruct> SerializeStruct for StructSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = Path::Struct {
            parent: self.path,
            key,
        };
        self.inner.serialize_field(
            key,
            &Capture {
                value,
                path: &link,
                proto: self.proto,
            },
        )
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

impl<C: SerializeStructVariant> SerializeStructVariant for StructSer<'_, C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> core::result::Result<(), Self::Error> {
        let link = Path::Struct {
            parent: self.path,
            key,
        };
        self.inner.serialize_field(
            key,
            &Capture {
                value,
                path: &link,
                proto: self.proto,
            },
        )
    }

    fn end(self) -> core::result::Result<Self::Ok, Self::Error> {
        self.inner.end()
    }
}

/// Error type for map keys that cannot be represented as a path segment.
#[derive(Debug)]
struct UnsupportedKey;

impl Display for UnsupportedKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("map key cannot be represented as a floathex path segment")
    }
}

impl ser::StdError for UnsupportedKey {}

impl ser::Error for UnsupportedKey {
    fn custom<T: Display>(_msg: T) -> Self {
        UnsupportedKey
    }
}

/// Mini serializer that renders a map key as its path-segment string.
struct KeyToString;

impl Serializer for KeyToString {
    type Ok = String;
    type Error = UnsupportedKey;

    type SerializeSeq = ser::Impossible<String, UnsupportedKey>;
    type SerializeTuple = ser::Impossible<String, UnsupportedKey>;
    type SerializeTupleStruct = ser::Impossible<String, UnsupportedKey>;
    type SerializeTupleVariant = ser::Impossible<String, UnsupportedKey>;
    type SerializeMap = ser::Impossible<String, UnsupportedKey>;
    type SerializeStruct = ser::Impossible<String, UnsupportedKey>;
    type SerializeStructVariant = ser::Impossible<String, UnsupportedKey>;

    fn serialize_str(self, v: &str) -> core::result::Result<String, UnsupportedKey> {
        Ok(v.to_owned())
    }

    fn serialize_bool(self, v: bool) -> core::result::Result<String, UnsupportedKey> {
        Ok(if v {
            "true".to_owned()
        } else {
            "false".to_owned()
        })
    }

    fn serialize_i8(self, v: i8) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i16(self, v: i16) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i32(self, v: i32) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i64(self, v: i64) -> core::result::Result<String, UnsupportedKey> {
        let mut buf = itoa::Buffer::new();
        Ok(buf.format(v).to_owned())
    }

    fn serialize_i128(self, v: i128) -> core::result::Result<String, UnsupportedKey> {
        let mut buf = itoa::Buffer::new();
        Ok(buf.format(v).to_owned())
    }

    fn serialize_u8(self, v: u8) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u16(self, v: u16) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u32(self, v: u32) -> core::result::Result<String, UnsupportedKey> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u64(self, v: u64) -> core::result::Result<String, UnsupportedKey> {
        let mut buf = itoa::Buffer::new();
        Ok(buf.format(v).to_owned())
    }

    fn serialize_u128(self, v: u128) -> core::result::Result<String, UnsupportedKey> {
        let mut buf = itoa::Buffer::new();
        Ok(buf.format(v).to_owned())
    }

    fn serialize_f32(self, _v: f32) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_f64(self, _v: f64) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_char(self, v: char) -> core::result::Result<String, UnsupportedKey> {
        Ok(v.to_string())
    }

    fn serialize_bytes(self, _v: &[u8]) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_none(self) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_some<T: ?Sized + Serialize>(
        self,
        _value: &T,
    ) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_unit(self) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_unit_struct(
        self,
        _name: &'static str,
    ) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> core::result::Result<String, UnsupportedKey> {
        Ok(variant.to_owned())
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> core::result::Result<String, UnsupportedKey> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> core::result::Result<String, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_seq(
        self,
        _len: Option<usize>,
    ) -> core::result::Result<Self::SerializeSeq, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_tuple(
        self,
        _len: usize,
    ) -> core::result::Result<Self::SerializeTuple, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> core::result::Result<Self::SerializeTupleStruct, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> core::result::Result<Self::SerializeTupleVariant, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_map(
        self,
        _len: Option<usize>,
    ) -> core::result::Result<Self::SerializeMap, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> core::result::Result<Self::SerializeStruct, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> core::result::Result<Self::SerializeStructVariant, UnsupportedKey> {
        Err(UnsupportedKey)
    }

    fn is_human_readable(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Deserialization
// ---------------------------------------------------------------------------

/// Deserializer wrapper that accepts hex bit-pattern strings for `f64`
/// values on registered paths.
struct DeWrap<'a, D> {
    de: D,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

/// Seed wrapper that applies [`DeWrap`] to the deserializer handed to the
/// wrapped seed.
struct SeedWrap<'a, S> {
    seed: S,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for SeedWrap<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> core::result::Result<Self::Value, D::Error> {
        self.seed.deserialize(DeWrap {
            de: deserializer,
            path: self.path,
            proto: self.proto,
        })
    }
}

macro_rules! fwd_plain {
    ($($method:ident ( $($arg:ident : $ty:ty),* ) ;)*) => {$(
        fn $method<V: Visitor<'de>>(
            self,
            $($arg: $ty,)*
            visitor: V,
        ) -> core::result::Result<V::Value, Self::Error> {
            self.de.$method($($arg,)* visitor)
        }
    )*};
}

macro_rules! fwd_tracked {
    ($($method:ident ( $($arg:ident : $ty:ty),* ) ;)*) => {$(
        fn $method<V: Visitor<'de>>(
            self,
            $($arg: $ty,)*
            visitor: V,
        ) -> core::result::Result<V::Value, Self::Error> {
            self.de.$method(
                $($arg,)*
                VisWrap {
                    visitor,
                    path: self.path,
                    proto: self.proto,
                },
            )
        }
    )*};
}

macro_rules! fwd_record {
    ($($method:ident ;)*) => {$(
        fn $method<V: Visitor<'de>>(
            self,
            visitor: V,
        ) -> core::result::Result<V::Value, Self::Error> {
            self.de.$method(RecordVisitor {
                visitor,
                slot: self.slot,
            })
        }
    )*};
}

impl<'de, D> Deserializer<'de> for DeWrap<'_, D>
where
    D: Deserializer<'de>,
{
    type Error = D::Error;

    fn deserialize_f64<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> core::result::Result<V::Value, Self::Error> {
        use serde::de::Error as _;

        if !matches(self.proto, self.path) {
            return self.de.deserialize_f64(visitor);
        }
        let text = match self.de.deserialize_str(HexStringVisitor) {
            Ok(text) => text,
            Err(err) => {
                return Err(D::Error::custom(alloc::format!(
                    "invalid float hex at {}: {}",
                    self.path,
                    err
                )));
            }
        };
        let bits = match parse_hex_bits(&text) {
            Ok(bits) => bits,
            Err(msg) => {
                return Err(D::Error::custom(alloc::format!(
                    "invalid float hex at {}: {}",
                    self.path,
                    msg
                )));
            }
        };
        visitor.visit_f64(f64::from_bits(bits))
    }

    fwd_tracked! {
        deserialize_any();
        deserialize_option();
        deserialize_newtype_struct(name: &'static str);
        deserialize_seq();
        deserialize_tuple(len: usize);
        deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map();
        deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
    }

    fwd_plain! {
        deserialize_bool();
        deserialize_i8();
        deserialize_i16();
        deserialize_i32();
        deserialize_i64();
        deserialize_i128();
        deserialize_u8();
        deserialize_u16();
        deserialize_u32();
        deserialize_u64();
        deserialize_u128();
        deserialize_f32();
        deserialize_char();
        deserialize_str();
        deserialize_string();
        deserialize_bytes();
        deserialize_byte_buf();
        deserialize_unit();
        deserialize_unit_struct(name: &'static str);
        deserialize_identifier();
        deserialize_ignored_any();
    }

    fn is_human_readable(&self) -> bool {
        self.de.is_human_readable()
    }
}

/// Visitor wrapper that keeps path tracking alive through compound and
/// optional values.
struct VisWrap<'a, V> {
    visitor: V,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

macro_rules! visit_forward {
    ($($method:ident ( $($arg:ident : $ty:ty),* ) ;)*) => {$(
        fn $method<E: de::Error>(
            self,
            $($arg: $ty,)*
        ) -> core::result::Result<Self::Value, E> {
            self.visitor.$method($($arg,)*)
        }
    )*};
}

impl<'de, V> Visitor<'de> for VisWrap<'_, V>
where
    V: Visitor<'de>,
{
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.visitor.expecting(formatter)
    }

    visit_forward! {
        visit_bool(v: bool);
        visit_i8(v: i8);
        visit_i16(v: i16);
        visit_i32(v: i32);
        visit_i64(v: i64);
        visit_i128(v: i128);
        visit_u8(v: u8);
        visit_u16(v: u16);
        visit_u32(v: u32);
        visit_u64(v: u64);
        visit_u128(v: u128);
        visit_f32(v: f32);
        visit_f64(v: f64);
        visit_char(v: char);
        visit_str(v: &str);
        visit_borrowed_str(v: &'de str);
        visit_string(v: String);
        visit_bytes(v: &[u8]);
        visit_borrowed_bytes(v: &'de [u8]);
        visit_byte_buf(v: Vec<u8>);
        visit_none();
        visit_unit();
    }

    fn visit_some<D2: Deserializer<'de>>(
        self,
        deserializer: D2,
    ) -> core::result::Result<Self::Value, D2::Error> {
        self.visitor.visit_some(DeWrap {
            de: deserializer,
            path: self.path,
            proto: self.proto,
        })
    }

    fn visit_newtype_struct<D2: Deserializer<'de>>(
        self,
        deserializer: D2,
    ) -> core::result::Result<Self::Value, D2::Error> {
        self.visitor.visit_newtype_struct(DeWrap {
            de: deserializer,
            path: self.path,
            proto: self.proto,
        })
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_seq(SeqWrap {
            access,
            path: self.path,
            proto: self.proto,
            index: 0,
        })
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_map(MapWrap {
            access,
            path: self.path,
            proto: self.proto,
            key: None,
        })
    }

    fn visit_enum<A: EnumAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_enum(EnumWrap {
            access,
            path: self.path,
            proto: self.proto,
        })
    }
}

/// SeqAccess wrapper that addresses elements by their decimal index.
struct SeqWrap<'a, A> {
    access: A,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
    index: usize,
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for SeqWrap<'_, A> {
    type Error = A::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> core::result::Result<Option<T::Value>, Self::Error> {
        let link = Path::Seq {
            parent: self.path,
            index: self.index,
        };
        let out = self.access.next_element_seed(SeedWrap {
            seed,
            path: &link,
            proto: self.proto,
        });
        self.index += 1;
        out
    }

    fn size_hint(&self) -> Option<usize> {
        self.access.size_hint()
    }
}

/// MapAccess wrapper that records each key and addresses values by it.
struct MapWrap<'a, A> {
    access: A,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
    key: Option<String>,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for MapWrap<'_, A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> core::result::Result<Option<K::Value>, Self::Error> {
        self.key = None;
        self.access.next_key_seed(KeySeedWrap {
            seed,
            slot: &mut self.key,
        })
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> core::result::Result<V::Value, Self::Error> {
        let key = self.key.take();
        let link = Path::Map {
            parent: self.path,
            key: key.as_deref().unwrap_or("?"),
        };
        self.access.next_value_seed(SeedWrap {
            seed,
            path: &link,
            proto: self.proto,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.access.size_hint()
    }
}

/// Seed wrapper that captures the string form of a map/struct key while
/// forwarding the real deserialization to the wrapped seed.
struct KeySeedWrap<'a, S> {
    seed: S,
    slot: &'a mut Option<String>,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for KeySeedWrap<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> core::result::Result<Self::Value, D::Error> {
        self.seed.deserialize(KeyCaptureDe {
            de: deserializer,
            slot: self.slot,
        })
    }
}

/// Deserializer wrapper that records string-like keys into a slot.
struct KeyCaptureDe<'a, D> {
    de: D,
    slot: &'a mut Option<String>,
}

impl<'de, D> Deserializer<'de> for KeyCaptureDe<'_, D>
where
    D: Deserializer<'de>,
{
    type Error = D::Error;

    fwd_record! {
        deserialize_any;
        deserialize_identifier;
        deserialize_str;
        deserialize_string;
        deserialize_bytes;
        deserialize_byte_buf;
    }

    fwd_plain! {
        deserialize_bool();
        deserialize_i8();
        deserialize_i16();
        deserialize_i32();
        deserialize_i64();
        deserialize_i128();
        deserialize_u8();
        deserialize_u16();
        deserialize_u32();
        deserialize_u64();
        deserialize_u128();
        deserialize_f32();
        deserialize_f64();
        deserialize_char();
        deserialize_option();
        deserialize_unit();
        deserialize_unit_struct(name: &'static str);
        deserialize_newtype_struct(name: &'static str);
        deserialize_seq();
        deserialize_tuple(len: usize);
        deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map();
        deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_ignored_any();
    }

    fn is_human_readable(&self) -> bool {
        self.de.is_human_readable()
    }
}

/// Visitor wrapper that records string-like keys into a slot and forwards
/// everything to the wrapped visitor.
struct RecordVisitor<'a, V> {
    visitor: V,
    slot: &'a mut Option<String>,
}

impl<V> RecordVisitor<'_, V> {
    fn record(&mut self, v: &str) {
        *self.slot = Some(v.to_owned());
    }
}

impl<'de, V> Visitor<'de> for RecordVisitor<'_, V>
where
    V: Visitor<'de>,
{
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.visitor.expecting(formatter)
    }

    fn visit_str<E: de::Error>(mut self, v: &str) -> core::result::Result<Self::Value, E> {
        self.record(v);
        self.visitor.visit_str(v)
    }

    fn visit_borrowed_str<E: de::Error>(
        mut self,
        v: &'de str,
    ) -> core::result::Result<Self::Value, E> {
        self.record(v);
        self.visitor.visit_borrowed_str(v)
    }

    fn visit_string<E: de::Error>(mut self, v: String) -> core::result::Result<Self::Value, E> {
        self.record(&v);
        self.visitor.visit_string(v)
    }

    visit_forward! {
        visit_bool(v: bool);
        visit_i8(v: i8);
        visit_i16(v: i16);
        visit_i32(v: i32);
        visit_i64(v: i64);
        visit_i128(v: i128);
        visit_u8(v: u8);
        visit_u16(v: u16);
        visit_u32(v: u32);
        visit_u64(v: u64);
        visit_u128(v: u128);
        visit_f32(v: f32);
        visit_f64(v: f64);
        visit_char(v: char);
        visit_bytes(v: &[u8]);
        visit_borrowed_bytes(v: &'de [u8]);
        visit_byte_buf(v: Vec<u8>);
        visit_none();
        visit_unit();
    }

    fn visit_some<D2: Deserializer<'de>>(
        self,
        deserializer: D2,
    ) -> core::result::Result<Self::Value, D2::Error> {
        self.visitor.visit_some(deserializer)
    }

    fn visit_newtype_struct<D2: Deserializer<'de>>(
        self,
        deserializer: D2,
    ) -> core::result::Result<Self::Value, D2::Error> {
        self.visitor.visit_newtype_struct(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_seq(access)
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_map(access)
    }

    fn visit_enum<A: EnumAccess<'de>>(
        self,
        access: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        self.visitor.visit_enum(access)
    }
}

/// EnumAccess wrapper. Variant names are not part of the path; the contents
/// of a variant keep the path of the enum value itself.
struct EnumWrap<'a, A> {
    access: A,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

impl<'de, 'a, A> EnumAccess<'de> for EnumWrap<'a, A>
where
    A: EnumAccess<'de>,
{
    type Error = A::Error;
    type Variant = VariantWrap<'a, A::Variant>;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> core::result::Result<(V::Value, Self::Variant), Self::Error> {
        let (value, variant) = tri!(self.access.variant_seed(seed));
        Ok((
            value,
            VariantWrap {
                variant,
                path: self.path,
                proto: self.proto,
            },
        ))
    }
}

/// VariantAccess wrapper; see [`EnumWrap`].
struct VariantWrap<'a, V> {
    variant: V,
    path: &'a Path<'a>,
    proto: &'a [Pattern],
}

impl<'de, V> VariantAccess<'de> for VariantWrap<'_, V>
where
    V: VariantAccess<'de>,
{
    type Error = V::Error;

    fn unit_variant(self) -> core::result::Result<(), Self::Error> {
        self.variant.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> core::result::Result<T::Value, Self::Error> {
        self.variant.newtype_variant_seed(SeedWrap {
            seed,
            path: self.path,
            proto: self.proto,
        })
    }

    fn tuple_variant<V2: Visitor<'de>>(
        self,
        len: usize,
        visitor: V2,
    ) -> core::result::Result<V2::Value, Self::Error> {
        self.variant.tuple_variant(
            len,
            VisWrap {
                visitor,
                path: self.path,
                proto: self.proto,
            },
        )
    }

    fn struct_variant<V2: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V2,
    ) -> core::result::Result<V2::Value, Self::Error> {
        self.variant.struct_variant(
            fields,
            VisWrap {
                visitor,
                path: self.path,
                proto: self.proto,
            },
        )
    }
}

/// Visitor that collects the hex bit-pattern string at a registered path.
struct HexStringVisitor;

impl<'de> Visitor<'de> for HexStringVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string of exactly 16 lowercase hex digits")
    }

    fn visit_str<E: de::Error>(self, v: &str) -> core::result::Result<Self::Value, E> {
        Ok(v.to_owned())
    }

    fn visit_borrowed_str<E: de::Error>(
        self,
        v: &'de str,
    ) -> core::result::Result<Self::Value, E> {
        Ok(v.to_owned())
    }

    fn visit_string<E: de::Error>(self, v: String) -> core::result::Result<Self::Value, E> {
        Ok(v)
    }
}
