//! Float field protocol adapter.
//!
//! This module provides a serializer and deserializer that encode `f64` values
//! at *registered paths* as fixed 16-character lowercase hexadecimal bit-pattern
//! strings. All other fields continue to use the normal JSON format.
//!
//! # Protocol
//!
//! A registered path identifies a field by its sequence of Serde field names
//! (after `#[serde(rename)]`). Nested lists are transparent to path matching:
//! descending into a `Vec` does not add a path segment.
//!
//! At a registered path:
//!
//! - **Serialization**: an `f64` is encoded as a JSON string containing exactly
//!   16 lowercase hexadecimal characters representing the IEEE 754 binary64 bit
//!   pattern of the value.
//! - **Deserialization**: a JSON string at a registered path is decoded as the
//!   binary64 value whose bit pattern is given by the 16-character hex string.
//!   Negative zero and NaN payloads are preserved bit-for-bit because no decimal
//!   conversion is involved.
//!
//! At an unregistered path, hex strings are not accepted for `f64` fields (they
//! produce a type mismatch), and ordinary JSON numbers are accepted as usual.
//!
//! # Example
//!
//! ```
//! use serde::{Deserialize, Serialize};
//! use serde_json::float_hex::{self, FloatHexConfig};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Measurement {
//!     value: f64,
//! }
//!
//! let mut config = FloatHexConfig::new();
//! config.register("value");
//!
//! let m = Measurement { value: -0.0 };
//! let json = float_hex::to_string(&m, &config).unwrap();
//! assert_eq!(json, r#"{"value":"8000000000000000"}"#);
//!
//! let back: Measurement = float_hex::from_str(&json, &config).unwrap();
//! assert_eq!(back.value.to_bits(), (-0.0f64).to_bits());
//! ```
//!
//! # Cargo commands
//!
//! Run the float hex protocol tests:
//!
//! ```sh
//! cargo test --test float_hex
//! ```
//!
//! Check the library compiles with the protocol adapter:
//!
//! ```sh
//! cargo check
//! ```
//!
//! # Protocol byte format
//!
//! At a registered path, an `f64` is encoded as a JSON string of exactly 16
//! lowercase hexadecimal characters. The hex string is the big-endian bit
//! representation of the IEEE 754 binary64 value (as returned by
//! [`f64::to_bits`]). For example:
//!
//! | Value        | Bits (hex)          | JSON string              |
//! |--------------|---------------------|--------------------------|
//! | `0.0`        | `0000000000000000`  | `"0000000000000000"`    |
//! | `-0.0`       | `8000000000000000`  | `"8000000000000000"`    |
//! | `1.0`        | `3ff0000000000000`  | `"3ff0000000000000"`    |
//! | `NaN` (payload 1) | `7ff8000000000001` | `"7ff8000000000001"` |
//!
//! Deserialization decodes the hex string directly to bits via
//! [`f64::from_bits`], so negative zero and NaN payloads are preserved exactly
//! without any decimal conversion.

use crate::error::{Error, Result};
use crate::number::Number;
use crate::ser::{CompactFormatter, Formatter};
use crate::value::Value;
use crate::io;
use alloc::borrow::ToOwned;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Display;
use serde::ser::{self, Error as _, Impossible, Serialize};
use serde::Deserialize;

/// Configuration for the float hex protocol.
///
/// Holds the set of registered paths at which `f64` values are encoded as hex
/// bit-pattern strings.
#[derive(Clone, Default, Debug)]
pub struct FloatHexConfig {
    paths: BTreeSet<String>,
}

impl FloatHexConfig {
    /// Creates a new empty configuration with no registered paths.
    pub fn new() -> Self {
        FloatHexConfig {
            paths: BTreeSet::new(),
        }
    }

    /// Registers a path at which `f64` values will be encoded as hex strings.
    ///
    /// The path is a dot-separated sequence of Serde field names (after rename).
    /// Nested lists are transparent to path matching.
    pub fn register(&mut self, path: &str) {
        self.paths.insert(path.to_owned());
    }

    /// Returns `true` if the given path is registered.
    pub fn is_registered(&self, path: &str) -> bool {
        self.paths.contains(path)
    }
}

//////////////////////////////////////////////////////////////////////////////
// Serializer
//////////////////////////////////////////////////////////////////////////////

/// A serializer that encodes `f64` at registered paths as hex bit-pattern
/// strings, while writing all other values as normal JSON.
pub struct FloatHexSerializer<W, F = CompactFormatter> {
    writer: W,
    formatter: F,
    path: Rc<RefCell<Vec<String>>>,
    config: FloatHexConfig,
}

impl<W> FloatHexSerializer<W>
where
    W: io::Write,
{
    /// Creates a new float hex serializer.
    pub fn new(writer: W, config: FloatHexConfig) -> Self {
        FloatHexSerializer::with_formatter(writer, CompactFormatter, config)
    }
}

impl<W, F> FloatHexSerializer<W, F>
where
    W: io::Write,
    F: Formatter,
{
    /// Creates a new float hex serializer with a custom formatter.
    pub fn with_formatter(writer: W, formatter: F, config: FloatHexConfig) -> Self {
        FloatHexSerializer {
            writer,
            formatter,
            path: Rc::new(RefCell::new(Vec::new())),
            config,
        }
    }

    /// Unwrap the `Writer` from the serializer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn current_path(&self) -> String {
        self.path.borrow().join(".")
    }

    fn is_registered(&self) -> bool {
        self.config.is_registered(&self.current_path())
    }
}

/// Tracks whether a compound structure has already written its first element.
#[derive(Eq, PartialEq)]
enum State {
    Empty,
    First,
    Rest,
}

/// Compound serializer for the float hex protocol.
pub struct Compound<'a, W, F> {
    ser: &'a mut FloatHexSerializer<W, F>,
    state: State,
    current_key: Option<String>,
}

impl<'a, W, F> Compound<'a, W, F>
where
    W: io::Write,
    F: Formatter,
{
    fn new(ser: &'a mut FloatHexSerializer<W, F>, state: State) -> Self {
        Compound {
            ser,
            state,
            current_key: None,
        }
    }
}

impl<'a, W, F> ser::Serializer for &'a mut FloatHexSerializer<W, F>
where
    W: io::Write,
    F: Formatter,
{
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Compound<'a, W, F>;
    type SerializeTuple = Compound<'a, W, F>;
    type SerializeTupleStruct = Compound<'a, W, F>;
    type SerializeTupleVariant = Compound<'a, W, F>;
    type SerializeMap = Compound<'a, W, F>;
    type SerializeStruct = Compound<'a, W, F>;
    type SerializeStructVariant = Compound<'a, W, F>;

    #[inline]
    fn serialize_bool(self, value: bool) -> Result<()> {
        self.formatter.write_bool(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<()> {
        self.formatter.write_i8(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_i16(self, value: i16) -> Result<()> {
        self.formatter.write_i16(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_i32(self, value: i32) -> Result<()> {
        self.formatter.write_i32(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_i64(self, value: i64) -> Result<()> {
        self.formatter.write_i64(&mut self.writer, value).map_err(Error::io)
    }

    fn serialize_i128(self, value: i128) -> Result<()> {
        self.formatter.write_i128(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_u8(self, value: u8) -> Result<()> {
        self.formatter.write_u8(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_u16(self, value: u16) -> Result<()> {
        self.formatter.write_u16(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_u32(self, value: u32) -> Result<()> {
        self.formatter.write_u32(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_u64(self, value: u64) -> Result<()> {
        self.formatter.write_u64(&mut self.writer, value).map_err(Error::io)
    }

    fn serialize_u128(self, value: u128) -> Result<()> {
        self.formatter.write_u128(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_f32(self, value: f32) -> Result<()> {
        self.formatter.write_f32(&mut self.writer, value).map_err(Error::io)
    }

    #[inline]
    fn serialize_f64(self, value: f64) -> Result<()> {
        if self.is_registered() {
            let bits = value.to_bits();
            let hex = format!("{:016x}", bits);
            tri!(self.formatter.begin_string(&mut self.writer).map_err(Error::io));
            tri!(self.formatter.write_string_fragment(&mut self.writer, &hex).map_err(Error::io));
            tri!(self.formatter.end_string(&mut self.writer).map_err(Error::io));
            Ok(())
        } else {
            self.formatter.write_f64(&mut self.writer, value).map_err(Error::io)
        }
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<()> {
        let mut buf = [0; 4];
        self.serialize_str(value.encode_utf8(&mut buf))
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<()> {
        tri!(self.formatter.begin_string(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.write_string_fragment(&mut self.writer, value).map_err(Error::io));
        tri!(self.formatter.end_string(&mut self.writer).map_err(Error::io));
        Ok(())
    }

    #[inline]
    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        // Serialize bytes as a sequence of u8 numbers, matching serde_json's
        // default behavior.
        let mut seq = tri!(self.serialize_seq(Some(value.len())));
        for byte in value {
            tri!(ser::SerializeSeq::serialize_element(&mut seq, byte));
        }
        ser::SerializeSeq::end(seq)
    }

    #[inline]
    fn serialize_none(self) -> Result<()> {
        self.formatter.write_null(&mut self.writer).map_err(Error::io)
    }

    #[inline]
    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    #[inline]
    fn serialize_unit(self) -> Result<()> {
        self.formatter.write_null(&mut self.writer).map_err(Error::io)
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.serialize_str(variant)
    }

    #[inline]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    #[inline]
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
        tri!(self.formatter.begin_object(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_key(&mut self.writer, true).map_err(Error::io));
        tri!(self.serialize_str(variant));
        tri!(self.formatter.end_object_key(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_value(&mut self.writer).map_err(Error::io));
        tri!(value.serialize(&mut *self));
        tri!(self.formatter.end_object_value(&mut self.writer).map_err(Error::io));
        self.formatter.end_object(&mut self.writer).map_err(Error::io)
    }

    #[inline]
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        tri!(self.formatter.begin_array(&mut self.writer).map_err(Error::io));
        if len == Some(0) {
            tri!(self.formatter.end_array(&mut self.writer).map_err(Error::io));
            Ok(Compound::new(self, State::Empty))
        } else {
            Ok(Compound::new(self, State::First))
        }
    }

    #[inline]
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(Some(len))
    }

    #[inline]
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(Some(len))
    }

    #[inline]
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        tri!(self.formatter.begin_object(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_key(&mut self.writer, true).map_err(Error::io));
        tri!(self.serialize_str(variant));
        tri!(self.formatter.end_object_key(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_value(&mut self.writer).map_err(Error::io));
        self.serialize_seq(Some(len))
    }

    #[inline]
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        tri!(self.formatter.begin_object(&mut self.writer).map_err(Error::io));
        if len == Some(0) {
            tri!(self.formatter.end_object(&mut self.writer).map_err(Error::io));
            Ok(Compound::new(self, State::Empty))
        } else {
            Ok(Compound::new(self, State::First))
        }
    }

    #[inline]
    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        self.serialize_map(Some(len))
    }

    #[inline]
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        tri!(self.formatter.begin_object(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_key(&mut self.writer, true).map_err(Error::io));
        tri!(self.serialize_str(variant));
        tri!(self.formatter.end_object_key(&mut self.writer).map_err(Error::io));
        tri!(self.formatter.begin_object_value(&mut self.writer).map_err(Error::io));
        self.serialize_map(Some(len))
    }

    fn collect_str<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Display,
    {
        use core::fmt::Write;

        struct Adapter<'ser, W: 'ser, F: 'ser> {
            writer: &'ser mut W,
            formatter: &'ser mut F,
            error: Option<io::Error>,
        }

        impl<'ser, W, F> Write for Adapter<'ser, W, F>
        where
            W: io::Write,
            F: Formatter,
        {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                debug_assert!(self.error.is_none());
                match self.formatter.write_string_fragment(self.writer, s) {
                    Ok(()) => Ok(()),
                    Err(err) => {
                        self.error = Some(err);
                        Err(core::fmt::Error)
                    }
                }
            }
        }

        tri!(self.formatter.begin_string(&mut self.writer).map_err(Error::io));
        let mut adapter = Adapter {
            writer: &mut self.writer,
            formatter: &mut self.formatter,
            error: None,
        };
        match write!(adapter, "{}", value) {
            Ok(()) => debug_assert!(adapter.error.is_none()),
            Err(core::fmt::Error) => {
                return Err(Error::io(adapter.error.expect("there should be an error")));
            }
        }
        self.formatter.end_string(&mut self.writer).map_err(Error::io)
    }
}

//////////////////////////////////////////////////////////////////////////////
// Compound implementations
//////////////////////////////////////////////////////////////////////////////

impl<'a, W, F> ser::SerializeSeq for Compound<'a, W, F>
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
        tri!(self
            .ser
            .formatter
            .begin_array_value(&mut self.ser.writer, self.state == State::First)
            .map_err(Error::io));
        self.state = State::Rest;
        // Lists are transparent to path matching: do not push a path segment.
        tri!(value.serialize(&mut *self.ser));
        self.ser
            .formatter
            .end_array_value(&mut self.ser.writer)
            .map_err(Error::io)
    }

    fn end(self) -> Result<()> {
        match self.state {
            State::Empty => Ok(()),
            _ => {
                tri!(self
                    .ser
                    .formatter
                    .end_array(&mut self.ser.writer)
                    .map_err(Error::io));
                Ok(())
            }
        }
    }
}

impl<'a, W, F> ser::SerializeTuple for Compound<'a, W, F>
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
        ser::SerializeSeq::end(self)
    }
}

impl<'a, W, F> ser::SerializeTupleStruct for Compound<'a, W, F>
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
        ser::SerializeSeq::end(self)
    }
}

impl<'a, W, F> ser::SerializeTupleVariant for Compound<'a, W, F>
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
        match self.state {
            State::Empty => Ok(()),
            _ => {
                tri!(self
                    .ser
                    .formatter
                    .end_array(&mut self.ser.writer)
                    .map_err(Error::io));
                Ok(())
            }
        }
    }
}

impl<'a, W, F> ser::SerializeMap for Compound<'a, W, F>
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
        tri!(self
            .ser
            .formatter
            .begin_object_key(&mut self.ser.writer, self.state == State::First)
            .map_err(Error::io));
        self.state = State::Rest;

        // Serialize the key as a string to capture the field name for path tracking.
        let mut key_str = String::new();
        let key_ser = KeySerializer {
            writer: &mut key_str,
        };
        tri!(key.serialize(key_ser).map_err(Error::custom));
        self.current_key = Some(key_str.clone());

        tri!(self
            .ser
            .formatter
            .begin_string(&mut self.ser.writer)
            .map_err(Error::io));
        tri!(self
            .ser
            .formatter
            .write_string_fragment(&mut self.ser.writer, &key_str)
            .map_err(Error::io));
        tri!(self
            .ser
            .formatter
            .end_string(&mut self.ser.writer)
            .map_err(Error::io));
        self.ser
            .formatter
            .end_object_key(&mut self.ser.writer)
            .map_err(Error::io)
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        tri!(self
            .ser
            .formatter
            .begin_object_value(&mut self.ser.writer)
            .map_err(Error::io));

        // Push the map key onto the path before serializing the value.
        if let Some(ref key) = self.current_key {
            self.ser.path.borrow_mut().push(key.clone());
        }
        let result = value.serialize(&mut *self.ser);
        if self.current_key.is_some() {
            self.ser.path.borrow_mut().pop();
        }
        tri!(result);

        self.ser
            .formatter
            .end_object_value(&mut self.ser.writer)
            .map_err(Error::io)
    }

    fn end(self) -> Result<()> {
        match self.state {
            State::Empty => Ok(()),
            _ => {
                tri!(self
                    .ser
                    .formatter
                    .end_object(&mut self.ser.writer)
                    .map_err(Error::io));
                Ok(())
            }
        }
    }
}

impl<'a, W, F> ser::SerializeStruct for Compound<'a, W, F>
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
        // Push the field name onto the path.
        self.ser.path.borrow_mut().push(key.to_string());

        tri!(self
            .ser
            .formatter
            .begin_object_key(&mut self.ser.writer, self.state == State::First)
            .map_err(Error::io));
        self.state = State::Rest;
        tri!(self
            .ser
            .formatter
            .begin_string(&mut self.ser.writer)
            .map_err(Error::io));
        tri!(self
            .ser
            .formatter
            .write_string_fragment(&mut self.ser.writer, key)
            .map_err(Error::io));
        tri!(self
            .ser
            .formatter
            .end_string(&mut self.ser.writer)
            .map_err(Error::io));
        tri!(self
            .ser
            .formatter
            .end_object_key(&mut self.ser.writer)
            .map_err(Error::io));

        tri!(self
            .ser
            .formatter
            .begin_object_value(&mut self.ser.writer)
            .map_err(Error::io));
        let result = value.serialize(&mut *self.ser);
        self.ser.path.borrow_mut().pop();
        tri!(result);
        self.ser
            .formatter
            .end_object_value(&mut self.ser.writer)
            .map_err(Error::io)
    }

    fn end(self) -> Result<()> {
        match self.state {
            State::Empty => Ok(()),
            _ => {
                tri!(self
                    .ser
                    .formatter
                    .end_object(&mut self.ser.writer)
                    .map_err(Error::io));
                Ok(())
            }
        }
    }
}

impl<'a, W, F> ser::SerializeStructVariant for Compound<'a, W, F>
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
        match self.state {
            State::Empty => Ok(()),
            _ => {
                tri!(self
                    .ser
                    .formatter
                    .end_object(&mut self.ser.writer)
                    .map_err(Error::io));
                Ok(())
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Key serializer (for map keys)
//////////////////////////////////////////////////////////////////////////////

struct KeySerializer<'a> {
    writer: &'a mut String,
}

impl<'a> ser::Serializer for KeySerializer<'a> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Impossible<(), Error>;
    type SerializeTuple = Impossible<(), Error>;
    type SerializeTupleStruct = Impossible<(), Error>;
    type SerializeTupleVariant = Impossible<(), Error>;
    type SerializeMap = Impossible<(), Error>;
    type SerializeStruct = Impossible<(), Error>;
    type SerializeStructVariant = Impossible<(), Error>;

    fn serialize_bool(self, _value: bool) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_i8(self, _value: i8) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_i16(self, _value: i16) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_i32(self, _value: i32) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_i64(self, _value: i64) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_u8(self, _value: u8) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_u16(self, _value: u16) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_u32(self, _value: u32) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_u64(self, _value: u64) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_f32(self, _value: f32) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_f64(self, _value: f64) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_char(self, value: char) -> Result<()> {
        let mut buf = [0; 4];
        self.serialize_str(value.encode_utf8(&mut buf))
    }

    fn serialize_str(self, value: &str) -> Result<()> {
        self.writer.push_str(value);
        Ok(())
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_none(self) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_some<T>(self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_unit(self) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<()> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        Err(Error::custom("map key must be a string"))
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
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(Error::custom("map key must be a string"))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(Error::custom("map key must be a string"))
    }
}

//////////////////////////////////////////////////////////////////////////////
// Serialization functions
//////////////////////////////////////////////////////////////////////////////

/// Serialize a value to a JSON string using the float hex protocol.
pub fn to_string<T>(value: &T, config: &FloatHexConfig) -> Result<String>
where
    T: ?Sized + Serialize,
{
    let mut writer = Vec::new();
    tri!(to_writer(value, &mut writer, config));
    Ok(String::from_utf8(writer).expect("invalid UTF-8"))
}

/// Serialize a value to a JSON byte vector using the float hex protocol.
pub fn to_vec<T>(value: &T, config: &FloatHexConfig) -> Result<Vec<u8>>
where
    T: ?Sized + Serialize,
{
    let mut writer = Vec::new();
    tri!(to_writer(value, &mut writer, config));
    Ok(writer)
}

/// Serialize a value to a writer using the float hex protocol.
pub fn to_writer<T, W>(value: &T, writer: W, config: &FloatHexConfig) -> Result<()>
where
    T: ?Sized + Serialize,
    W: io::Write,
{
    let mut ser = FloatHexSerializer::new(writer, config.clone());
    value.serialize(&mut ser)
}

//////////////////////////////////////////////////////////////////////////////
// Deserialization
//////////////////////////////////////////////////////////////////////////////

/// Deserialize an instance of type  from a JSON string using the float hex
/// protocol.
///
/// At registered paths, JSON strings are decoded as binary64 bit patterns.
/// At unregistered paths, hex strings are rejected for  fields.
pub fn from_str<'de, T>(s: &'de str, config: &FloatHexConfig) -> Result<T>
where
    T: Deserialize<'de>,
{
    from_slice(s.as_bytes(), config)
}

/// Deserialize an instance of type  from a JSON byte slice using the float
/// hex protocol.
pub fn from_slice<'de, T>(s: &'de [u8], config: &FloatHexConfig) -> Result<T>
where
    T: Deserialize<'de>,
{
    let mut value = tri!(crate::from_slice::<Value>(s));
    tri!(convert_value(&mut value, &mut Vec::new(), config));
    T::deserialize(value)
}

/// Recursively walk a JSON value, converting hex strings at registered paths to
/// `f64` values.
fn convert_value(
    value: &mut Value,
    path: &mut Vec<String>,
    config: &FloatHexConfig,
) -> Result<()> {
    match value {
        Value::Object(map) => {
            // Collect keys to avoid borrowing issues.
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                path.push(key.clone());
                if let Some(v) = map.get_mut(&key) {
                    tri!(convert_value(v, path, config));
                }
                path.pop();
            }
        }
        Value::Array(arr) => {
            // Lists are transparent to path matching: do not push a segment.
            for v in arr.iter_mut() {
                tri!(convert_value(v, path, config));
            }
        }
        Value::String(s) => {
            let current_path = path.join(".");
            if config.is_registered(&current_path) {
                // Decode the 16-character hex string as an f64 bit pattern.
                match decode_hex_f64(s) {
                    Some(bits) => {
                        *value = Value::Number(Number::from_f64_bits(f64::from_bits(bits)));
                    }
                    None => {
                        return Err(Error::custom(format!(
                            "invalid float hex protocol at path `{}`: expected 16 lowercase hex characters, got `{}`",
                            current_path, s
                        )));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Decode a 16-character lowercase hex string as an f64 bit pattern.
///
/// Returns `None` if the string is not exactly 16 lowercase hex characters.
fn decode_hex_f64(s: &str) -> Option<u64> {
    if s.len() != 16 {
        return None;
    }
    if !s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return None;
    }
    u64::from_str_radix(s, 16).ok()
}
