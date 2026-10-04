//! Acceptance tests for the [`Locator`](serde_json::locator::Locator) adapter.
//!
//! These tests verify that spans are recorded during the same serialization
//! pass, reflect the real output (including escaping and container brackets),
//! and that failures produce a partial index marked incomplete.

use serde::ser::SerializeStruct;
use serde::Serialize;
use serde_json::locator::Locator;
use serde_json::Value;

/// Converts a byte slice to a string slice for comparison.
fn s(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).unwrap()
}

// ---------------------------------------------------------------------------
// The core acceptance test: {"a/b":["猫",2]}
// ---------------------------------------------------------------------------

#[test]
fn core_acceptance() {
    #[derive(Serialize)]
    struct Doc {
        #[serde(rename = "a/b")]
        a_b: (String, u32),
    }

    let doc = Doc {
        a_b: ("猫".to_owned(), 2),
    };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert_eq!(s(&out), r#"{"a/b":["猫",2]}"#);
    assert!(index.is_complete());

    // The root span covers the entire document.
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), r#"{"a/b":["猫",2]}"#);

    // The parent array span is accurate.
    let parent = index.get("/a~1b").unwrap();
    assert_eq!(s(&out[parent.start..parent.end]), r#"["猫",2]"#);

    // The first element is the complete JSON string, quotes included.
    let first = index.get("/a~1b/0").unwrap();
    assert_eq!(s(&out[first.start..first.end]), r#""猫""#);

    // The second element is the number.
    let second = index.get("/a~1b/1").unwrap();
    assert_eq!(s(&out[second.start..second.end]), "2");
}

// ---------------------------------------------------------------------------
// Skipped fields must not appear in the index.
// ---------------------------------------------------------------------------

#[test]
fn skipped_fields_absent() {
    #[derive(Serialize)]
    struct Doc {
        present: u32,
        #[serde(skip)]
        #[allow(dead_code)]
        skipped: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        maybe: Option<u32>,
    }

    let doc = Doc {
        present: 1,
        skipped: 2,
        maybe: None,
    };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert_eq!(s(&out), r#"{"present":1}"#);
    assert!(index.is_complete());

    // Only the root and the present field are indexed.
    assert!(index.get("").is_some());
    assert!(index.get("/present").is_some());
    assert!(index.get("/skipped").is_none());
    assert!(index.get("/maybe").is_none());
}

// ---------------------------------------------------------------------------
// A custom Serialize impl that changes field names follows the real output.
// ---------------------------------------------------------------------------

#[test]
fn custom_serialize_renames() {
    struct Custom(u32, u32);

    impl Serialize for Custom {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut state = serializer.serialize_struct("Custom", 2)?;
            state.serialize_field("renamed", &self.0)?;
            state.serialize_field("other", &self.1)?;
            state.end()
        }
    }

    let doc = Custom(10, 20);
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert_eq!(s(&out), r#"{"renamed":10,"other":20}"#);
    assert!(index.is_complete());

    assert!(index.get("/renamed").is_some());
    assert!(index.get("/other").is_some());
    // The Rust field names (0, 1) are not used; the custom names are.
    assert!(index.get("/0").is_none());
    assert!(index.get("/1").is_none());

    let renamed = index.get("/renamed").unwrap();
    assert_eq!(s(&out[renamed.start..renamed.end]), "10");
}

// ---------------------------------------------------------------------------
// flatten follows the real output.
// ---------------------------------------------------------------------------

#[test]
fn flatten_follows_output() {
    #[derive(Serialize)]
    struct Inner {
        x: u32,
        y: u32,
    }

    #[derive(Serialize)]
    struct Outer {
        a: u32,
        #[serde(flatten)]
        inner: Inner,
        b: u32,
    }

    let doc = Outer {
        a: 1,
        inner: Inner { x: 2, y: 3 },
        b: 4,
    };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert_eq!(s(&out), r#"{"a":1,"x":2,"y":3,"b":4}"#);
    assert!(index.is_complete());

    assert!(index.get("/a").is_some());
    assert!(index.get("/x").is_some());
    assert!(index.get("/y").is_some());
    assert!(index.get("/b").is_some());
}

// ---------------------------------------------------------------------------
// Pretty indentation: spans still point at the exact value bytes.
// ---------------------------------------------------------------------------

#[test]
fn pretty_spans_are_exact() {
    #[derive(Serialize)]
    struct Doc {
        #[serde(rename = "a/b")]
        a_b: (String, u32),
    }

    let doc = Doc {
        a_b: ("猫".to_owned(), 2),
    };
    let mut locator = Locator::pretty(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    let expected = "{\n  \"a/b\": [\n    \"猫\",\n    2\n  ]\n}";
    assert_eq!(s(&out), expected);
    assert!(index.is_complete());

    // The string span is still just the string, no indentation.
    let first = index.get("/a~1b/0").unwrap();
    assert_eq!(s(&out[first.start..first.end]), r#""猫""#);

    // The array span includes its brackets and the pretty whitespace.
    let parent = index.get("/a~1b").unwrap();
    assert_eq!(s(&out[parent.start..parent.end]), "[\n    \"猫\",\n    2\n  ]");

    // The root span is the entire pretty document.
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), expected);
}

// ---------------------------------------------------------------------------
// Streaming writer failure: partial index, marked incomplete.
// ---------------------------------------------------------------------------

#[test]
fn writer_failure_partial_index() {
    use std::io;

    struct FailingWriter {
        remaining: usize,
    }

    impl io::Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::new(io::ErrorKind::Other, "writer failed"));
            }
            let n = buf.len().min(self.remaining);
            self.remaining -= n;
            Ok(n)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[derive(Serialize)]
    struct Broken {
        ok: u32,
        bad: u32,
    }

    // Allow exactly {"ok":1 (7 bytes); the comma before "bad" fails.
    let writer = FailingWriter { remaining: 7 };
    let mut locator = Locator::new(writer);
    let result = Broken { ok: 1, bad: 2 }.serialize(&mut locator);

    assert!(result.is_err());
    assert!(!locator.is_complete());

    let index = locator.index();
    // The field written before the failure is still recorded.
    let ok = index.get("/ok").unwrap();
    assert_eq!(ok.len(), 1);

    // The writer only accepted 7 bytes.
    let (writer, _) = locator.into_parts();
    assert_eq!(writer.remaining, 0);
}

// ---------------------------------------------------------------------------
// Enum variants.
// ---------------------------------------------------------------------------

#[test]
fn enum_variants() {
    #[derive(Serialize)]
    enum E {
        Newtype(u32),
        Tuple(u32, u32),
        Struct { a: u32, b: u32 },
    }

    // Newtype variant: {"Newtype":5}
    let mut locator = Locator::new(Vec::new());
    E::Newtype(5).serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#"{"Newtype":5}"#);
    assert!(index.get("/Newtype").is_some());
    let nt = index.get("/Newtype").unwrap();
    assert_eq!(s(&out[nt.start..nt.end]), "5");

    // Tuple variant: {"Tuple":[1,2]}
    let mut locator = Locator::new(Vec::new());
    E::Tuple(1, 2).serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#"{"Tuple":[1,2]}"#);
    assert!(index.get("/Tuple").is_some());
    assert!(index.get("/Tuple/0").is_some());
    assert!(index.get("/Tuple/1").is_some());
    let t = index.get("/Tuple").unwrap();
    assert_eq!(s(&out[t.start..t.end]), "[1,2]");

    // Struct variant: {"Struct":{"a":1,"b":2}}
    let mut locator = Locator::new(Vec::new());
    E::Struct { a: 1, b: 2 }.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#"{"Struct":{"a":1,"b":2}}"#);
    assert!(index.get("/Struct").is_some());
    assert!(index.get("/Struct/a").is_some());
    assert!(index.get("/Struct/b").is_some());
    let st = index.get("/Struct").unwrap();
    assert_eq!(s(&out[st.start..st.end]), r#"{"a":1,"b":2}"#);
}

// ---------------------------------------------------------------------------
// Root scalars and nested objects.
// ---------------------------------------------------------------------------

#[test]
fn root_scalars() {
    let mut locator = Locator::new(Vec::new());
    42u32.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), "42");
    assert!(index.get("").is_some());
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), "42");

    let mut locator = Locator::new(Vec::new());
    "hello".serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#""hello""#);
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), r#""hello""#);
}

#[test]
fn nested_objects() {
    #[derive(Serialize)]
    struct Inner {
        z: u32,
    }

    #[derive(Serialize)]
    struct Outer {
        a: Inner,
    }

    let doc = Outer {
        a: Inner { z: 9 },
    };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#"{"a":{"z":9}}"#);

    assert!(index.get("").is_some());
    assert!(index.get("/a").is_some());
    assert!(index.get("/a/z").is_some());

    let a = index.get("/a").unwrap();
    assert_eq!(s(&out[a.start..a.end]), r#"{"z":9}"#);
}

// ---------------------------------------------------------------------------
// Round-trip: every extracted span parses as valid JSON.
// ---------------------------------------------------------------------------

#[test]
fn every_span_is_valid_json() {
    #[derive(Serialize)]
    struct Doc {
        #[serde(rename = "a/b")]
        a_b: (String, u32),
        nested: Nested,
    }

    #[derive(Serialize)]
    struct Nested {
        x: Vec<bool>,
    }

    let doc = Doc {
        a_b: ("猫".to_owned(), 2),
        nested: Nested {
            x: vec![true, false, true],
        },
    };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    for (pointer, span) in index.entries() {
        let extracted = &out[span.start..span.end];
        let parsed: Value = serde_json::from_slice(extracted).unwrap_or_else(|e| {
            panic!(
                "span for {pointer:?} is not valid JSON: {e}\n  bytes: {extracted:?}"
            )
        });
        // The root span must equal the full output.
        if pointer.is_empty() {
            assert_eq!(extracted, out);
        }
        let _ = parsed;
    }
}

// ---------------------------------------------------------------------------
// serde_json::Value round-trip.
// ---------------------------------------------------------------------------

#[test]
fn value_round_trip() {
    let v = serde_json::json!({
        "a": [1, 2, 3],
        "b": {"c": "hello"}
    });
    let mut locator = Locator::new(Vec::new());
    v.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert!(index.is_complete());
    assert!(index.get("").is_some());
    assert!(index.get("/a").is_some());
    assert!(index.get("/a/0").is_some());
    assert!(index.get("/a/1").is_some());
    assert!(index.get("/a/2").is_some());
    assert!(index.get("/b").is_some());
    assert!(index.get("/b/c").is_some());

    let c = index.get("/b/c").unwrap();
    assert_eq!(s(&out[c.start..c.end]), r#""hello""#);
}

// ---------------------------------------------------------------------------
// Escaped keys with ~ and /.
// ---------------------------------------------------------------------------

#[test]
fn escaped_keys() {
    let mut map = std::collections::BTreeMap::new();
    map.insert("a~b".to_owned(), 1u32);
    map.insert("c/d".to_owned(), 2u32);
    let mut locator = Locator::new(Vec::new());
    map.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();

    assert!(index.get("/a~0b").is_some());
    assert!(index.get("/c~1d").is_some());

    let a = index.get("/a~0b").unwrap();
    assert_eq!(s(&out[a.start..a.end]), "1");
}

// ---------------------------------------------------------------------------
// Index entries are in serialization order.
// ---------------------------------------------------------------------------

#[test]
fn entries_in_order() {
    #[derive(Serialize)]
    struct Doc {
        a: u32,
        b: u32,
        c: u32,
    }

    let doc = Doc { a: 1, b: 2, c: 3 };
    let mut locator = Locator::new(Vec::new());
    doc.serialize(&mut locator).unwrap();
    let (_, index) = locator.into_parts();

    let pointers: Vec<&str> = index.iter().map(|(p, _)| p).collect();
    // Entries are recorded in completion order: leaf fields first, then the
    // root container once its closing bracket has been written.
    assert_eq!(pointers, vec!["/a", "/b", "/c", ""]);
}

// ---------------------------------------------------------------------------
// Feature-gated: arbitrary_precision Number.
// ---------------------------------------------------------------------------

#[cfg(feature = "arbitrary_precision")]
#[test]
fn arbitrary_precision_number() {
    use serde_json::Number;
    let n: Number = "123456789012345678901234567890".parse().unwrap();
    let mut locator = Locator::new(Vec::new());
    n.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), "123456789012345678901234567890");
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), "123456789012345678901234567890");
}

// ---------------------------------------------------------------------------
// Feature-gated: raw_value RawValue.
// ---------------------------------------------------------------------------

#[cfg(feature = "raw_value")]
#[test]
fn raw_value() {
    use serde_json::value::RawValue;
    let raw: &RawValue = serde_json::from_str(r#"{"a":1}"#).unwrap();
    let mut locator = Locator::new(Vec::new());
    raw.serialize(&mut locator).unwrap();
    let (out, index) = locator.into_parts();
    assert_eq!(s(&out), r#"{"a":1}"#);
    let root = index.get("").unwrap();
    assert_eq!(s(&out[root.start..root.end]), r#"{"a":1}"#);
}
