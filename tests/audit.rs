//! Acceptance tests for the explicit deserializer borrow audit.
//!
//! Run locally with:
//!
//! ```text
//! cargo test --test audit
//! cargo test --doc audit
//! ```

use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::audit::{BorrowAudit, InputSource, StringKind, StringRole};
use serde_json::Deserializer;
use std::collections::HashMap;
use std::fmt;

// Example string 1: no escapes, so both key and value are borrowed from the
// original slice and the recorded ranges point back into it.
#[test]
fn slice_unescaped_string_is_borrowed_from_original_slice() {
    let json = br#"{"name": "ferris"}"#;
    let mut de = Deserializer::from_slice(json);
    de.enable_borrow_audit();
    let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
    assert_eq!(value["name"], "ferris");

    let audit = de.borrow_audit().unwrap();
    assert_eq!(audit.source(), InputSource::Slice);
    let events = audit.events();
    assert_eq!(events.len(), 2);

    let key = &events[0];
    assert_eq!(key.path, "$.name");
    assert_eq!(key.role, StringRole::Key);
    assert_eq!(key.kind, StringKind::Borrowed);
    assert_eq!(key.source, InputSource::Slice);
    // The range is a genuine source range into the original slice.
    assert_eq!(&json[key.start..key.end], b"\"name\"");
    assert_eq!(&json[key.start + 1..key.end - 1], b"name");

    let val = &events[1];
    assert_eq!(val.path, "$.name");
    assert_eq!(val.role, StringRole::Value);
    assert_eq!(val.kind, StringKind::Borrowed);
    assert_eq!(&json[val.start..val.end], b"\"ferris\"");
    assert_eq!(&json[val.start + 1..val.end - 1], b"ferris");
}

// Example string 2: an escaped string must go through the copy path and must
// never be reported as borrowed.
#[test]
fn slice_escaped_string_is_copied_never_borrowed() {
    let json = br#"{"greeting": "hi\nthere"}"#;
    let mut de = Deserializer::from_slice(json);
    de.enable_borrow_audit();
    let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
    assert_eq!(value["greeting"], "hi\nthere");

    let audit = de.borrow_audit().unwrap();
    let events = audit.events();
    assert_eq!(events.len(), 2);

    // The key has no escape: borrowed.
    assert_eq!(events[0].role, StringRole::Key);
    assert_eq!(events[0].kind, StringKind::Borrowed);

    // The value contains \n: copied, and no event anywhere claims a borrow
    // of the escaped string.
    assert_eq!(events[1].path, "$.greeting");
    assert_eq!(events[1].role, StringRole::Value);
    assert_eq!(events[1].kind, StringKind::Copied);
    assert_eq!(&json[events[1].start..events[1].end], br#""hi\nthere""#);
    assert_eq!(audit.borrowed().count(), 1);
    assert_eq!(audit.copied().count(), 1);
}

// A reader-based deserializer cannot borrow from the input at all.
#[test]
fn reader_never_claims_borrowing() {
    let json = br#"{"name": "ferris", "plain": "noescapes"}"#;
    let mut de = Deserializer::from_reader(&json[..]);
    de.enable_borrow_audit();
    let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
    assert_eq!(value["name"], "ferris");

    let audit = de.borrow_audit().unwrap();
    assert_eq!(audit.source(), InputSource::Reader);
    assert_eq!(audit.events().len(), 4);
    // Even the escape-free strings are copies out of the stream.
    assert_eq!(audit.borrowed().count(), 0);
    assert!(audit
        .events()
        .iter()
        .all(|e| e.kind == StringKind::Copied && e.source == InputSource::Reader));
}

// from_str reports the Str source and still borrows unescaped strings.
#[test]
fn str_source_borrows_like_slice() {
    let json = r#"{"k": "v"}"#;
    let mut de = Deserializer::from_str(json);
    de.enable_borrow_audit();
    let value: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();
    assert_eq!(value["k"], "v");

    let audit = de.borrow_audit().unwrap();
    assert_eq!(audit.source(), InputSource::Str);
    assert!(audit.events().iter().all(|e| e.kind == StringKind::Borrowed));
    assert_eq!(&json[audit.events()[1].start..audit.events()[1].end], "\"v\"");
}

// Nested structs/arrays produce correct nested paths, map keys and values
// are distinguished, and borrowing &str fields still works (lifetimes and
// parse results unchanged).
#[test]
fn nested_paths_and_borrowed_fields() {
    #[derive(serde::Deserialize, Debug)]
    struct User<'a> {
        #[serde(borrow)]
        name: &'a str,
        // Contains an escape, so it cannot be a borrowed &str; the copied
        // string lands in an owned String instead.
        bio: String,
    }

    #[derive(serde::Deserialize, Debug)]
    struct Outer<'a> {
        #[serde(borrow)]
        user: User<'a>,
        #[serde(borrow)]
        tags: Vec<&'a str>,
    }

    let json = r#"{"user": {"name": "ann", "bio": "x\ty"}, "tags": ["t1", "t2"]}"#;
    let mut de = Deserializer::from_str(json);
    de.enable_borrow_audit();
    let outer: Outer = serde::Deserialize::deserialize(&mut de).unwrap();

    // Parse result and lifetimes are unchanged: borrowed fields still point
    // into the input.
    assert_eq!(outer.user.name, "ann");
    assert_eq!(outer.user.bio, "x\ty");
    assert_eq!(outer.tags, ["t1", "t2"]);

    let audit = de.borrow_audit().unwrap();
    let got: Vec<(&str, StringRole, StringKind)> = audit
        .events()
        .iter()
        .map(|e| (e.path.as_str(), e.role, e.kind))
        .collect();
    let expected = [
        ("$.user", StringRole::Key, StringKind::Borrowed),
        ("$.user.name", StringRole::Key, StringKind::Borrowed),
        ("$.user.name", StringRole::Value, StringKind::Borrowed),
        ("$.user.bio", StringRole::Key, StringKind::Borrowed),
        // Escaped string nested inside a struct: copied, not borrowed.
        ("$.user.bio", StringRole::Value, StringKind::Copied),
        ("$.tags", StringRole::Key, StringKind::Borrowed),
        ("$.tags[0]", StringRole::Value, StringKind::Borrowed),
        ("$.tags[1]", StringRole::Value, StringKind::Borrowed),
    ];
    assert_eq!(got, expected);
}

// Map keys and values are recorded on separate roles for a HashMap.
#[test]
fn map_key_and_value_paths_are_distinguished() {
    let json = br#"{"a": "1", "b\n": "2"}"#;
    let mut de = Deserializer::from_slice(json);
    de.enable_borrow_audit();
    let map: HashMap<String, String> = serde::Deserialize::deserialize(&mut de).unwrap();
    assert_eq!(map["a"], "1");
    assert_eq!(map["b\n"], "2");

    let audit = de.borrow_audit().unwrap();
    let events = audit.events();
    assert_eq!(events.len(), 4);
    for pair in events.chunks_exact(2) {
        assert_eq!(pair[0].role, StringRole::Key);
        assert_eq!(pair[1].role, StringRole::Value);
        assert_eq!(pair[0].path, pair[1].path);
    }
    assert_eq!(events[0].kind, StringKind::Borrowed);
    assert_eq!(events[1].kind, StringKind::Borrowed);
    // The escaped key is copied and gets a quoted path segment.
    assert_eq!(events[2].kind, StringKind::Copied);
    assert_eq!(events[2].path, r#"$["b\n"]"#);
    assert_eq!(events[3].kind, StringKind::Borrowed);
}

// A visitor that counts how many times each visit method is called.
#[derive(Default, Debug, PartialEq, Eq)]
struct Counts {
    borrowed_str: usize,
    str_: usize,
    string: usize,
    map: usize,
    seq: usize,
    other: usize,
}

impl Counts {
    fn merge(&mut self, other: Counts) {
        self.borrowed_str += other.borrowed_str;
        self.str_ += other.str_;
        self.string += other.string;
        self.map += other.map;
        self.seq += other.seq;
        self.other += other.other;
    }
}

struct Counter;

impl<'de> Visitor<'de> for Counter {
    type Value = Counts;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("anything")
    }

    fn visit_borrowed_str<E>(self, _v: &'de str) -> Result<Counts, E> {
        Ok(Counts {
            borrowed_str: 1,
            ..Counts::default()
        })
    }

    fn visit_str<E>(self, _v: &str) -> Result<Counts, E> {
        Ok(Counts {
            str_: 1,
            ..Counts::default()
        })
    }

    fn visit_string<E>(self, _v: String) -> Result<Counts, E> {
        Ok(Counts {
            string: 1,
            ..Counts::default()
        })
    }

    fn visit_map<A>(self, mut access: A) -> Result<Counts, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut counts = Counts {
            map: 1,
            ..Counts::default()
        };
        while let Some((key, value)) = access.next_entry_seed(CounterSeed, CounterSeed)? {
            counts.merge(key);
            counts.merge(value);
        }
        Ok(counts)
    }

    fn visit_seq<A>(self, mut access: A) -> Result<Counts, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut counts = Counts {
            seq: 1,
            ..Counts::default()
        };
        while let Some(element) = access.next_element_seed(CounterSeed)? {
            counts.merge(element);
        }
        Ok(counts)
    }

    fn visit_bool<E>(self, _v: bool) -> Result<Counts, E> {
        Ok(Counts {
            other: 1,
            ..Counts::default()
        })
    }

    fn visit_i64<E>(self, _v: i64) -> Result<Counts, E> {
        Ok(Counts {
            other: 1,
            ..Counts::default()
        })
    }

    fn visit_u64<E>(self, _v: u64) -> Result<Counts, E> {
        Ok(Counts {
            other: 1,
            ..Counts::default()
        })
    }

    fn visit_f64<E>(self, _v: f64) -> Result<Counts, E> {
        Ok(Counts {
            other: 1,
            ..Counts::default()
        })
    }

    fn visit_unit<E>(self) -> Result<Counts, E> {
        Ok(Counts {
            other: 1,
            ..Counts::default()
        })
    }
}

struct CounterSeed;

impl<'de> DeserializeSeed<'de> for CounterSeed {
    type Value = Counts;

    fn deserialize<D>(self, deserializer: D) -> Result<Counts, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(Counter)
    }
}

fn count_visits(json: &[u8], audit: bool) -> (Counts, Option<BorrowAudit>) {
    let mut de = Deserializer::from_slice(json);
    if audit {
        de.enable_borrow_audit();
    }
    let counts = serde::Deserializer::deserialize_any(&mut de, Counter).unwrap();
    (counts, de.take_borrow_audit())
}

// Enabling the audit must not increase (or otherwise change) the number of
// visitor calls, and the parse result must be identical.
#[test]
fn visitor_call_count_is_unchanged_by_audit() {
    let json = br#"{"a": "x", "b": ["y", "z\n", 1, true, null], "c": {"d": "w"}}"#;

    let (plain, none_audit) = count_visits(json, false);
    assert!(none_audit.is_none());

    let (audited, Some(audit)) = count_visits(json, true) else {
        panic!("audit should be recorded");
    };
    assert_eq!(plain, audited);

    // Every string visit is accounted for by exactly one audit event.
    let string_visits = audited.borrowed_str + audited.str_ + audited.string;
    assert_eq!(string_visits, audit.events().len());
    assert_eq!(audited.borrowed_str, audit.borrowed().count());
    assert_eq!(audited.str_ + audited.string, audit.copied().count());
}

// A failed deserialization reports an incomplete audit but does not change
// the error, and the deserializer state stays consistent.
#[test]
fn failure_report_is_incomplete_but_error_unchanged() {
    let json = br#"{"a": "x", "b": "#;
    let mut de = Deserializer::from_slice(json);
    de.enable_borrow_audit();
    let result: Result<serde_json::Value, _> = serde::Deserialize::deserialize(&mut de);
    let err = result.unwrap_err();
    assert!(err.is_eof());

    // Only the strings parsed before the failure were recorded.
    let audit = de.borrow_audit().unwrap();
    let paths: Vec<&str> = audit.events().iter().map(|e| e.path.as_str()).collect();
    assert_eq!(paths, ["$.a", "$.a", "$.b"]);

    // Same input without audit produces the same error.
    let mut de2 = Deserializer::from_slice(json);
    let err2: serde_json::Error = serde_json::Value::deserialize(&mut de2).unwrap_err();
    assert_eq!(err.to_string(), err2.to_string());
}

// The human-readable report mentions kind, role, path, range and source.
#[test]
fn report_mentions_all_fields() {
    let json = br#"{"name": "ferris", "greeting": "hi\n"}"#;
    let mut de = Deserializer::from_slice(json);
    de.enable_borrow_audit();
    let _: serde_json::Value = serde::Deserialize::deserialize(&mut de).unwrap();

    let report = de.borrow_audit().unwrap().report();
    assert!(report.contains("input source: slice"));
    assert!(report.contains("borrowed key $.name [1..7] (slice)"));
    assert!(report.contains("borrowed value $.name [9..17] (slice)"));
    assert!(report.contains("copied value $.greeting"));
}
