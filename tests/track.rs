use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::json;
use serde_json::track::{
    to_string_pretty_tracked, to_string_tracked, to_vec_tracked, to_writer_tracked,
    PointerSerializer, TrackedError,
};
use std::io;

fn slice_str<'a>(idx: &serde_json::track::PointerIndex, ptr: &str, out: &'a [u8]) -> &'a str {
    std::str::from_utf8(idx.slice(ptr, out).expect("pointer present")).unwrap()
}

#[test]
fn acceptance_pointer_and_ranges() {
    // {"a/b":["猫",2]}
    let value = json!({"a/b": ["猫", 2]});
    let (bytes, index) = to_vec_tracked(&value).unwrap();
    assert!(index.complete());
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), r#"{"a/b":["猫",2]}"#);

    // /a~1b/0 covers the complete JSON string including quotes and escapes.
    assert_eq!(slice_str(&index, "/a~1b/0", &bytes), "\"\u{732b}\"");
    // /a~1b/1 covers the number.
    assert_eq!(slice_str(&index, "/a~1b/1", &bytes), "2");
    // Parent array spans '['..']'.
    assert_eq!(slice_str(&index, "/a~1b", &bytes), r#"["猫",2]"#);
    // Root object spans '{'..'}'.
    assert_eq!(slice_str(&index, "", &bytes), r#"{"a/b":["猫",2]}"#);

    // Every slice is standalone valid JSON and re-parses to the same node.
    for entry in index.entries() {
        let raw = entry.get(&bytes).unwrap();
        let reparsed: serde_json::Value = serde_json::from_slice(raw).unwrap();
        let expected = value
            .pointer(&entry.pointer)
            .expect("json-pointer lookup matches");
        assert_eq!(&reparsed, expected, "pointer {}", entry.pointer);
    }

    // Keys are not indexed as standalone values.
    assert!(index.range("/a~1b/猫").is_none());
}

#[derive(serde::Serialize)]
#[allow(dead_code)]
struct HasSkipped {
    kept: i32,
    #[serde(skip_serializing)]
    skipped: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    maybe: Option<u8>,
    present: Option<u8>,
}

#[test]
fn skipped_fields_absent_from_index() {
    let v = HasSkipped {
        kept: 1,
        skipped: 2,
        maybe: None,
        present: Some(3),
    };
    let (bytes, index) = to_vec_tracked(&v).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_eq!(text, r#"{"kept":1,"present":3}"#);

    let pointers: Vec<&str> = index.entries().iter().map(|e| e.pointer.as_str()).collect();
    assert!(pointers.contains(&"/kept"));
    assert!(pointers.contains(&"/present"));
    assert!(!pointers.iter().any(|p| p.contains("skipped")));
    assert!(!pointers.iter().any(|p| p.contains("maybe")));

    assert_eq!(slice_str(&index, "/kept", &bytes), "1");
    assert_eq!(slice_str(&index, "/present", &bytes), "3");
}

// Custom Serialize that renames a field and synthesizes another.
struct Renamed;

impl Serialize for Renamed {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("output-name", &7)?; // Rust field had another name
        map.serialize_entry("nested", &json!({"k": ["x", "y"]}))?;
        map.end()
    }
}

#[test]
fn custom_serialize_rename_follows_output() {
    let (bytes, index) = to_vec_tracked(&Renamed).unwrap();
    assert_eq!(
        std::str::from_utf8(&bytes).unwrap(),
        r#"{"output-name":7,"nested":{"k":["x","y"]}}"#
    );
    assert_eq!(slice_str(&index, "/output-name", &bytes), "7");
    assert_eq!(slice_str(&index, "/nested/k/1", &bytes), r#""y""#);
    assert!(index.range("/original-name").is_none());
}

#[derive(serde::Serialize)]
struct FlatOuter {
    id: u8,
    #[serde(flatten)]
    inner: FlatInner,
}

#[derive(serde::Serialize)]
struct FlatInner {
    a: u8,
    b: String,
}

#[test]
fn flatten_follows_real_output() {
    let v = FlatOuter {
        id: 0,
        inner: FlatInner {
            a: 1,
            b: "a/b~c".into(),
        },
    };
    let (bytes, index) = to_vec_tracked(&v).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_eq!(text, r#"{"id":0,"a":1,"b":"a/b~c"}"#);
    assert_eq!(slice_str(&index, "/id", &bytes), "0");
    assert_eq!(slice_str(&index, "/a", &bytes), "1");
    assert_eq!(slice_str(&index, "/b", &bytes), r#""a/b~c""#);
}

#[test]
fn pretty_print_indent_ranges() {
    let value = json!({"a/b": ["猫", 2]});
    let (string, index) = to_string_pretty_tracked(&value).unwrap();
    assert_eq!(string, "{\n  \"a/b\": [\n    \"\u{732b}\",\n    2]}");
    // No leading newline/indent, no trailing comma: standalone JSON.
    assert_eq!(
        std::str::from_utf8(index.slice("/a~1b/0", string.as_bytes()).unwrap()).unwrap(),
        "\"\u{732b}\""
    );
    assert_eq!(
        std::str::from_utf8(index.slice("/a~1b", string.as_bytes()).unwrap()).unwrap(),
        "[\n    \"\u{732b}\",\n    2]"
    );
    assert_eq!(
        std::str::from_utf8(index.slice("", string.as_bytes()).unwrap()).unwrap(),
        string
    );
}

#[test]
fn string_escapes_covered_in_range() {
    let value = json!({"quote\"\nl": "val/ue"});
    let (bytes, index) = to_vec_tracked(&value).unwrap();
    // Key containing quote/backslash/newline: pointer built from decoded key,
    // range covers the fully escaped JSON string.
    let raw_key = "quote\"\nl";
    let token = raw_key.replace('~', "~0").replace('/', "~1");
    let pointer = format!("/{}", token);
    assert!(index.range(&pointer).is_some(), "entry for {pointer:?}");
    assert_eq!(
        std::str::from_utf8(index.slice(&pointer, &bytes).unwrap()).unwrap(),
        "\"val/ue\""
    );
}

struct FailWriter {
    allow: usize,
}

impl io::Write for FailWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.allow == 0 {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "boom"));
        }
        let n = self.allow.min(buf.len());
        self.allow -= n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn streaming_writer_failure_marks_incomplete() {
    let value = json!({"a": [1, 2, 3], "b": {"c": "longish-value-here"}});
    let mut writer = FailWriter { allow: 8 };
    let err = to_writer_tracked(&mut writer, &value).unwrap_err();
    let TrackedError { error, index } = err;
    assert!(!index.complete(), "must not claim complete index");
    assert!(index.entries().iter().all(|e| e.start <= e.end));
    // Every recorded entry is a *finished* value: none of the ranges may
    // extend past the bytes the writer accepted.
    for e in index.entries() {
        assert!(e.end <= 8, "entry {} ends beyond accepted bytes", e.pointer);
    }
    // Original I/O error preserved.
    assert_eq!(error.io_error_kind(), Some(io::ErrorKind::BrokenPipe));
}

struct AlwaysFails;

impl Serialize for AlwaysFails {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Write one finished field, then fail mid-object.
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("before", &1)?;
        Err(serde::ser::Error::custom("intentional failure"))
    }
}

#[test]
fn short_writes_count_partial_bytes() {
    // Writer that accepts only a few bytes per write call and then errors:
    // ranges recorded before the error must match the bytes actually landed.
    struct SlowFail {
        accept: usize,
        seen: Vec<u8>,
    }
    impl io::Write for SlowFail {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.accept == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "done"));
            }
            let n = self.accept.min(buf.len()).min(1);
            self.accept -= n;
            self.seen.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut w = SlowFail {
        accept: 12,
        seen: Vec::new(),
    };
    let err = to_writer_tracked(&mut w, &json!({"abc": [10, 20]})).unwrap_err();
    assert!(!err.index.complete());
    for e in err.index.entries() {
        let slice = e.get(&w.seen).unwrap();
        assert_eq!(slice, &w.seen[e.start..e.end]);
    }
}

#[test]
fn serialize_error_is_incomplete() {
    let res = to_string_tracked(&AlwaysFails);
    let err = res.unwrap_err();
    assert!(!err.index.complete());
    // The field written before the failure is a finished, valid fragment.
    assert!(err.index.range("/before").is_some());
}

#[test]
fn integer_map_keys_and_bytes() {
    use std::collections::BTreeMap;
    let mut map = BTreeMap::new();
    map.insert(10u32, "x");
    map.insert(20, "y");
    let (bytes, index) = to_vec_tracked(&map).unwrap();
    assert_eq!(
        std::str::from_utf8(&bytes).unwrap(),
        r#"{"10":"x","20":"y"}"#
    );
    assert_eq!(slice_str(&index, "/10", &bytes), r#""x""#);
    assert_eq!(slice_str(&index, "/20", &bytes), r#""y""#);

    // &[u8] as byte array: children and container all tracked.
    let (bytes, index) = to_vec_tracked(&serde_bytes(&[1u8, 2, 3])).unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), "[1,2,3]");
    assert_eq!(slice_str(&index, "/0", &bytes), "1");
    assert_eq!(slice_str(&index, "/2", &bytes), "3");
    assert_eq!(slice_str(&index, "", &bytes), "[1,2,3]");
}

fn serde_bytes(slice: &[u8]) -> impl serde::Serialize + '_ {
    // Wrapper implementing serialize_bytes via serde's bytes helper.
    struct B<'a>(&'a [u8]);
    impl<'a> serde::Serialize for B<'a> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_bytes(self.0)
        }
    }
    B(slice)
}

#[test]
fn root_scalar_ranges() {
    let (bytes, index) = to_vec_tracked(&42).unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), "42");
    assert_eq!(slice_str(&index, "", &bytes), "42");

    let (bytes, index) = to_vec_tracked(&"hi").unwrap();
    assert_eq!(slice_str(&index, "", &bytes), r#""hi""#);
}

#[test]
fn empty_containers() {
    let v = json!({"a": [], "b": {}});
    let (bytes, index) = to_vec_tracked(&v).unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), r#"{"a":[],"b":{}}"#);
    assert_eq!(slice_str(&index, "/a", &bytes), "[]");
    assert_eq!(slice_str(&index, "/b", &bytes), "{}");
}

#[test]
fn adapter_reuse_and_into_inner() {
    let mut out: Vec<u8> = Vec::new();
    let mut ser = PointerSerializer::new(&mut out);
    ser.serialize(&json!({"x": 1})).unwrap();
    assert!(ser.index().complete());
    ser.serialize(&json!({"y": 2})).unwrap();
    // Second serialization into the same byte stream: offsets continue.
    let idx = ser.index();
    assert_eq!(std::str::from_utf8(&out).unwrap(), r#"{"x":1}{"y":2}"#);
    assert_eq!(
        std::str::from_utf8(idx.slice("/y", &out).unwrap()).unwrap(),
        "2"
    );
}

#[test]
fn doc_example_works() {
    #[derive(serde::Serialize)]
    struct Data<'a> {
        #[serde(rename = "a/b")]
        ab: Vec<serde_json::Value>,
        note: &'a str,
    }

    let data = Data {
        ab: vec![serde_json::Value::String("猫".into()), 2.into()],
        note: "x~y/z",
    };
    let mut out = Vec::new();
    let mut ser = PointerSerializer::new(&mut out);
    ser.serialize(&data).unwrap();
    let index = ser.index();
    assert!(index.complete());
    assert_eq!(
        std::str::from_utf8(index.slice("/a~1b/0", &out).unwrap()).unwrap(),
        "\"猫\""
    );
    assert_eq!(
        std::str::from_utf8(index.slice("/a~1b", &out).unwrap()).unwrap(),
        "[\"猫\",2]"
    );
    assert_eq!(
        std::str::from_utf8(index.slice("/note", &out).unwrap()).unwrap(),
        "\"x~y/z\""
    );
}

#[test]
fn custom_indent_formatter() {
    use serde_json::ser::PrettyFormatter;
    let value = json!({"k": [1]});
    let mut out = Vec::new();
    let index = {
        let mut ser =
            PointerSerializer::with_formatter(&mut out, PrettyFormatter::with_indent(b"\t"));
        ser.serialize(&value).unwrap();
        ser.index()
    };
    assert_eq!(std::str::from_utf8(&out).unwrap(), "{\n\t\"k\": [\n\t\t1]}");
    assert_eq!(
        std::str::from_utf8(index.slice("/k/0", &out).unwrap()).unwrap(),
        "1"
    );
}

#[test]
fn non_finite_float_key_is_serde_error() {
    // A custom Serialize that emits an f64 key: serde_json's key serializer
    // rejects non-finite floats with a serde error.
    struct NaNKey;
    impl Serialize for NaNKey {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut map = serializer.serialize_map(Some(1))?;
            use serde::ser::SerializeMap as _;
            map.serialize_key(&f64::NAN)?;
            map.serialize_value(&1)?;
            map.end()
        }
    }
    let err = to_vec_tracked(&NaNKey).unwrap_err();
    assert!(!err.index.complete());
}

#[test]
fn nested_deep_ranges_all_valid_json() {
    let v = json!({"a": {"b": [{"c": "d"}, {"e": [true, false, null]}]}});
    let (bytes, index) = to_vec_tracked(&v).unwrap();
    assert_eq!(slice_str(&index, "/a/b/1/e/2", &bytes), "null");
    assert_eq!(slice_str(&index, "/a/b/1/e/0", &bytes), "true");
    assert_eq!(slice_str(&index, "/a/b/0/c", &bytes), r#""d""#);
    for e in index.entries() {
        let _: serde_json::Value = serde_json::from_slice(e.get(&bytes).unwrap()).unwrap();
    }
}

#[test]
fn special_keys_pointer_lookup() {
    let v = json!({
        "~": 1,
        "a/b": 2,
        "héllo": 3,
        "\"": 4,
        "\n": 5
    });
    let (bytes, index) = to_vec_tracked(&v).unwrap();
    assert_eq!(slice_str(&index, "/~0", &bytes), "1");
    assert_eq!(slice_str(&index, "/a~1b", &bytes), "2");
    assert_eq!(slice_str(&index, "/h\u{e9}llo", &bytes), "3");
    assert_eq!(slice_str(&index, "/\"", &bytes), "4");
    assert_eq!(slice_str(&index, "/\n", &bytes), "5");
    // Cross-check every pointer with serde_json's own Value::pointer.
    for entry in index.entries() {
        let got = v.pointer(&entry.pointer).expect("serde pointer resolves");
        let raw = std::str::from_utf8(entry.get(&bytes).unwrap()).unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(&reparsed, got);
    }
}
