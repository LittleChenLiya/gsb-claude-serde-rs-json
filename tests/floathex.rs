//! Acceptance tests for the explicit float field protocol adapter
//! (`serde_json::floathex`).
//!
//! Protocol byte format: a registered `f64` field is encoded as a JSON
//! string of exactly 16 lowercase hex characters — the big-endian hex of
//! the IEEE-754 binary64 bit pattern (`f64::to_bits()`), two hex digits per
//! byte, most-significant byte first. Decoding goes straight from hex to
//! `u64` to `f64::from_bits`, never through a decimal intermediary, so
//! negative zero and NaN payloads round-trip bit-for-bit.
//!
//! Run with: cargo test --test floathex

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::floathex::FloatHexProtocol;

const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const QNAN_PAYLOAD_1: u64 = 0x7ff8_0000_0000_0001;
const SNAN_PAYLOAD: u64 = 0x7ff0_0000_0000_0001;
const NEG_NAN_PAYLOAD: u64 = 0xfff8_0000_0000_0042;
const ONE: u64 = 0x3ff0_0000_0000_0000;

#[derive(Serialize, Deserialize, Debug)]
struct Reading {
    name: String,
    value: f64,
}

fn bits(v: f64) -> u64 {
    v.to_bits()
}

#[test]
fn negative_zero_and_nan_payloads_roundtrip_bit_exact() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Floats {
        neg_zero: f64,
        qnan: f64,
        snan: f64,
        neg_nan: f64,
        one: f64,
    }

    let proto = FloatHexProtocol::new(["neg_zero", "qnan", "snan", "neg_nan", "one"]).unwrap();
    let value = Floats {
        neg_zero: f64::from_bits(NEG_ZERO),
        qnan: f64::from_bits(QNAN_PAYLOAD_1),
        snan: f64::from_bits(SNAN_PAYLOAD),
        neg_nan: f64::from_bits(NEG_NAN_PAYLOAD),
        one: 1.0,
    };

    let json = proto.to_string(&value).unwrap();
    assert_eq!(
        json,
        concat!(
            r#"{"neg_zero":"8000000000000000","qnan":"7ff8000000000001","#,
            r#""snan":"7ff0000000000001","neg_nan":"fff8000000000042","#,
            r#""one":"3ff0000000000000"}"#,
        ),
    );

    let back: Floats = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.neg_zero), NEG_ZERO);
    assert_eq!(bits(back.qnan), QNAN_PAYLOAD_1);
    assert_eq!(bits(back.snan), SNAN_PAYLOAD);
    assert_eq!(bits(back.neg_nan), NEG_NAN_PAYLOAD);
    assert_eq!(bits(back.one), ONE);
}

#[test]
fn unregistered_fields_keep_plain_json() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Pair {
        a: f64,
        b: f64,
    }

    let proto = FloatHexProtocol::new(["a"]).unwrap();
    let json = proto.to_string(&Pair { a: 1.0, b: 1.5 }).unwrap();
    // Only `a` uses the protocol; `b` stays an ordinary JSON number.
    assert_eq!(json, r#"{"a":"3ff0000000000000","b":1.5}"#);

    let back: Pair = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.a), ONE);
    assert_eq!(bits(back.b), 1.5f64.to_bits());
}

#[test]
fn unregistered_field_rejects_tagged_string() {
    #[derive(Deserialize, Debug)]
    #[allow(dead_code)]
    struct Pair {
        a: f64,
        b: f64,
    }

    let proto = FloatHexProtocol::new(["a"]).unwrap();
    // `b` is not registered, so a tagged hex string must not be accepted.
    let err = proto
        .from_str::<Pair>(r#"{"a":"3ff0000000000000","b":"3ff0000000000000"}"#)
        .unwrap_err();
    assert!(err.to_string().contains("invalid type"), "{}", err);
}

#[test]
fn malformed_protocol_strings_are_rejected_with_path() {
    let proto = FloatHexProtocol::new(["value"]).unwrap();

    let cases = [
        // too short / too long
        r#"{"name":"x","value":"3ff000000000000"}"#,
        r#"{"name":"x","value":"3ff00000000000000"}"#,
        // uppercase is not the protocol
        r#"{"name":"x","value":"3FF0000000000000"}"#,
        // non-hex character
        r#"{"name":"x","value":"3ff000000000000g"}"#,
        // not a string at all
        r#"{"name":"x","value":1.5}"#,
        // empty
        r#"{"name":"x","value":""}"#,
    ];
    for case in cases {
        let err = proto.from_str::<Reading>(case).unwrap_err();
        assert!(
            err.to_string().contains("$.value"),
            "error {:?} should name the path for input {}",
            err,
            case
        );
    }

    // Trailing garbage after an otherwise valid document is rejected too.
    let err = proto
        .from_str::<Reading>(r#"{"name":"x","value":"3ff0000000000000"} garbage"#)
        .unwrap_err();
    assert!(err.to_string().contains("trailing characters"), "{}", err);
}

#[test]
fn nested_lists_are_addressed_by_path() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Matrix {
        matrix: Vec<Vec<f64>>,
    }

    let value = Matrix {
        matrix: vec![
            vec![f64::from_bits(NEG_ZERO), 1.5],
            vec![f64::from_bits(QNAN_PAYLOAD_1)],
        ],
    };

    // Wildcard segments address every element of the nested lists.
    let proto = FloatHexProtocol::new(["matrix.*.*"]).unwrap();
    let json = proto.to_string(&value).unwrap();
    assert_eq!(
        json,
        r#"{"matrix":[["8000000000000000","3ff8000000000000"],["7ff8000000000001"]]}"#
    );
    let back: Matrix = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.matrix[0][0]), NEG_ZERO);
    assert_eq!(bits(back.matrix[0][1]), 1.5f64.to_bits());
    assert_eq!(bits(back.matrix[1][0]), QNAN_PAYLOAD_1);

    // A literal index only registers the matching row; other rows keep the
    // plain JSON number format and reject tagged strings.
    let proto = FloatHexProtocol::new(["matrix.0.*"]).unwrap();
    let value = Matrix {
        matrix: vec![vec![f64::from_bits(NEG_ZERO)], vec![2.5]],
    };
    let json = proto.to_string(&value).unwrap();
    assert_eq!(json, r#"{"matrix":[["8000000000000000"],[2.5]]}"#);
    let back: Matrix = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.matrix[0][0]), NEG_ZERO);
    assert_eq!(bits(back.matrix[1][0]), 2.5f64.to_bits());
    let err = proto
        .from_str::<Matrix>(r#"{"matrix":[["8000000000000000"],["3ff0000000000000"]]}"#)
        .unwrap_err();
    assert!(err.to_string().contains("invalid type"), "{}", err);
}

#[test]
fn paths_use_actual_serde_field_names_after_rename() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Renamed {
        #[serde(rename = "type")]
        kind: f64,
    }

    // The registered path follows the renamed Serde field name.
    let proto = FloatHexProtocol::new(["type"]).unwrap();
    let json = proto
        .to_string(&Renamed {
            kind: f64::from_bits(NEG_ZERO),
        })
        .unwrap();
    assert_eq!(json, r#"{"type":"8000000000000000"}"#);
    let back: Renamed = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.kind), NEG_ZERO);

    // Registering the Rust field name instead of the Serde name does not
    // engage the protocol.
    let proto = FloatHexProtocol::new(["kind"]).unwrap();
    let json = proto.to_string(&Renamed { kind: 1.0 }).unwrap();
    assert_eq!(json, r#"{"type":1.0}"#);
}

#[test]
fn map_keys_and_deep_paths() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Outer {
        inner: Inner,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct Inner {
        lookup: HashMap<String, f64>,
    }

    let proto = FloatHexProtocol::new(["inner.lookup.*"]).unwrap();
    let mut lookup = HashMap::new();
    lookup.insert("x".to_owned(), f64::from_bits(NEG_ZERO));
    let value = Outer {
        inner: Inner { lookup },
    };
    let json = proto.to_string(&value).unwrap();
    assert_eq!(json, r#"{"inner":{"lookup":{"x":"8000000000000000"}}}"#);
    let back: Outer = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.inner.lookup["x"]), NEG_ZERO);

    // Format errors deep in the tree are labeled with the full path.
    let err = proto
        .from_str::<Outer>(r#"{"inner":{"lookup":{"x":"zz"}}}"#)
        .unwrap_err();
    assert!(
        err.to_string().contains("$.inner.lookup.x"),
        "{}",
        err
    );
}

#[test]
fn optional_and_enum_fields_follow_the_same_path() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Holder {
        maybe: Option<f64>,
        choice: Choice,
    }

    #[derive(Serialize, Deserialize, Debug)]
    enum Choice {
        Value { payload: f64 },
        Empty,
    }

    let proto = FloatHexProtocol::new(["maybe", "choice.payload"]).unwrap();

    let json = proto
        .to_string(&Holder {
            maybe: Some(f64::from_bits(NEG_ZERO)),
            choice: Choice::Value {
                payload: f64::from_bits(QNAN_PAYLOAD_1),
            },
        })
        .unwrap();
    assert_eq!(
        json,
        r#"{"maybe":"8000000000000000","choice":{"Value":{"payload":"7ff8000000000001"}}}"#
    );
    let back: Holder = proto.from_str(&json).unwrap();
    assert_eq!(bits(back.maybe.unwrap()), NEG_ZERO);
    match back.choice {
        Choice::Value { payload } => assert_eq!(bits(payload), QNAN_PAYLOAD_1),
        Choice::Empty => panic!("wrong variant"),
    }

    // None still serializes as plain null.
    let json = proto
        .to_string(&Holder {
            maybe: None,
            choice: Choice::Empty,
        })
        .unwrap();
    assert_eq!(json, r#"{"maybe":null,"choice":"Empty"}"#);
}

#[test]
fn invalid_patterns_are_rejected_at_registration() {
    assert!(FloatHexProtocol::new([""]).is_err());
    assert!(FloatHexProtocol::new(["a..b"]).is_err());
    assert!(FloatHexProtocol::new([".a"]).is_err());
    assert!(FloatHexProtocol::new(["a."]).is_err());
    assert!(FloatHexProtocol::new(["a.*", "b.0.c"]).is_ok());
}

#[test]
fn plain_serializer_and_deserializer_are_unchanged() {
    // The ordinary serde_json entry points know nothing about the protocol.
    assert_eq!(serde_json::to_string(&-0.0f64).unwrap(), "-0.0");
    assert_eq!(serde_json::to_string(&1.5f64).unwrap(), "1.5");
    assert!(serde_json::from_str::<f64>("\"8000000000000000\"").is_err());

    // And a value that went through the protocol still deserializes
    // normally when no protocol is involved (hex string stays a string).
    let proto = FloatHexProtocol::new(["value"]).unwrap();
    let reading = Reading {
        name: "x".to_owned(),
        value: -0.0,
    };
    let json = proto.to_string(&reading).unwrap();
    let plain: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(plain["value"], serde_json::Value::from("8000000000000000"));
}
