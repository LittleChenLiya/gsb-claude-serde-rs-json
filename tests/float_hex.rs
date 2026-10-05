//! Tests for the float hex protocol adapter.

use serde::{Deserialize, Serialize};
use serde_json::float_hex::{self, FloatHexConfig};

#[test]
fn negative_zero_roundtrip() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("value");

    let m = Measurement { value: -0.0 };
    let json = float_hex::to_string(&m, &config).unwrap();
    assert_eq!(json, r#"{"value":"8000000000000000"}"#);

    let back: Measurement = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.value.to_bits(), (-0.0f64).to_bits());
}

#[test]
fn nan_payload_roundtrip() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("value");

    // Create a NaN with a specific payload.
    let nan_bits: u64 = 0x7FF8_0000_0000_0001;
    let nan = f64::from_bits(nan_bits);

    let m = Measurement { value: nan };
    let json = float_hex::to_string(&m, &config).unwrap();
    assert_eq!(json, r#"{"value":"7ff8000000000001"}"#);

    let back: Measurement = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.value.to_bits(), nan_bits);
}

#[test]
fn unregistered_field_uses_normal_json() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let config = FloatHexConfig::new();

    let m = Measurement { value: 1.5 };
    let json = float_hex::to_string(&m, &config).unwrap();
    assert_eq!(json, r#"{"value":1.5}"#);

    let back: Measurement = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.value, 1.5);
}

#[test]
fn unregistered_field_rejects_hex_string() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let config = FloatHexConfig::new();

    // A hex string at an unregistered path should be rejected.
    let json = r#"{"value":"8000000000000000"}"#;
    let result: Result<Measurement, _> = float_hex::from_str(json, &config);
    assert!(result.is_err());
}

#[test]
fn malformed_hex_rejected() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("value");

    // Wrong length.
    let json = r#"{"value":"800000000000000"}"#;
    let result: Result<Measurement, _> = float_hex::from_str(json, &config);
    assert!(result.is_err());

    // Non-hex characters.
    let json = r#"{"value":"800000000000000g"}"#;
    let result: Result<Measurement, _> = float_hex::from_str(json, &config);
    assert!(result.is_err());

    // Uppercase hex.
    let json = r#"{"value":"800000000000000A"}"#;
    let result: Result<Measurement, _> = float_hex::from_str(json, &config);
    assert!(result.is_err());
}

#[test]
fn nested_lists_handled_by_path() {
    #[derive(Serialize, Deserialize)]
    struct Grid {
        values: Vec<Vec<f64>>,
    }

    let mut config = FloatHexConfig::new();
    config.register("values");

    let g = Grid {
        values: vec![vec![-0.0, 1.0], vec![f64::from_bits(0x7FF8_0000_0000_0001)]],
    };
    let json = float_hex::to_string(&g, &config).unwrap();
    // All f64 values should be encoded as hex strings.
    assert!(json.contains("8000000000000000"));
    assert!(json.contains("3ff0000000000000"));
    assert!(json.contains("7ff8000000000001"));

    let back: Grid = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.values[0][0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(back.values[0][1].to_bits(), 1.0f64.to_bits());
    assert_eq!(back.values[1][0].to_bits(), 0x7FF8_0000_0000_0001);
}

#[test]
fn rename_follows_field_name() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        #[serde(rename = "temp")]
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("temp");

    let m = Measurement { value: -0.0 };
    let json = float_hex::to_string(&m, &config).unwrap();
    assert_eq!(json, r#"{"temp":"8000000000000000"}"#);

    let back: Measurement = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.value.to_bits(), (-0.0f64).to_bits());
}

#[test]
fn nested_struct_path() {
    #[derive(Serialize, Deserialize)]
    struct Outer {
        inner: Inner,
    }

    #[derive(Serialize, Deserialize)]
    struct Inner {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("inner.value");

    let o = Outer {
        inner: Inner { value: -0.0 },
    };
    let json = float_hex::to_string(&o, &config).unwrap();
    assert_eq!(json, r#"{"inner":{"value":"8000000000000000"}}"#);

    let back: Outer = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.inner.value.to_bits(), (-0.0f64).to_bits());
}

#[test]
fn unregistered_nested_field_uses_normal_json() {
    #[derive(Serialize, Deserialize)]
    struct Outer {
        inner: Inner,
    }

    #[derive(Serialize, Deserialize)]
    struct Inner {
        value: f64,
        other: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("inner.value");

    let o = Outer {
        inner: Inner {
            value: -0.0,
            other: 2.5,
        },
    };
    let json = float_hex::to_string(&o, &config).unwrap();
    // value should be hex, other should be normal JSON.
    assert!(json.contains("8000000000000000"));
    assert!(json.contains("2.5"));

    let back: Outer = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.inner.value.to_bits(), (-0.0f64).to_bits());
    assert_eq!(back.inner.other, 2.5);
}

#[test]
fn malformed_hex_error_includes_path() {
    #[derive(Serialize, Deserialize, Debug)]
    struct Measurement {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("value");

    // Wrong length - error should include the path.
    let json = r#"{"value":"800000000000000"}"#;
    let result: Result<Measurement, _> = float_hex::from_str(json, &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    assert!(err_msg.contains("value"), "error should contain path, got: {}", err_msg);
}

#[test]
fn infinity_roundtrip() {
    #[derive(Serialize, Deserialize)]
    struct Measurement {
        value: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("value");

    let m = Measurement { value: f64::INFINITY };
    let json = float_hex::to_string(&m, &config).unwrap();
    assert_eq!(json, r#"{"value":"7ff0000000000000"}"#);

    let back: Measurement = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.value, f64::INFINITY);
}

#[test]
fn top_level_f64_not_registered() {
    let config = FloatHexConfig::new();

    // Top-level f64 with no registered path uses normal JSON.
    let json = float_hex::to_string(&1.5f64, &config).unwrap();
    assert_eq!(json, "1.5");

    let back: f64 = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back, 1.5);
}

#[test]
fn multiple_registered_paths() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        a: f64,
        b: f64,
        c: f64,
    }

    let mut config = FloatHexConfig::new();
    config.register("a");
    config.register("c");

    let d = Data {
        a: -0.0,
        b: 1.5,
        c: f64::from_bits(0x7FF8_0000_0000_0001),
    };
    let json = float_hex::to_string(&d, &config).unwrap();
    // a and c should be hex, b should be normal JSON.
    assert!(json.contains("8000000000000000"));
    assert!(json.contains("7ff8000000000001"));
    assert!(json.contains("1.5"));

    let back: Data = float_hex::from_str(&json, &config).unwrap();
    assert_eq!(back.a.to_bits(), (-0.0f64).to_bits());
    assert_eq!(back.b, 1.5);
    assert_eq!(back.c.to_bits(), 0x7FF8_0000_0000_0001);
}
