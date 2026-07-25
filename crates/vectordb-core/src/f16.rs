/// An IEEE 754 half-precision (binary16) value.
///
/// The value's bit pattern is IEEE 754 binary16, accessible through [`F16::from_bits`] and
/// [`F16::to_bits`]. No third-party types appear in `F16`'s API. `F16` values may be non-finite;
/// [`Doc::validate`](crate::Doc::validate) rejects non-finite values stored in a document.
///
/// Equality and ordering inherit IEEE 754 partial semantics: NaN is unequal to itself and unordered,
/// while `-0.0 == 0.0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct F16(half::f16);

impl F16 {
    /// Converts an `f32` value to half precision using round-to-nearest, ties-to-even.
    pub fn from_f32(value: f32) -> Self {
        Self(half::f16::from_f32(value))
    }

    /// Creates a half-precision value from its IEEE 754 binary representation.
    pub const fn from_bits(bits: u16) -> Self {
        Self(half::f16::from_bits(bits))
    }

    /// Converts this value to `f32` without loss of precision.
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    /// Returns this value's IEEE 754 binary representation.
    pub const fn to_bits(self) -> u16 {
        self.0.to_bits()
    }

    /// Returns whether this value is neither infinite nor NaN.
    pub const fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}
