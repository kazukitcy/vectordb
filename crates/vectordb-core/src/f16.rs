/// An IEEE 754 half-precision (binary16) value.
///
/// The value's bit pattern is IEEE 754 binary16, accessible through [`F16::from_bits`] and
/// [`F16::to_bits`]. `F16` is layout-identical to that `u16` bit representation: it is two bytes
/// in size, two-byte aligned, and a `&[F16]` slice's memory is a sequence of binary16 bit
/// patterns. No third-party types appear in `F16`'s API. `F16` values may be non-finite;
/// [`Doc::validate`](crate::Doc::validate) rejects non-finite values stored in a document.
///
/// Equality and ordering inherit IEEE 754 partial semantics: NaN is unequal to itself and unordered,
/// while `-0.0 == 0.0`.
// repr(transparent) over half::f16 (itself transparent over u16) backs the documented layout
// guarantee; SIMD kernels load &[F16] directly as u16 lanes. See ADR 0002 for the upgrade
// constraint this places on the `half` dependency.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
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
