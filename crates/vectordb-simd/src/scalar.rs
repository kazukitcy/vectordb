use vectordb_core::F16;

// Length checks live at the public chokepoint.
pub(crate) fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index] - b[index];
        sum += difference * difference;
    }
    sum
}

// Length checks live at the public chokepoint.
pub(crate) fn neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum += a[index] * b[index];
    }
    -sum
}

// Length checks live at the public chokepoint.
pub(crate) fn squared_l2_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index].to_f32() - b[index].to_f32();
        sum += difference * difference;
    }
    sum
}

// Length checks live at the public chokepoint.
pub(crate) fn neg_dot_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum += a[index].to_f32() * b[index].to_f32();
    }
    -sum
}

// Length and dimension checks live at the public chokepoint.
// The final conversion has the rounding behavior specified by ADR 0002.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum += difference * difference;
    }
    sum as f32
}

// Length and dimension checks live at the public chokepoint.
// The final conversion has the rounding behavior specified by ADR 0002.
// Negation happens after the conversion so that an exactly cancelled sum
// yields -0.0, matching the float kernels' negate-after-reduction contract.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
    }
    -(sum as f32)
}

#[cfg(test)]
// Exact values are the contract under test.
#[allow(clippy::float_cmp)]
mod tests {
    use super::F16;

    #[test]
    fn squared_l2_f32_matches_hand_computed_value() {
        // (1-4)² + (2-6)² + (3-8)² = 9 + 16 + 25 = 50
        assert_eq!(
            super::squared_l2_f32(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0]),
            50.0
        );
    }

    #[test]
    fn neg_dot_f32_negates_the_inner_product() {
        // 1·4 + 2·6 + 3·8 = 40
        assert_eq!(
            super::neg_dot_f32(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0]),
            -40.0
        );
    }

    #[test]
    fn neg_dot_i8_is_exact_at_extremes() {
        // 4 · (−128 · 127) = −65_024 → negated = 65_024
        assert_eq!(super::neg_dot_i8(&[i8::MIN; 4], &[i8::MAX; 4]), 65024.0);
    }

    #[test]
    fn squared_l2_i8_widens_before_squaring() {
        // (−128 − 127)² = 65_025; must not wrap in i8/i16.
        assert_eq!(super::squared_l2_i8(&[i8::MIN], &[i8::MAX]), 65025.0);
    }

    #[test]
    fn f16_kernels_convert_then_accumulate_in_f32() {
        let a: Vec<F16> = [1.0f32, 2.0].iter().map(|v| F16::from_f32(*v)).collect();
        let b: Vec<F16> = [3.0f32, 5.0].iter().map(|v| F16::from_f32(*v)).collect();
        assert_eq!(super::neg_dot_f16(&a, &b), -13.0);
        assert_eq!(super::squared_l2_f16(&a, &b), 13.0);
    }

    #[test]
    fn zero_dimension_yields_signed_additive_identity() {
        assert_eq!(super::squared_l2_f32(&[], &[]).to_bits(), 0.0f32.to_bits());
        assert_eq!(super::neg_dot_f32(&[], &[]).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn neg_dot_i8_cancellation_yields_negative_zero() {
        assert_eq!(
            super::neg_dot_i8(&[1, 1], &[1, -1]).to_bits(),
            (-0.0f32).to_bits()
        );
    }
}
