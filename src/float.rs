#[allow(dead_code)]
pub(crate) trait Float {
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn acos(self) -> Self;
}

#[cfg(feature = "std")]
impl Float for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }
}

#[cfg(not(feature = "std"))]
impl Float for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self)
    }
}
