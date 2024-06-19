use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use async_compat::CompatExt;
use futures::{SinkExt, StreamExt};
use nalgebra::{SMatrix, SVector, Vector2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use bayonese::distributions::ExponentialFamilyDistribution;

use crate::compat::setup_tls_MultivariateNormal;
use crate::compat::setup_tls_Normal;
use crate::compat::PySerialisationFormat;

mod compat;

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
#[pyclass]
pub struct Normal {
    #[pyo3(get, set)]
    pub mean: f64,
    #[pyo3(get, set)]
    pub variance: f64,
}
#[pymethods]
impl Normal {
    #[new]
    fn new(mean: f64, variance: f64) -> Self {
        Normal { mean, variance }
    }
}
impl Default for Normal {
    fn default() -> Self {
        Normal {
            mean: 0.0,
            variance: 1.0,
        }
    }
}
impl ExponentialFamilyDistribution for Normal {
    type NaturalArgs = Vector2<f64>;

    fn from_nat_params(args: Self::NaturalArgs) -> Self {
        let variance = -1.0 / (2.0 * args[1]);
        let mean = args[0] * variance;
        Normal { mean, variance }
    }

    fn nat_params(&self) -> Self::NaturalArgs {
        Vector2::new(self.mean / self.variance, -1.0 / (2.0 * self.variance))
    }
}
const N_ARGS: usize = 4;
#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
#[pyclass]
pub struct MultivariateNormal {
    pub means: SVector<f64, N_ARGS>,
    pub variances: SMatrix<f64, N_ARGS, N_ARGS>,
}

#[pymethods]
impl MultivariateNormal {
    #[new]
    fn new(means: [f64; N_ARGS], variances: [f64; N_ARGS]) -> Self {
        MultivariateNormal {
            means: SVector::from_iterator(means),
            variances: SMatrix::from_diagonal(&SVector::<f64, N_ARGS>::from_iterator(variances)),
        }
    }
    fn get_means(&self) -> Vec<f64> {
        self.means.iter().cloned().collect()
    }
    fn get_variances(&self) -> Vec<f64> {
        self.variances.diagonal().iter().cloned().collect()
    }
}
impl Default for MultivariateNormal {
    fn default() -> Self {
        MultivariateNormal {
            means: SVector::repeat(0.0),
            variances: SMatrix::from_diagonal_element(1.0),
        }
    }
}
#[derive(Debug)]
pub struct MultiNatArgs<const T: usize>(SVector<f64, T>, SMatrix<f64, T, T>);
impl<const T: usize> Add for MultiNatArgs<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<const T: usize> Sub for MultiNatArgs<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl<const T: usize> Mul<f64> for MultiNatArgs<T> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}
impl<const T: usize> Sum for MultiNatArgs<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut accum = MultiNatArgs(SVector::repeat(0.0), (SMatrix::repeat(0.0)));
        for i in iter {
            accum = accum + i;
        }
        accum
    }
}
impl ExponentialFamilyDistribution for MultivariateNormal {
    type NaturalArgs = MultiNatArgs<N_ARGS>;

    fn from_nat_params(args: Self::NaturalArgs) -> Self {
        let variances = (-2.0 * args.1).try_inverse().unwrap();
        let means = variances * args.0;
        MultivariateNormal { means, variances }
    }

    fn nat_params(&self) -> Self::NaturalArgs {
        let inverse_var = self.variances.try_inverse().unwrap();
        MultiNatArgs(inverse_var * self.means, (-1.0 / 2.0) * inverse_var)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn bayonese_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(setup_tls, m)?)?;
    m.add_class::<PySerialisationFormat>()?;
    m.add_function(wrap_pyfunction!(setup_tls_Normal, m)?)?;
    m.add_function(wrap_pyfunction!(setup_tls_MultivariateNormal, m)?)?;
    m.add_class::<Normal>()?;
    m.add_class::<MultivariateNormal>()?;
    //m.add_class::<PyClientMsgNormal>()?;
    Ok(())
}
