use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use serde::de::DeserializeOwned;
use serde::Serialize;

pub trait ExponentialFamilyDistribution:
    Send + Debug + Serialize + DeserializeOwned + Clone + Unpin
{
    type NaturalArgs: Add<Output = Self::NaturalArgs>
        + Sum
        + Sub<Output = Self::NaturalArgs>
        + Mul<f64, Output = Self::NaturalArgs>
        + Debug;

    fn from_nat_params(args: Self::NaturalArgs) -> Self;

    fn nat_params(&self) -> Self::NaturalArgs;
}
impl<T> TrainableModel for T
where
    T: ExponentialFamilyDistribution,
{
    type ParametricPosterior = Self;

    type LikelihoodFactor = Self;

    type ShapeWithoutLikelihood = Self;

    type Delta = Self;

    fn update_posterior(
        curr: Self::ParametricPosterior,
        deltas: impl IntoIterator<Item = Self::Delta>,
    ) -> Self::ParametricPosterior {
        let sum = deltas
            .into_iter()
            .map(|x| x.nat_params())
            .sum::<T::NaturalArgs>();
        T::from_nat_params(curr.nat_params() + sum)
    }

    fn extract_likelihood_factor(
        curr: Self::ParametricPosterior,
        factor: Option<Self::LikelihoodFactor>,
    ) -> Self::ShapeWithoutLikelihood {
        if let Some(factor) = factor {
            T::from_nat_params(curr.nat_params() - factor.nat_params())
        } else {
            curr
        }
    }

    fn calculate_delta(
        past_likelihood: Option<Self::LikelihoodFactor>,
        new_likelihood: Self::LikelihoodFactor,
    ) -> Self::Delta {
        if let Some(past_likelihood) = past_likelihood {
            T::from_nat_params(new_likelihood.nat_params() - past_likelihood.nat_params())
        } else {
            new_likelihood
        }
    }

    fn calculate_new_likelihood(
        approximated_posterior: Self::ParametricPosterior,
        past_posterior: Self::ParametricPosterior,
        past_likelihood: Option<Self::LikelihoodFactor>,
        damping: Option<f64>,
    ) -> Self::LikelihoodFactor {
        let factor = if let Some(factor) = damping {
            factor
        } else {
            1.0
        };

        if let Some(past_likelihood) = past_likelihood {
            let new_likelihood_params = approximated_posterior.nat_params()
                - Self::extract_likelihood_factor(past_posterior, Some(past_likelihood.clone()))
                    .nat_params();
            T::from_nat_params(
                new_likelihood_params * factor + (past_likelihood.nat_params() * (1.0 - factor)),
            )
        } else {
            T::from_nat_params(
                (approximated_posterior.nat_params() - past_posterior.nat_params()) * factor,
            )
        }
    }
}

pub trait TrainableModel: Debug {
    type ParametricPosterior: Send + Serialize + Unpin + DeserializeOwned + Debug + Clone;
    type LikelihoodFactor: Send + Serialize + Unpin + DeserializeOwned + Clone + Debug;
    type ShapeWithoutLikelihood: Debug;

    type Delta: Send + Serialize + Unpin + DeserializeOwned + Debug + Clone;
    fn update_posterior(
        curr: Self::ParametricPosterior,
        deltas: impl IntoIterator<Item = Self::Delta>,
    ) -> Self::ParametricPosterior;
    fn extract_likelihood_factor(
        curr: Self::ParametricPosterior,
        factor: Option<Self::LikelihoodFactor>,
    ) -> Self::ShapeWithoutLikelihood;

    fn calculate_delta(
        past_likelihood: Option<Self::LikelihoodFactor>,
        new_likelihood: Self::LikelihoodFactor,
    ) -> Self::Delta;

    fn calculate_new_likelihood(
        approximated_posterior: Self::ParametricPosterior,
        past_posterior: Self::ParametricPosterior,
        past_likelihood: Option<Self::LikelihoodFactor>,
        damping: Option<f64>,
    ) -> Self::LikelihoodFactor;
}
