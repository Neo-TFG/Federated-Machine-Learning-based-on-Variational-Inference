use crate::distributions::TrainableModel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ClientMsg<T>
where
    T: TrainableModel,
{
    JoinCluster {
        data_size: u64,
    },
    ReJoinCluster,
    UpdatedLikelihood {
        new_likelihood: T::LikelihoodFactor,
        delta: T::Delta,
        loss: f64,
    },
    ReturnLastLikelihood {
        likelihood: T::LikelihoodFactor,
    },
    EarlyLeaveCluster {
        reason: Option<String>,
        expected_absence: Option<Duration>,
    },
    FinalLeaveTraining {
        available_for_future_training: bool,
    },
    Error {
        error: Option<String>,
    },
    EndOfConnectionAcknowledgement,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ServerMsg<T>
where
    T: TrainableModel,
{
    AcceptedIntoCluster {
        expected_start_time: Option<DateTime<Utc>>,
    },
    ReAcceptanceIntoCluster {
        last_likelihood: T::LikelihoodFactor,
    },
    RejectionFromCluster {
        reason: Option<String>,
        fixable: bool,
    },
    SelectedForTraining {
        current_posterior: T::ParametricPosterior,
        damping_factor: Option<f64>,
    },
    EarlyCloseOfConnection {
        reason: Option<String>,
        return_in: Option<Duration>,
    },
    EndOfTraining {
        final_posterior: T::ParametricPosterior,
        future_training: Option<DateTime<Utc>>,
    },
    Error {
        error: Option<String>,
    },
    EndOfConnectionAcknowledgement,
}
