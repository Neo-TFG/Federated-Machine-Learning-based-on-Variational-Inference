use crate::communication::protocol::{ClientMsg, ServerMsg};
use crate::communication::tls::setup_client_tls;
use crate::distributions::TrainableModel;
use crate::{ClientBayoneseFramed, SerialisationFormat, ServerMsgPossibility};
use std::path::PathBuf;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

pub struct BayoneseClient<T>
where
    T: TrainableModel,
{
    channel: ClientBayoneseFramed<T>,
    last_likelihood: Option<T::LikelihoodFactor>,
    damping: Option<f64>,
}

#[derive(Debug)]
pub enum ServerError {
    ClosedChannel,
    ErrorReadingFromChannel,
    ErrorSendingIntoChannel,
    WrongStateInput,
    NotAcceptedIntoCluster {
        reason: Option<String>,
        fixable: bool,
    },
}

impl<T: TrainableModel> BayoneseClient<T> {
    pub async fn send(&mut self, msg: ClientMsg<T>) -> Result<(), std::io::Error> {
        self.channel.send(msg).await
    }
    pub async fn join_cluster(&mut self, data_size: u64) -> Result<(), ServerError> {
        self.channel
            .send(ClientMsg::JoinCluster { data_size })
            .await
            .map_err(|x| ServerError::ErrorSendingIntoChannel)?;

        match self.channel.next().await.unwrap().unwrap() {
            ServerMsg::AcceptedIntoCluster { .. } => Ok(()),
            ServerMsg::RejectionFromCluster { reason, fixable } => {
                Err(ServerError::NotAcceptedIntoCluster { reason, fixable })
            }
            _ => Err(ServerError::WrongStateInput),
        }
    }
    pub async fn wait_for_posterior(
        &mut self,
    ) -> Result<(T::ParametricPosterior, bool), ServerError> {
        Ok(match self.channel.get_message().await {
            ServerMsgPossibility::Msg(ServerMsg::SelectedForTraining {
                current_posterior,
                damping_factor,
            }) => {
                self.damping = damping_factor;

                (current_posterior, false)
            }
            ServerMsgPossibility::Msg(ServerMsg::EndOfTraining {
                final_posterior, ..
            }) => {
                self.channel
                    .send(ClientMsg::FinalLeaveTraining {
                        available_for_future_training: false,
                    })
                    .await
                    .unwrap();
                (final_posterior, true)
            }
            ServerMsgPossibility::Msg(_) => {
                todo!("Handle the rest of combinations")
            }
            ServerMsgPossibility::ChannelClosed => {
                return Err(ServerError::ClosedChannel);
            }
            ServerMsgPossibility::MsgError(_e) => {
                return Err(ServerError::ErrorReadingFromChannel);
            } //ServerMsg::SelectedForTraining { current_posterior } => current_posterior,
        })
    }
    pub async fn send_updated_likelihood(
        &mut self,
        delta: T::Delta,
        new_likelihood: T::LikelihoodFactor,
        loss: f64,
    ) -> Result<(), std::io::Error> {
        self.channel
            .send(ClientMsg::UpdatedLikelihood {
                loss,
                delta,
                new_likelihood,
            })
            .await?;
        Ok(())
    }
    pub async fn extract_last_likelihood(
        &self,
        posterior: T::ParametricPosterior,
    ) -> T::ShapeWithoutLikelihood {
        T::extract_likelihood_factor(posterior, self.last_likelihood.clone())
    }
    pub async fn calculate_new_likelihood_and_delta(
        &mut self,
        new_posterior: T::ParametricPosterior,
        past_posterior: T::ParametricPosterior,
    ) -> (T::LikelihoodFactor, T::Delta) {
        let new_likelihood = T::calculate_new_likelihood(
            new_posterior,
            past_posterior,
            self.last_likelihood.clone(),
            self.damping,
        );
        let delta = T::calculate_delta(self.last_likelihood.clone(), new_likelihood.clone());
        self.last_likelihood = Some(new_likelihood.clone());
        (new_likelihood, delta)
    }
}

pub async fn setup_client<T>(
    cafile: PathBuf,
    certfile: PathBuf,
    keyfile: PathBuf,
    server_hostname: String,
    server_address: &str,
    serialisation_format: SerialisationFormat,
) -> BayoneseClient<T>
where
    T: TrainableModel,
{
    let stream = setup_client_tls(cafile, certfile, keyfile, server_hostname, server_address).await;

    let length_delimited = Framed::new(stream, LengthDelimitedCodec::new());
    let client_framed = ClientBayoneseFramed::new(serialisation_format, length_delimited);

    BayoneseClient {
        channel: client_framed,
        last_likelihood: None,
        damping: None,
    }
}
