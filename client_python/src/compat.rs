use async_compat::Compat;
use futures::StreamExt;
use paste::paste;
use pyo3::prelude::*;
use pyo3::pyclass;

use crate::{MultivariateNormal, Normal};
use bayonese::client::setup_client;
use bayonese::client::BayoneseClient;
use bayonese::communication::protocol::ClientMsg;
use bayonese::distributions::TrainableModel;
use bayonese::SerialisationFormat;
use pyo3::exceptions::PyValueError;
use std::path::PathBuf;

macro_rules! create_interface_unwrapped_exponential {
    ($name: ident, $type: ty) => {
        paste! {
                        #[pyclass]
                        #[derive(Debug, Clone)]
                        pub struct [<PyClientMsg $name>]
                        {
                            inner: ClientMsg<$type>
                        }
                        #[pyclass]
                        pub struct [<PyBayoneseClient $name>] {
                            client: BayoneseClient<$type>,
                        }

                        #[pymethods]
            impl [<PyBayoneseClient $name>] {

                pub fn join_cluster(&mut self, data_size: u64) -> PyResult<()> {
            Ok(futures::executor::block_on(Compat::new(async {
                self.client
                    .join_cluster(data_size)
                    .await
                    .unwrap()
            })))
        }
        pub fn wait_for_posterior(&mut self) -> PyResult<($type, bool)> {
            Ok(futures::executor::block_on(Compat::new(async { self.client.wait_for_posterior().await.unwrap()} )))
        }
        pub fn send_updated_likelihood(
            &mut self,
            delta: $type ,
            new_likelihood: $type,
            loss: f64,
        ) -> PyResult<()> {
            Ok(futures::executor::block_on(Compat::new(async {
                self.client
                    .send_updated_likelihood(delta, new_likelihood, loss).await.unwrap()
            })))
        }

                pub fn extract_last_likelihood(
        &self,
        posterior: $type,
    ) -> PyResult<$type> {
                    Ok(futures::executor::block_on(Compat::new(async {
                self.client.extract_last_likelihood(posterior).await
            })))

    }
                pub fn calculate_new_likelihood_and_delta(&mut self, new_posterior: $type, old_posterior: $type) -> PyResult<($type, $type)> {
                    Ok(futures::executor::block_on(Compat::new(async {
                self.client.calculate_new_likelihood_and_delta(new_posterior, old_posterior).await
            })))


    }

            }
                        #[pyfunction]
                        pub fn [<setup_tls_ $name>](
    cafile: PathBuf,
    certfile: PathBuf,
    keyfile: PathBuf,
    server_hostname: String,
    server_address: &str,
    serialisation_format: PySerialisationFormat,
) -> [<PyBayoneseClient $name>] {
    futures::executor::block_on(Compat::new(async {
        [<PyBayoneseClient $name>]{client:
        setup_client(
            cafile,
            certfile,
            keyfile,
            server_hostname,
            server_address,
            serialisation_format.into(),
        ).await
    }
    }))
}
                    }
    };
}
create_interface_unwrapped_exponential!(Normal, Normal);
create_interface_unwrapped_exponential!(MultivariateNormal, MultivariateNormal);

#[pyclass]
#[derive(Copy, Clone)]
pub enum PySerialisationFormat {
    Json,
    Cbor,
    MessagePack,
    Bincode,
}
impl Into<SerialisationFormat> for PySerialisationFormat {
    fn into(self) -> SerialisationFormat {
        match self {
            PySerialisationFormat::Json => SerialisationFormat::Json,
            PySerialisationFormat::Cbor => SerialisationFormat::Cbor,
            PySerialisationFormat::MessagePack => SerialisationFormat::MessagePack,
            PySerialisationFormat::Bincode => SerialisationFormat::Bincode,
        }
    }
}
