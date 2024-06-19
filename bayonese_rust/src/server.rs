use crate::communication::protocol::ServerMsg::{AcceptedIntoCluster, RejectionFromCluster};
use crate::communication::protocol::{ClientMsg, ServerMsg};
use crate::communication::tls::get_certificate_from_client;
use crate::distributions::TrainableModel;
use crate::server::ChildCoordinatorTaskMsg::{
    EarlyLeaveCluster, FinalLeaveTraining, NewLikelihood, NewMember,
};
use crate::server::ClientError::{ClosedChannel, ErrorReadingFromChannel, WrongStateInput};
use crate::server::ClientState::{Computing, Disconnected, Waiting};
use crate::server::ConnAcceptorMsg::StopAccepting;
use crate::server::CoordinatorChildTaskMsg::ComputeNewLikelihood;
use crate::server::InputEvent::{Finish, StopWaitingForClients};
use crate::{ClientMsgPossibility, SerialisationFormat, ServerBayoneseFramed, TrainingMode};
use chrono::{DateTime, Utc};
use rustls::pki_types::CertificateDer;
use std::io::{stdin, BufRead};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_rustls::TlsAcceptor;
use tokio_util::codec::LengthDelimitedCodec;

enum TrainingMessageHandler {
    Sequential {
        order: Vec<ClientId>,
        position: usize,
    },
    Synchronous {
        remaining: usize,
        damping_factor: f64,
    },
    Asynchronous {
        damping_factor: f64,
    },
    Uninitialized,
}

impl TrainingMessageHandler {
    pub fn init<T>(
        mode: TrainingMode,
        prior: T::ParametricPosterior,
        clients: &mut Vec<(ClientInfo<T>, UnboundedSender<CoordinatorChildTaskMsg<T>>)>,
    ) -> Self
    where
        T: TrainableModel,
    {
        let num_clients = clients.len();
        match mode {
            TrainingMode::Sequential => {
                clients[0]
                    .1
                    .send(ComputeNewLikelihood {
                        current_posterior: prior,
                        damping_factor: None,
                    })
                    .expect("Client Channel dropped");
            }
            TrainingMode::Synchronous | TrainingMode::Asynchronous => {
                for (_, channel) in clients {
                    let current_posterior = prior.clone();
                    channel
                        .send(ComputeNewLikelihood {
                            current_posterior,
                            damping_factor: Some(1.0 / (2 * num_clients) as f64),
                        })
                        .expect("Client Channel dropped");
                }
            }
        };
        match mode {
            TrainingMode::Sequential => TrainingMessageHandler::Sequential {
                order: (0..num_clients).collect(),
                position: 1 % num_clients,
            },
            TrainingMode::Synchronous => TrainingMessageHandler::Synchronous {
                damping_factor: 1.0 / num_clients as f64,
                remaining: num_clients,
            },
            TrainingMode::Asynchronous => TrainingMessageHandler::Asynchronous {
                damping_factor: 1.0 / num_clients as f64,
            },
        }
    }

    pub fn handle<T>(
        &mut self,
        id: ClientId,
        current_posterior: T::ParametricPosterior,
        clients: &mut Vec<(ClientInfo<T>, UnboundedSender<CoordinatorChildTaskMsg<T>>)>,
    ) -> bool
    where
        T: TrainableModel,
    {
        match self {
            TrainingMessageHandler::Sequential { order, position } => {
                let next_id = order[*position];
                *position = (*position + 1) % order.len();

                clients[next_id]
                    .1
                    .send(ComputeNewLikelihood {
                        current_posterior,
                        damping_factor: None,
                    })
                    .expect("Client Channel dropped");
                false
            }
            TrainingMessageHandler::Synchronous {
                remaining,
                damping_factor,
            } => {
                *remaining -= 1;
                if *remaining == 0 {
                    *remaining = clients.len();

                    for (_, channel) in clients {
                        let current_posterior = current_posterior.clone();
                        channel
                            .send(ComputeNewLikelihood {
                                current_posterior,
                                damping_factor: Some(*damping_factor),
                            })
                            .expect("Client Channel dropped");
                    }
                }
                false
            }
            TrainingMessageHandler::Asynchronous { damping_factor } => {
                clients[id]
                    .1
                    .send(ComputeNewLikelihood {
                        current_posterior,
                        damping_factor: Some(*damping_factor),
                    })
                    .expect("Client Channel dropped");
                false
            }
            TrainingMessageHandler::Uninitialized => {
                panic!("The training message handler has not been initialized")
            }
        }
    }
}

pub enum CoordinatorChildTaskMsg<T>
where
    T: TrainableModel,
{
    ComputeNewLikelihood {
        current_posterior: T::ParametricPosterior,
        damping_factor: Option<f64>,
    },
    AcceptedIntoTraining {
        id: ClientId,
        expected_start: Option<DateTime<Utc>>,
    },
    RejectedFromTraining {
        reason: Option<String>,
        fixable: bool,
    },
    EndOfTraining {
        posterior: T::ParametricPosterior,
    },
}

type ClientId = usize;

struct ClientInfo<T>
where
    T: TrainableModel,
{
    certs: Vec<CertificateDer<'static>>,
    delta: Option<T::Delta>,
    last_likelihood: Option<T::LikelihoodFactor>,
    loss: f64,
}

async fn handle_child_message<T>(
    msg: ChildCoordinatorTaskMsg<T>,
    current_posterior: &mut T::ParametricPosterior,
    training_handler: &mut TrainingMessageHandler,
    clients: &mut Vec<(ClientInfo<T>, UnboundedSender<CoordinatorChildTaskMsg<T>>)>,
) -> bool
where
    T: TrainableModel,
{
    match msg {
        NewMember(certs, chann) => {
            if clients.iter().any(|(info, _)| info.certs[0] == certs[0]) {
                chann
                    .send(CoordinatorChildTaskMsg::RejectedFromTraining {
                        reason: Some("There is already a client with this certificate".to_string()),
                        fixable: false,
                    })
                    .unwrap();
            } else {
                let new_id = clients.len();
                chann
                    .send(CoordinatorChildTaskMsg::AcceptedIntoTraining {
                        id: new_id,
                        expected_start: None,
                    })
                    .unwrap();
                clients.push((
                    ClientInfo {
                        certs,
                        delta: None,
                        last_likelihood: None,
                        loss: 0.0,
                    },
                    chann,
                ));
            }
        }
        NewLikelihood {
            client_id,
            new_likelihood,
            delta,
            loss,
        } => {
            clients[client_id].0.delta = Some(delta);
            clients[client_id].0.last_likelihood = Some(new_likelihood);
            clients[client_id].0.loss = loss;

            *current_posterior = T::update_posterior(
                current_posterior.clone(),
                clients
                    .iter()
                    .flat_map(|(client, _)| &client.delta)
                    .cloned(),
            );
            return training_handler.handle(client_id, current_posterior.clone(), clients);
        }
        EarlyLeaveCluster {
            ..
        } => (unimplemented!("In this implementation of the PVI protocol, role transfer has not been implemented due to vastly different requirements between users")),
        FinalLeaveTraining {
            ..
        } => panic!("Should not ever receive this message in this stage of the process"),
    };
    false
}

async fn coordinator<T>(
    mut client_coord_receiver: UnboundedReceiver<ChildCoordinatorTaskMsg<T>>,
    mut inp_coord_receiver: UnboundedReceiver<InputEvent>,
    coord_conn_sender: UnboundedSender<ConnAcceptorMsg>,
    prior: T::ParametricPosterior,
    training_mode: TrainingMode,
) -> T::ParametricPosterior
where
    T: TrainableModel,
{
    let mut clients: Vec<(ClientInfo<T>, UnboundedSender<CoordinatorChildTaskMsg<T>>)> = vec![];
    let mut curr = prior;
    let mut training_handler = TrainingMessageHandler::Uninitialized;
    loop {
        tokio::select! {
            input_event = inp_coord_receiver.recv() => {
                match input_event.expect("Input Channel was closed"){
                    StopWaitingForClients => {
                        coord_conn_sender.send(StopAccepting).unwrap();
                        training_handler = TrainingMessageHandler::init(training_mode, curr.clone(), &mut clients);
                    },
                    Finish => {
                        for (_index, (info, chann)) in clients.iter().enumerate(){
                            chann.send(CoordinatorChildTaskMsg::EndOfTraining{posterior:curr.clone()}).unwrap();
                        }
                        break;
                    },
                };
            },
            client_event = client_coord_receiver.recv() => {

                let Some(msg) = client_event else {
                    // Client channel was closed
                    break;
                };

                if handle_child_message(msg, &mut curr, &mut training_handler, &mut clients).await{

                        for (index, (info, chann)) in clients.iter().enumerate(){
                            chann.send(CoordinatorChildTaskMsg::EndOfTraining{posterior:curr.clone()}).unwrap();
                        }
                        break;
                }
            }
        }
    }

    let mut remaining = clients.len();
    while remaining != 0 {
        if let Some(FinalLeaveTraining {
            client_id: _,
            available_for_future_training: _,
        }) = client_coord_receiver.recv().await
        {
            remaining -= 1
        }
    }

    curr
}

pub async fn create_server<T>(
    acceptor: TlsAcceptor,
    listener: TcpListener,
    prior: T::ParametricPosterior,
    mode: TrainingMode,
    serialisation_format: SerialisationFormat,
) -> T::ParametricPosterior
where
    T: TrainableModel + 'static,
{
    let (train_coord_tx, train_coord_rx) = tokio::sync::mpsc::unbounded_channel();
    let (input_coord_tx, input_coord_rx) = tokio::sync::mpsc::unbounded_channel();
    let (coord_conn_tx, coord_conn_rx) = tokio::sync::mpsc::unbounded_channel();

    let _input_handler = tokio::task::spawn_blocking(|| handle_input(input_coord_tx));
    let coordinator = tokio::spawn(coordinator::<T>(
        train_coord_rx,
        input_coord_rx,
        coord_conn_tx,
        prior,
        mode,
    ));
    let _connection_handler = tokio::spawn(client_acceptor(
        acceptor,
        listener,
        train_coord_tx,
        coord_conn_rx,
        serialisation_format,
    ));
    let pos = coordinator.await.unwrap();
    _input_handler.abort();
    _connection_handler.abort();
    pos
}

pub enum ConnAcceptorMsg {
    StopAccepting,
}

async fn client_acceptor<T>(
    acceptor: TlsAcceptor,
    listener: TcpListener,
    client_coord_sender: UnboundedSender<ChildCoordinatorTaskMsg<T>>,
    mut coord_conn_receiver: UnboundedReceiver<ConnAcceptorMsg>,
    serialisation_format: SerialisationFormat,
) where
    T: TrainableModel + 'static,
{
    let mut clients = vec![];
    loop {
        let (socket, _addr) = tokio::select! {
            _stop = coord_conn_receiver.recv() => {
                // Finished waiting for clients
                break;
            }
            list = listener.accept() => {
                match list {
            Ok((sock, addr)) => {
                (sock, addr)
            }
            Err(e) => {
                continue;
            }
            }
        }
        };
        let acceptor = acceptor.clone();
        let Ok(stream) = acceptor.accept(socket).await else {
            continue;
        };
        let client_handle = tokio::spawn(handle_client(
            stream,
            client_coord_sender.clone(),
            serialisation_format,
        ));
        clients.push(client_handle);
    }
}

#[derive(Copy, Clone)]
pub enum InputEvent {
    StopWaitingForClients,
    Finish,
}

fn handle_input(tx: UnboundedSender<InputEvent>) {
    println!("Press s to stop waiting for clients and f to stop the training: -------");
    let stdin = stdin().lock();
    for line in stdin.lines() {
        let Ok(line) = line else {
            println!("Invalid Input");
            continue;
        };
        let event = match &*line {
            "s" | "stop" => StopWaitingForClients,
            "f" | "finish" => Finish,
            _ => continue,
        };
        if let Err(x) = tx.send(event) {
            println!("Channel Error {x}, closing down");
            break;
        };
        if let Finish = event {
            println!("Ending training");
            break;
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChildCoordinatorTaskMsg<T>
where
    T: TrainableModel,
{
    NewMember(
        Vec<CertificateDer<'static>>,
        UnboundedSender<CoordinatorChildTaskMsg<T>>,
    ),
    NewLikelihood {
        client_id: ClientId,
        new_likelihood: T::LikelihoodFactor,
        delta: T::Delta,
        loss: f64,
    },
    EarlyLeaveCluster {
        client_id: ClientId,
        reason: Option<String>,
        expected_absence: Option<Duration>,
    },
    FinalLeaveTraining {
        client_id: ClientId,
        available_for_future_training: bool,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum ClientState {
    Waiting,
    Disconnected,
    Computing,
    WaitingForClientFinishResponse,
}

async fn handle_client<T>(
    stream: tokio_rustls::server::TlsStream<TcpStream>,
    client_coord_sender: UnboundedSender<ChildCoordinatorTaskMsg<T>>,
    serialisation_format: SerialisationFormat,
) where
    T: TrainableModel + 'static,
{
    let certs = get_certificate_from_client(&stream);
    let length_delimited = tokio_util::codec::Framed::new(stream, LengthDelimitedCodec::new());
    let mut framed = ServerBayoneseFramed::new(serialisation_format, length_delimited);
    let (coord_child_tx, mut coord_child_rx) = tokio::sync::mpsc::unbounded_channel();
    match framed.get_message().await {
        ClientMsgPossibility::Msg(ClientMsg::JoinCluster {
            data_size: _data_size,
        }) => {
            client_coord_sender
                .send(NewMember(certs, coord_child_tx))
                .unwrap();
        }
        _ => {
            // If the client does not want to join the cluster shut down connection
            return;
        }
    };

    let id = match coord_child_rx
        .recv()
        .await
        .expect("Coord Channel should be open")
    {
        CoordinatorChildTaskMsg::AcceptedIntoTraining { id, expected_start } => {
            framed
                .send(AcceptedIntoCluster {
                    expected_start_time: expected_start,
                })
                .await
                .unwrap();
            id
        }
        CoordinatorChildTaskMsg::RejectedFromTraining { reason, fixable } => {
            framed
                .send(RejectionFromCluster { reason, fixable })
                .await
                .unwrap();
            return;
        }
        _ => {
            panic!("The coordinator should respond to the question of joining first")
        }
    };
    let mut state = Waiting;
    loop {
        tokio::select! {
            coord_msg = coord_child_rx.recv() => {
                let Some(msg) = coord_msg else{
                    // Coord channel closed
                    break;
                };
                match msg{
                    ComputeNewLikelihood {current_posterior, damping_factor} => {
                        state = Computing;
                        framed.send(ServerMsg::SelectedForTraining {current_posterior, damping_factor}).await.unwrap()
                    },
                    CoordinatorChildTaskMsg::EndOfTraining {posterior} => {
                        framed.send(ServerMsg::EndOfTraining {final_posterior: posterior,future_training: None}).await.unwrap();
                        state = ClientState::WaitingForClientFinishResponse;
                    }
                    _ => panic!("The coordinator should only ask new likelihoods to train or shutdown the training")
                };
            },
            client_msg = framed.get_message() => {
                match handle_client_message(&mut framed, client_msg, id, &client_coord_sender, &state).await{
                    Ok(new_state) => state = new_state,
                    Err(_) => break,
                };

            }
        }
    }
}

pub enum ClientError {
    ClosedChannel,
    ErrorReadingFromChannel,
    WrongStateInput,
}

async fn handle_client_message<T>(
    framed: &mut ServerBayoneseFramed<T>,
    msg_possibility: ClientMsgPossibility<T>,
    client_id: ClientId,
    child_coord_tx: &UnboundedSender<ChildCoordinatorTaskMsg<T>>,
    state: &ClientState,
) -> Result<ClientState, ClientError>
where
    T: TrainableModel,
{
    let msg = match msg_possibility {
        ClientMsgPossibility::Msg(msg) => msg,
        ClientMsgPossibility::ChannelClosed => {
            return Err(ClosedChannel);
        }
        ClientMsgPossibility::MsgError(_e) => {
            return Err(ErrorReadingFromChannel);
        }
    };
    if let ClientMsg::EarlyLeaveCluster {
        reason,
        expected_absence,
    } = msg
    {
        child_coord_tx
            .send(ChildCoordinatorTaskMsg::EarlyLeaveCluster {
                client_id,
                reason,
                expected_absence,
            })
            .unwrap();
        framed
            .send(ServerMsg::EndOfConnectionAcknowledgement)
            .await
            .unwrap();
        return Ok(Disconnected);
    };

    match (&state, msg) {
        (
            Computing,
            ClientMsg::UpdatedLikelihood {
                new_likelihood,
                delta,
                loss,
            },
        ) => {
            child_coord_tx
                .send(ChildCoordinatorTaskMsg::NewLikelihood {
                    client_id,
                    new_likelihood,
                    delta,
                    loss,
                })
                .unwrap();

            Ok(Waiting)
        }
        (ClientState::WaitingForClientFinishResponse, ClientMsg::UpdatedLikelihood { .. }) => {
            Ok(ClientState::WaitingForClientFinishResponse)
        }
        (
            ClientState::WaitingForClientFinishResponse,
            ClientMsg::FinalLeaveTraining {
                available_for_future_training,
            },
        ) => {
            child_coord_tx
                .send(ChildCoordinatorTaskMsg::FinalLeaveTraining {
                    client_id,
                    available_for_future_training,
                })
                .unwrap();
            Ok(Disconnected)
        }
        _ => {
            framed.send(get_state_err_msg()).await.unwrap();
            Err(WrongStateInput)
        }
    }
}

fn get_state_err_msg<T>() -> ServerMsg<T>
where
    T: TrainableModel,
{
    ServerMsg::Error {
        error: Some("Unexpected State Message combination".to_string()),
    }
}
