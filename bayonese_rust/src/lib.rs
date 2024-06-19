use futures::{SinkExt, StreamExt};
use std::io;
use std::io::BufRead;
use tokio::net::TcpStream;
use tokio_serde::formats::{Bincode, Cbor, Json, MessagePack};
use tokio_serde::Framed;
use tokio_util::codec::LengthDelimitedCodec;
use x509_parser::prelude::FromDer;

use crate::communication::protocol::{ClientMsg, ServerMsg};
use crate::distributions::TrainableModel;

pub mod client;
pub mod communication;
pub mod distributions;
pub mod server;

#[derive(Copy, Clone, Debug)]
pub enum TrainingMode {
    Sequential,
    Synchronous,
    Asynchronous,
}
#[derive(Copy, Clone, Debug)]
pub enum SerialisationFormat {
    Json,
    Bincode,
    Cbor,
    MessagePack,
}

macro_rules! generate_frame {
    ($receive_type: ident, $send_type:ident, $name:ident, $stream: ty, $msg_pos: ident) => {
        enum $name<T>
        where
            T: TrainableModel,
        {
            Json(
                tokio_serde::Framed<
                    tokio_util::codec::Framed<$stream, LengthDelimitedCodec>,
                    $receive_type<T>,
                    $send_type<T>,
                    Json<$receive_type<T>, $send_type<T>>,
                >,
            ),
            Bincode(
                tokio_serde::Framed<
                    tokio_util::codec::Framed<$stream, LengthDelimitedCodec>,
                    $receive_type<T>,
                    $send_type<T>,
                    Bincode<$receive_type<T>, $send_type<T>>,
                >,
            ),
            Cbor(
                tokio_serde::Framed<
                    tokio_util::codec::Framed<$stream, LengthDelimitedCodec>,
                    $receive_type<T>,
                    $send_type<T>,
                    Cbor<$receive_type<T>, $send_type<T>>,
                >,
            ),
            MessagePack(
                tokio_serde::Framed<
                    tokio_util::codec::Framed<$stream, LengthDelimitedCodec>,
                    $receive_type<T>,
                    $send_type<T>,
                    MessagePack<$receive_type<T>, $send_type<T>>,
                >,
            ),
        }
        impl<T> $name<T>
        where
            T: TrainableModel,
        {
            fn new(
                format: SerialisationFormat,
                inner: tokio_util::codec::Framed<$stream, LengthDelimitedCodec>,
            ) -> Self {
                match format {
                    SerialisationFormat::Json => Self::Json(Framed::new(inner, Json::default())),
                    SerialisationFormat::Bincode => {
                        Self::Bincode(Framed::new(inner, Bincode::default()))
                    }
                    SerialisationFormat::Cbor => Self::Cbor(Framed::new(inner, Cbor::default())),
                    SerialisationFormat::MessagePack => {
                        Self::MessagePack(Framed::new(inner, MessagePack::default()))
                    }
                }
            }
            async fn next(&mut self) -> Option<Result<$receive_type<T>, std::io::Error>> {
                match self {
                    Self::Json(c) => c.next().await,
                    Self::Bincode(c) => c.next().await,
                    Self::Cbor(c) => c.next().await,
                    Self::MessagePack(c) => c.next().await,
                }
            }
            async fn send(&mut self, msg: $send_type<T>) -> Result<(), std::io::Error> {
                match self {
                    Self::Json(c) => c.send(msg).await,
                    Self::Bincode(c) => c.send(msg).await,
                    Self::Cbor(c) => c.send(msg).await,
                    Self::MessagePack(c) => c.send(msg).await,
                }
            }
            async fn get_message(&mut self) -> $msg_pos<T> {
                let Some(msg) = self.next().await else {
                    return $msg_pos::ChannelClosed;
                };
                match msg {
                    Ok(msg) => $msg_pos::Msg(msg),
                    Err(s) => {
                        if let io::ErrorKind::UnexpectedEof = s.kind() {
                            return $msg_pos::ChannelClosed;
                        }
                        $msg_pos::MsgError(s)
                    }
                }
            }
        }
        #[derive(Debug)]
        enum $msg_pos<T>
        where
            T: TrainableModel,
        {
            ChannelClosed,
            MsgError(std::io::Error),
            Msg($receive_type<T>),
        }
    };
}
generate_frame!(
    ClientMsg,
    ServerMsg,
    ServerBayoneseFramed,
    tokio_rustls::server::TlsStream<TcpStream>,
    ClientMsgPossibility
);
generate_frame!(
    ServerMsg,
    ClientMsg,
    ClientBayoneseFramed,
    tokio_rustls::client::TlsStream<TcpStream>,
    ServerMsgPossibility
);
