use std::path::PathBuf;

use clap::Parser;
use tokio::net::TcpListener;

use bayonese::communication::tls::setup_server_tls;
use bayonese::server::create_server;
use bayonese::{SerialisationFormat, TrainingMode};
use bayonese_py::Normal;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Port number to host the TCP listener, 0 means
    /// OS assigned
    #[arg(short, long)]
    port: u16,
    /// File containing the CA root certificates for the server
    #[arg(short = 'C', long)]
    cafile: PathBuf,
    /// File containing the server's private keys
    #[arg(short, long)]
    keyfile: PathBuf,
    /// File containing the server's certificate
    #[arg(short = 'c', long)]
    certfile: PathBuf,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // Choose the address you want to host the server in
    let addr = ("192.168.1.23", args.port);
    let ca_cert_path = args.cafile;

    let key_path = args.keyfile;
    let cert_path = args.certfile;

    let acceptor = setup_server_tls(ca_cert_path, cert_path, key_path);

    let listener = TcpListener::bind(addr).await?;
    let format = SerialisationFormat::Bincode;
    // Select Multivariate or Normal depending on the training or
    // a different distribution that implements the necessary trait
    let prior = Normal::default();
    println!("Listening on: {:?}", addr);
    let posterior =
        create_server::<Normal>(acceptor, listener, prior, TrainingMode::Synchronous, format).await;
    println!("Final posterior is {:?}", posterior);
    Ok(())
}
