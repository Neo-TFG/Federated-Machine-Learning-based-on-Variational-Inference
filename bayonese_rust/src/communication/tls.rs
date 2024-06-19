use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::{fs, io};

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{pki_types, ClientConfig, RootCertStore, ServerConfig};
use rustls_pemfile::certs;
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio_rustls::client::TlsStream;
use tokio_rustls::{TlsAcceptor, TlsConnector};

pub fn setup_server_tls(
    cafile: impl AsRef<Path>,
    server_certfile: impl AsRef<Path>,
    server_keyfile: impl AsRef<Path>,
) -> TlsAcceptor {
    let mut root_cert_store = RootCertStore::empty();
    root_cert_store.add_parsable_certificates(load_certs(cafile));
    let config =
        create_server_tls_config(Arc::new(root_cert_store), server_certfile, server_keyfile);
    TlsAcceptor::from(Arc::new(config))
}
pub async fn setup_client_tls(
    cafile: impl AsRef<Path>,
    certfile: impl AsRef<Path>,
    keyfile: impl AsRef<Path>,
    server_name: impl Into<String>,
    server_addr: impl ToSocketAddrs,
) -> TlsStream<TcpStream> {
    let mut root_cert_store = RootCertStore::empty();
    root_cert_store.add_parsable_certificates(load_certs(cafile));
    let config = create_client_tls_config(Arc::new(root_cert_store), certfile, keyfile);
    let connector = TlsConnector::from(Arc::new(config));
    let stream = TcpStream::connect(&server_addr)
        .await
        .expect("Error connecting to server");
    let domain = pki_types::ServerName::try_from(server_name.into())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid dnsname"))
        .unwrap()
        .to_owned();
    connector.connect(domain, stream).await.unwrap()
}
fn create_client_tls_config(
    root_cert_store: Arc<RootCertStore>,
    certfile: impl AsRef<Path>,
    keyfile: impl AsRef<Path>,
) -> ClientConfig {
    let certs = load_certs(certfile);
    let key = load_private_key(keyfile);
    ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_client_auth_cert(certs, key)
        .unwrap()
}

fn create_server_tls_config(
    root_cert_store: Arc<RootCertStore>,
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> ServerConfig {
    let cert_verifier = WebPkiClientVerifier::builder(root_cert_store)
        .build()
        .expect("No valid trust anchors provided");

    let cert = load_certs(cert_path);
    let key = load_private_key(key_path);
    ServerConfig::builder()
        .with_client_cert_verifier(cert_verifier)
        .with_single_cert(cert, key)
        .expect("Non Valid Key Cert")
}

fn load_certs(certs_path: impl AsRef<Path>) -> Vec<CertificateDer<'static>> {
    let certfile = fs::File::open(certs_path).expect("cannot open certificate file");
    let mut reader = BufReader::new(certfile);
    certs(&mut reader).map(|result| result.unwrap()).collect()
}
fn load_private_key(filename: impl AsRef<Path>) -> PrivateKeyDer<'static> {
    let keyfile = fs::File::open(&filename).expect("cannot open private key file");
    let mut reader = BufReader::new(keyfile);

    loop {
        match rustls_pemfile::read_one(&mut reader).expect("cannot parse private key .pem file") {
            Some(rustls_pemfile::Item::Pkcs1Key(key)) => return key.into(),
            Some(rustls_pemfile::Item::Pkcs8Key(key)) => return key.into(),
            Some(rustls_pemfile::Item::Sec1Key(key)) => return key.into(),
            None => break,
            _ => {}
        }
    }

    panic!(
        "no keys found in {:?} (encrypted keys not supported)",
        filename.as_ref()
    );
}
pub struct WrappedCertificates();
pub fn get_certificate_from_client<T>(
    stream: &tokio_rustls::server::TlsStream<T>,
) -> Vec<CertificateDer<'static>> {
    let (_, conn) = stream.get_ref();
    conn.peer_certificates().unwrap().to_owned()
}
pub fn get_certificate_from_server<T>(stream: &mut TlsStream<T>) -> Vec<CertificateDer<'static>> {
    let (_, conn) = stream.get_ref();
    conn.peer_certificates().unwrap().to_owned()
}
