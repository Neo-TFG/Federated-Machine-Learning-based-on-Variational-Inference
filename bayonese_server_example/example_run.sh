#!/bin/bash
cargo run --release -- --port 8081 --cafile ../certs_tfg/Root_CA_TFG.crt --keyfile ../certs_tfg/Server_Certificate.pem --certfile ../certs_tfg/Server_Certificate.crt