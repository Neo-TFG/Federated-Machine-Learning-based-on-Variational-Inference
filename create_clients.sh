#!/bin/bash

for ((i=$1;i<=$2;i++)); do
    python ./client_python/client_example.py "$i" &
done