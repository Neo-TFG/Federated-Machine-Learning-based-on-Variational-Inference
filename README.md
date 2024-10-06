Repository containing the code and data related to the thesis Federated Machine Learning based on Variational Inference.

bayonese is the general name of the library created in the thesis.

The directories contain:
- bayonese_rust: code of the general library.
- bayonese_server_example: code for an example of a server and a bash script to run it.
- certs_tfg: contains the certificates used for the experiments.
- data: contains the data obtained from the experiments.
- client_python: contains both the code of the python interface as an example client.

To execute a test you must first setup the python library, check in [Pyo3's webpage](https://pyo3.rs/v0.21.2/), it mainly boils down to installing maturin in a virtual environment and executing maturin develop, but depending on the setup it could vary.

Once that is done, first execute the example_run.sh bash script from inside its directory, and then execute the create_clients.sh from the root of the directory. This last script has two arguments, the initial client number and final client number, both inclusive. Right now there are only 10 certificates uploaded so the maximum number is 10, and the lowest is 1, but more can be added if desired.

If one wants to add different distributions to test, it should go into the lib.rs inside client_python and implement them there. Once that is done, that type should be wrapped using the compatibility macro.
Then this type can be used from both the client and the server. Do not forget to call maturin after making changes that are related to Python.



