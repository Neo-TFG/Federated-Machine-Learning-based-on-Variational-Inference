import bayonese_py as b
import numpy as np
import sys
from matplotlib import pyplot as plt
from pyro.distributions import constraints
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import math
import pyro
import time
import random


def model(obs, mu_prior, sigma_prior):
    mu = pyro.sample("mu", dist.Normal(mu_prior, sigma_prior))
    with pyro.plate("samples", len(obs)):
        pyro.sample("test", dist.Normal(mu, torch.tensor(1.0)),
                    obs=obs)


def guide(obs, mu_prior, sigma_prior):
    mu_mean = pyro.param('mu_mean', torch.tensor(mu_prior))
    mu_scale = pyro.param('mu_scale', torch.tensor(sigma_prior),
                          constraint=constraints.positive)
    pyro.sample("mu", dist.Normal(mu_mean, mu_scale))


pyro.clear_param_store()

adam = pyro.optim.Adam({"lr": 0.02, "betas": (0.90, 0.999)})
elbo = pyro.infer.Trace_ELBO(num_particles=10)
svi = pyro.infer.SVI(model, guide, adam, elbo)

num_client = int(sys.argv[1])
print(f"Client number {num_client}")
np.random.seed(num_client)

obs = torch.tensor(np.random.normal(loc=5., scale=1.0, size=1000))

format = b.PySerialisationFormat.Bincode
bayon = b.setup_tls_Normal("./certs_tfg/Root_CA_TFG.crt", f"./certs_tfg/Client{num_client}.crt",
                           f"./certs_tfg/Client{num_client}.pem",
                           "Server",
                           "192.168.1.23:8081", format)

bayon.join_cluster(1000)
pos, finished = bayon.wait_for_posterior()
losses = []

while not finished:
    pos_without = bayon.extract_last_likelihood(pos)
    print("Without", pos_without.mean, pos_without.variance)

    scale = math.sqrt(abs(pos_without.variance))
    for step in range(1000):  # Consider running for more steps.
        loss = svi.step(obs, pos_without.mean, scale)
        losses.append(loss)
        if step % 100 == 0:
            print("Elbo loss: {}".format(loss))
    new_mean = pyro.param("mu_mean").item()
    new_scale = pyro.param("mu_scale").item()
    print(f"Pyro mean {new_mean}, Pyro scale {new_scale}")
    new_pos = b.Normal(new_mean, new_scale ** 2)
    new_likelihood, delta = bayon.calculate_new_likelihood_and_delta(new_pos, pos)
    print("likelihood", new_likelihood.mean, new_likelihood.variance)
    print("Delta", delta.mean, delta.variance)
    bayon.send_updated_likelihood(delta, new_likelihood, losses[-1])
    pos, finished = bayon.wait_for_posterior()

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
plt.show(block=True)
