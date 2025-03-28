import sys
import jax
import numpy as np
import jax.numpy as jnp
import blackjax
import tqdm
import anesthetic
from blackjax.ns.utils import finalise
from jesi import logl_jax
import tensorflow_probability.substrates.jax as tfp
from flexknot.utils import create_theta
from common import flexknotparamnames

tfd = tfp.distributions

rng_key = jax.random.PRNGKey(0)


N = int(sys.argv[1])


prior = tfd.JointDistributionNamed(dict(
    h0rd=tfd.Uniform(jnp.array([3650.0]), jnp.array([18250.0])),
    omegam=tfd.Uniform(jnp.array([0.01]), jnp.array([0.99])),
    a=tfd.Uniform(jnp.zeros(N-2), jnp.ones(N-2)),
    w=tfd.Uniform(-3*jnp.ones(N), jnp.zeros(N)),
))

h0rd_prior = tfd.Uniform(jnp.array([3650.0]), jnp.array([18250.0]))
omegam_prior = tfd.Uniform(jnp.array([0.01]), jnp.array([0.99]))
a_prior = tfd.Uniform(jnp.zeros(N-2), jnp.ones(N-2))
w_prior = tfd.Uniform(-3*jnp.ones(N), jnp.zeros(N))


def prior_fn(x):
    prior = jnp.sum(h0rd_prior.log_prob(x['h0rd']))
    prior += jnp.sum(omegam_prior.log_prob(x['omegam']))
    prior += jnp.sum(a_prior.log_prob(x['a']))
    prior += jnp.sum(w_prior.log_prob(x['w']))
    return prior

test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)


@jax.jit
def wrapped_stepper(x, n, t):
    y = jax.tree.map(lambda x, n: x + t * n, x, n)
    i = jnp.argsort(y['a'], descending=True)
    y['a'] = jnp.take_along_axis(y['a'], i, -1)
    y['w'] = jnp.concatenate([
        y['w'][..., :1],
        jnp.take_along_axis(y['w'][..., 1:-1], i, -1),
        y['w'][..., -1:],
    ], axis=-1)
    return y


nlive = 5000
nprior = 10 * nlive
n_delete = nlive // 2
rng_key, init_key = jax.random.split(rng_key, 2)


logv = (jnp.log(18250.0 - 3650.0) + jnp.log(0.99 - 0.01)
        + (N-2)*jnp.log(1) + N*jnp.log(3))


ns = blackjax.ns.adaptive.nss(
    # logprior_fn=lambda x: -logv,
    logprior_fn=prior_fn,
    loglikelihood_fn=logl_jax,
    n_delete=n_delete,
    num_mcmc_steps=2*N*3,
    stepper=wrapped_stepper,
    ravel_fn=ravel_fn,
)


dead = []


def integrate(ns, rng_key):
    rng_key, init_key = jax.random.split(rng_key, 2)
    particles = prior.sample(seed=init_key, sample_shape=(nprior,))
    i = jnp.argsort(particles['a'], axis=-1, descending=True)
    particles['a'] = jnp.take_along_axis(particles['a'], i, -1)
    particles['w'] = jnp.concatenate([
        particles['w'][..., :1],
        jnp.take_along_axis(particles['w'][..., 1:-1], i, -1),
        particles['w'][..., -1:],
    ], axis=-1)

    logl = jax.vmap(logl_jax)(particles)
    from scipy.special import logsumexp
    print(f"logZ = {logsumexp(logl)-jnp.log(nprior)}")
    top_logl, idx = jax.lax.top_k(logl, nlive)
    state = ns.init(jax.tree.map(lambda x: x[idx], particles), top_logl)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = ns.step(subk, state)
        return (state, k), dead_point

    one_step((state, rng_key), None)
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while (not state.sampler_state.logZ_live - state.sampler_state.logZ < -3):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    return state, finalise(state, dead), particles, logl


state, final, cold_particles, cold_logl = integrate(ns, rng_key)

cold_theta = np.concatenate([cold_particles['h0rd'], cold_particles['omegam'],
                             [create_theta(a[::-1], w[::-1])
                             for a, w in zip(cold_particles['a'],
                                             cold_particles['w'])],
                             ], axis=-1)
theta = np.concatenate([final.particles['h0rd'], final.particles['omegam'],
                        [create_theta(a[::-1], w[::-1])
                        for a, w in zip(final.particles['a'],
                                        final.particles['w'])],
                        ], axis=-1)
theta = np.concatenate([cold_theta, theta], axis=0)
logl = np.concatenate([cold_logl, final.logL])
# cold points have logl_birth -ing
logl_birth = np.concatenate([jnp.full_like(cold_logl, -jnp.inf), final.logL_birth])
print(theta.shape)
print(logl.shape)
print(logl_birth.shape)

labels=[("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")] + flexknotparamnames(N)
print(labels)
labels_map = {l[0]: '$'+l[1]+'$' for l in labels}

samples = anesthetic.NestedSamples(
    data=theta,
    logL=logl,
    logL_birth=logl_birth,
    columns=[l[0] for l in labels],
    labels=labels_map,
)
samples.to_csv(f"chains/samples_{N}.csv")
print(samples)

print(f"log Z = {state.sampler_state.logZ}")
