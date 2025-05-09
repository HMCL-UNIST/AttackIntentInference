import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as jrandom
from functools import partial

earth_radius = 6371000.0  
g = 9.81  

@jit
def get_density_jax(altitude):
    return 1.225 * jnp.exp(-altitude / 10000)
@jit
def hgv_dynamics_fixed_params_jax(state, ui, uj, uk):
    r, lambda_, phi, v, theta, psi = state
    altitude = r - earth_radius
    rho = get_density_jax(altitude)

    dr = v * jnp.sin(theta)
    dlambda = v * jnp.cos(theta) * jnp.sin(psi) / (r * jnp.cos(phi))
    dphi = v * jnp.cos(theta) * jnp.cos(psi) / r
    dv = -0.5 * rho * v**2 * ui - g * jnp.sin(theta)
    dtheta = (0.5 * rho * v * uj * jnp.cos(uk)) - (g * jnp.cos(theta)) / v + (v * jnp.cos(theta)) / r
    dpsi = (0.5 * rho * v * uj * jnp.sin(uk))/ (jnp.cos(theta)) + v * jnp.cos(theta) * jnp.sin(psi) * jnp.tan(phi) / r

    return jnp.array([dr, dlambda, dphi, dv, dtheta, dpsi])

@partial(jit, static_argnums=(4,))
def rollout_until_ground_hit_single_scan(state, ui, uj, uk, dt):
    max_steps = 700 

    def dynamics_fn(s, _):
        ds = hgv_dynamics_fixed_params_jax(s, ui, uj, uk)
        next_s = s + dt * ds
        return next_s, next_s

    final_state, traj = lax.scan(dynamics_fn, state, None, length=max_steps)
    altitudes = traj[:, 0] - earth_radius
    impact_idx = jnp.argmax(altitudes <= 0.0)

    return traj, impact_idx
    

rollout_until_ground_hit = vmap(
    rollout_until_ground_hit_single_scan,
    in_axes=(0, 0, 0, 0, None)
)
