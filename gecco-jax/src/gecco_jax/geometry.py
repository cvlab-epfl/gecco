from functools import partial

import jax.numpy as jnp
from torch_dimcheck import dimchecked, A

@dimchecked
def distance_matrix(
    a: A['N D'],
    b: A['M D'],
    squared: bool = False,
) -> A['N M']:
    aa = jnp.einsum('nd,nd->n', a, a)
    bb = jnp.einsum('md,md->m', b, b)
    ab = jnp.einsum('nd,md->nm', a, b)

    dist_sqr = aa[:, None] + bb[None, :] - 2 * ab
    # protection against numerical errors resulting in NaN
    dist_sqr = jnp.maximum(dist_sqr, 0.)

    if squared:
        return dist_sqr
    else:
        return jnp.sqrt(dist_sqr)

@partial(jnp.vectorize, signature='(a)->(b)')
@dimchecked
def convert_points_to_homogeneous(pt: A['a']) -> A['b']:
    return jnp.concatenate([
        pt,
        jnp.array([1.]),
    ], axis=-1)

@dimchecked
def convert_points_from_homogeneous(
    pt: A['B* a'],
    eps: float = 1e-8,
) -> A['B* b']:

    @partial(jnp.vectorize, signature='(a)->(b)', excluded=(1, ))
    @dimchecked
    def inner(pt: A['a'], eps: float) -> A['b']:
        z = pt[-1:]
        mask = jnp.abs(z) > eps
        scale = jnp.where(mask, 1. / (z + eps), 1.)
        return scale * pt[:-1]

    return inner(pt, eps)

#@dimchecked
def unproject_points(
    xy: A['2'],
    depth: A[''],
    camera_matrix: A['3 3'],
    normalized: bool = True,
) -> A['3']:

    @partial(jnp.vectorize, signature='(a),(),(b,c)->(d)', excluded=(3, ))
    @dimchecked
    def inner(
        xy: A['2'],
        depth: A[''],
        camera_matrix: A['3 3'],
        normalized: bool,
    ) -> A['3']:
        uvw = convert_points_to_homogeneous(xy)
        xyw = jnp.einsum('e,ae->a', uvw, jnp.linalg.inv(camera_matrix))
        if normalized:
            xyw = xyw / jnp.linalg.norm(xyw)
        return xyw * depth
    
    return inner(xy, depth, camera_matrix, normalized)

@partial(jnp.vectorize, signature='(a),(b,c)->(d)')
@dimchecked
def project_points(xyz: A['3'], camera_matrix: A['3 3']) -> A['2']:
    xyw = jnp.einsum('e,ae->a', xyz, camera_matrix) 
    return convert_points_from_homogeneous(xyw)