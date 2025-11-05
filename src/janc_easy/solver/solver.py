import jax.numpy as jnp
from jax import jit,vmap,pmap
from ..solver import aux_func
from .flux import weno5,HLLC
from ..thermodynamics import thermo
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from ..parallel import grid_partion
from functools import partial

def CFL(U,dx,dy,cfl=0.50):
    _,u,v,_,a = aux_func.U_to_prim(U)
    sx = jnp.abs(u) + a
    sy = jnp.abs(v) + a
    dt = cfl*jnp.min(1/(sx/dx + sy/dy))
    return dt

def set_solver(thermo_set, boundary_set, source_set = None, nondim_set = None, solver_mode='base',is_parallel=False,parallel_set=None):
    thermo.set_thermo(thermo_set,nondim_set)
    boundary.set_boundary(boundary_set)
    aux_func.set_source_terms(source_set)
    
    if is_parallel:
        boundary_conditions = parallel_boundary.boundary_conditions
    else:
        boundary_conditions = boundary.boundary_conditions

    
    def rhs(U,dx,dy,theta=None):
        U_with_ghost = boundary_conditions(U,theta)
        physical_rhs = weno5(U_with_ghost,dx,dy) + aux_func.source_terms(U, theta)
        return physical_rhs

    def advance_flux(U,dx,dy,dt,theta=None):
        U1 = U + dt * rhs(U,dx,dy,theta)
        U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,dx,dy,theta))
        U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,dx,dy,theta))
        return U3
  
    @jit    
    def advance_one_step(field,dx,dy,theta=None,time=0.0):
        dt = CFL(field,dx,dy)
        field = advance_flux(field,dx,dy,dt,theta)
        time = time + dt
        return field, time

    if is_parallel:
        if parallel_set is not None:
            assert 'theta_pmap_axis' in parallel_set, "You should define the pmap axes of theta in your setting dict with key 'theta_pmap_axis'."
            theta_pmap_axis = parallel_set['theta_pmap_axis']
            if solver_mode == 'base':
                advance_one_step = pmap(advance_one_step,axis_name='x',in_axes=(0,None,None,None,theta_pmap_axis))
        else:
            if solver_mode == 'base':
                advance_one_step = pmap(advance_one_step,axis_name='x',in_axes=(0,None,None,None))
        
    print('solver is initialized successfully!')
    return advance_one_step,rhs
        

    















