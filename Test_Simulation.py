import openterrace
import matplotlib.pyplot as plt

n= 10
m= 5

# initalize Simulatino
sim = openterrace.Simulate(t_start=0, t_end=3600, dt=1)

# Creating Phases
fluid = sim.create_phase(n=n,n_other=m, type='fluid')
bed = sim.create_phase(n=m, n_other=n, type='bed')

#Defining materials
fluid.select_substance('air')
bed.select_substance('swedish_diabase')

# Define Domain
fluid.select_domain_shape('cylinder_1d',D=1, H=5)
bed.select_domain_shape('sphere_1d', R=0.05)

bed.select_porosity(phi=0.5)

fluid.select_schemes(diff='central_difference_1d', conv='upwind_1d')
fluid.select_initial_conditions(T=273.15+20)
fluid.select_massflow(mdot=0.5)

# Boundary conditions
fluid.select_bc(bc_type='fixed_value',
                    parameter='T',
                    position=(slice(None, None, None), 0),
                    value=273.15+80)
fluid.select_bc(bc_type='zero_gradient',
                    parameter='T',
                    position=(slice(None, None, None), -1))

bed.select_bc(bc_type='zero_gradient',
                parameter='T',
                position=(slice(None, None, None), 0))
bed.select_bc(bc_type='zero_gradient',
                parameter='T',
                position=(slice(None, None, None), -1))


fluid.select_output(times=range(0, 60, 3600))
bed.select_output(times=range(0, 60, 3600))

ot.select_coupling(fluid_phase=0, bed_phase=1, h_exp='constant', h_value=200)

