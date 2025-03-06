"""
Simulation wit a packed bed of swedish_diabase spheres. The fluid phase is air. The charging temperature is 700℃, and the discharging temperature is 90°C (both at the bottom). The bed is 10 m high and has a diameter of 1 m. The bed is exposed to a convection heat transfer with a heat transfer coefficient of 10 W/m²K. The simulation time is 600 minutes, and the output is saved every 10 seconds.
"""

import openterrace
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.colors as pc

D = 2                 # Diameter of the bed
D_b = 0.05              # Diameter of the spheres
H = 10                  # Height of the bed
T_ini = 273.15+20      # Initial temperature
T_charge = 273.15+90   # Charging temperature
T_discharge = 273.15+90 # Discharging temperature
T_amb = 273.15+20       # Ambient temperature
t_start = 0             # Start time
t_end = 3600*10          # End time
dt = 1                  # Time step
U = 20                  # Heat transfer coefficient between the phases
n = 101                 # Number of nodes fluid
m = 10                  # Number of nodes bed
time_steps = np.arange(t_start, t_end + dt, dt)
mass_flow_base = 1    # Base mass flow rate
massflow_values = mass_flow_base*np.sin(np.pi/3600*time_steps - np.pi/2)+mass_flow_base
massflow_data = np.column_stack((time_steps, massflow_values))
phi = 0.2               # Porosity


sim = openterrace.Simulate(t_end=t_end, dt=dt)

# Define the fluid phase
fluid = sim.create_phase(n=n, type='fluid')
fluid.select_substance('water')
fluid.select_domain_shape(domain='cylinder_1d', D=D, H=H)
fluid.select_porosity(phi=phi)
fluid.select_schemes(diff='central_difference_1d', conv='upwind_1d')
fluid.select_initial_conditions(T=T_ini)
fluid.select_massflow(mdot=massflow_data)
# fluid.select_massflow(mdot=1)
fluid.select_bc(bc_type='fixed_value',
                parameter='T',
                position=(slice(None, None, None), 0),
                value=T_charge)
fluid.select_bc(bc_type='zero_gradient',
                parameter='T',
                position=(slice(None, None, None), -1))
fluid.select_output(times=time_steps)

# Define the bed phase
bed = sim.create_phase(n=m, n_other=n, type='bed')
bed.select_substance('swedish_diabase')
bed.select_domain_shape(domain='sphere_1d', R=D_b/2)
bed.select_schemes(diff='central_difference_1d')
bed.select_initial_conditions(T=T_ini)
bed.select_bc(bc_type='zero_gradient',
               parameter='T',
               position=(slice(None, None, None), 0))
bed.select_bc(bc_type='zero_gradient',
               parameter='T',
               position=(slice(None, None, None), -1))
bed.select_output(times=time_steps)

sim.select_coupling(fluid_phase=0, bed_phase=1, h_exp='constant', h_value=U)
sim.run_simulation()

# Save the results in a pandas DataFrame
# df_fluid = pd.DataFrame(fluid.data.T[:,0,:].T-273.15, columns=fluid.node_pos, index=fluid.data.time)
df_fluid = pd.DataFrame(fluid.data.T[:,0,:].T-273.15, index=fluid.node_pos, columns=fluid.data.time)
df_bed_center = pd.DataFrame(bed.data.T[:,:,0].T-273.15, index=fluid.node_pos, columns=bed.data.time)
df_bed_outer = pd.DataFrame(bed.data.T[:,:,-1].T-273.15, index=fluid.node_pos, columns=bed.data.time)

# Transpose the DataFrames
df_fluid = df_fluid.T
df_bed_center = df_bed_center.T
df_bed_center = df_bed_center.T

# Add the time column
df_fluid['time'] = df_fluid.index
df_bed_center['time'] = df_bed_center.index
df_bed_outer['time'] = df_bed_outer.index

# drop time 
df_fluid = df_fluid.drop(columns='time')
df_bed_center = df_bed_center.drop(columns='time')
df_bed_outer = df_bed_outer.drop(columns='time')


# Name columns
df_fluid.columns = [f'{col} m' if col != 'time' else col for col in df_fluid.columns]
df_bed_center.columns = [f'{col} m' if col != 'time' else col for col in df_bed_center.columns]
df_bed_outer.columns = [f'{col} m' if col != 'time' else col for col in df_bed_outer.columns]


# Save results as a csv
# df_fluid.to_csv('fluid_phase.csv')
# df_bed.to_csv('bed_phase.csv')

xx = 1800

# Determine how many lines will be plotted
indices_to_plot = [i for i in range(df_fluid.shape[0]) if i % xx == 0]
num_lines = len(indices_to_plot)  # Actual number of lines being plotted

# Generate evenly spaced colors from the colormap
colors = pc.sample_colorscale('Viridis', np.linspace(0, 1, num_lines))

fig = go.Figure()

color_idx = 0  # Separate color index tracker
for i, (index, row) in enumerate(df_fluid.iterrows()):
    if i % xx == 0:  # Only every xx-th entry
        fig.add_trace(go.Scatter(
            x=df_fluid.columns, 
            y=row, 
            mode='lines', 
            name=f'{index} s', 
            line=dict(color=colors[color_idx])  # Ensure valid indexing
        ))
        color_idx += 1  # Only increment for plotted lines

fig.update_layout(
    title='Fluid phase',
    xaxis_title='Height (m)',
    yaxis_title='Temperature (°C)'
)

# X-axis labels only at full numbers
fig.update_xaxes(tickvals=df_fluid.columns[0::10])

# Save the plot
fig.write_html('fluid_phase.html')


# Plot df_bed_center same as df_fluid

fig = go.Figure()

color_idx = 0  # Separate color index tracker
for i, (index, row) in enumerate(df_bed_center.iterrows()):
    if i % xx == 0:  # Only every xx-th entry
        fig.add_trace(go.Scatter(
            x=df_bed_center.columns, 
            y=row, 
            mode='lines', 
            name=f'{index} s', 
            line=dict(color=colors[color_idx])  # Ensure valid indexing
        ))
        color_idx += 1

fig.update_layout(
    title='Bed phase center',
    xaxis_title='Height (m)',
    yaxis_title='Temperature (°C)'
)  

# X-axis labels only at full numbers
fig.update_xaxes(tickvals=df_bed_center.columns[0::10])

# Save the plot
fig.write_html('bed_phase_center.html')
