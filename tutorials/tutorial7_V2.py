"""
This example sets up a water tank with PCM material
as the bed material. The tank has a diameter of 10 cm
and a height of 1 m. The initial temperature of the
water is 20℃. The water is heated up to 80℃ at the
bottom. Heat transfer from the fluid phase to the bed
phase is modelled with a constant heat transfer coefficient
of 200 W/m²K. The simulation time is 100 minutes, and the
output is saved every 5 minutes.
"""

import openterrace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    ot = openterrace.Simulate(t_start=0, t_end=0.5, dt=0.5)
    time_steps = np.arange(ot.t_start, ot.t_end + ot.dt, ot.dt)
    fluid = ot.create_phase(n=100, type='fluid')
    fluid.select_substance(substance='water')
    fluid.select_domain_shape(domain='cylinder_1d', D=0.1, H=1)
    fluid.select_porosity(phi=0.4)
    fluid.select_schemes(diff='central_difference_1d', conv='upwind_1d')
    fluid.select_initial_conditions(T=273.15+20)
    fluid.select_massflow(mdot=0.01)
    fluid.select_bc(bc_type='fixed_value',
                    parameter='T',
                    position=(slice(None, None, None), 0),
                    value=273.15+80)
    fluid.select_bc(bc_type='zero_gradient',
                    parameter='T',
                    position=(slice(None, None, None), -1))
    fluid.select_output(times=time_steps)
    # fluid.select_output(times=[0, 30, 60, 90, 120, 150, 180, 210,
    #                     240, 270, 300, 600, 900,
    #                     1800, 3600, 5400, 6000])

    bed = ot.create_phase(n=20, n_other=100, type='bed')
    bed.select_substance('ATS58')
    bed.select_domain_shape(domain='hollow_sphere_1d', Rinner=0.005, Router=0.025)
    bed.select_schemes(diff='central_difference_1d')
    bed.select_initial_conditions(T=273.15+20)
    bed.select_bc(bc_type='zero_gradient',
                parameter='T',
                position=(slice(None, None, None), 0))
    bed.select_bc(bc_type='zero_gradient',
                parameter='T',
                position=(slice(None, None, None), -1))
    bed.select_output(times=range(0,600+300,300))

    ot.select_coupling(fluid_phase=0, bed_phase=1, h_exp='constant', h_value=200)
    ot.run_simulation()

    # Save the data to a csv file
    Simulation_data = pd.DataFrame(fluid.data.T[:,0,:], index=fluid.data.time, columns=fluid.node_pos)
    # print(fluid.data.T.shape)
    Simulation_data.to_csv('Simulation_data.csv')
    # fluid.data.to_csv('fluid_data.csv')
    # bed.data.to_csv('bed_data.csv')



    # plt.plot(fluid.node_pos,fluid.data.T[:,0,:].T-273.15, label=fluid.data.time/60)
    # plt.legend(title='Simulation time (min)')
    # plt.show()
    # plt.xlabel(u'Cylinder position (m)')
    # plt.ylabel(u'Temperature (℃)')
    # plt.grid()
    # plt.grid(which='major', color='#DDDDDD', linewidth=1)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.8)
    # plt.minorticks_on()
    # plt.savefig('ot_plot_tutorial7.svg')

if __name__ == "__main__":
    main()