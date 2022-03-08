# 2021.04.13 written by Tae Hoon Oh, E-mail: rozenk30@gmail.com
# Penicillin plant dynamics, see the reference:
# "The development of an industrial-scale fed-batch fermentation simulation (2015)"

import numpy as np
import casadi as ca

import UtilityFunctions


# state (29): Time, A_0, A_1, A_3, A_4, Integral_X, S, P, V, T, H, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, DO,
#             DCO2, viscosity, PAA, O2, CO2, Ni
# input (11): F_S, F_oil, F_PAA, F_a, F_b, F_w, F_g, F_c, F_h, Pressure, NH3
# RPM, F_dis are fixed


class PlantDynamicsEquation(object):
    def __init__(self, seed, ini_perturb, para_perturb):
        # Seed
        self.seed = seed
        np.random.seed(self.seed)

        self.state_dimension = 29
        self.input_dimension = 11

        self.initial_state = np.array([0., 0.1667, 0.3333, 0., 0., 0., 1., 0., 58000., 298., 10 ** (-6.5), 2. * 10 ** 6,
                                       4. * 10 ** 4, 650., 7., 0.05, 0., 0., 0., 0., 0., 0., 15., 0.25, 4., 1400., 0.2,
                                       0.1, 1800.])
        # Parameter perturbation
        self.ini_perturb_coefficient = np.array([0., 0.02, 0.03, 0., 0., 0., 0.5, 0., 1000., 0., 0., 0., 0., 0., 0., 0,
                                                 0., 0., 0., 0., 0., 0., 1., 0.01, 0.1, 100., 0.02, 0.01, 500.])
        if ini_perturb:
            self.initial_state = self.initial_state \
                                 + self.ini_perturb_coefficient * np.random.uniform(-1, 1, self.state_dimension)

        self.initial_input = np.array([8., 22., 5., 0., 24.6, 0., 0.5, 0., 0., 0.6, 0.])
        self.initial_time = 0.
        self.terminal_time = 230.  # +-25 terminal time
        self.time_interval = 0.5  # hr
        self.horizon_length = int((self.terminal_time - self.initial_time) / self.time_interval)  # episode length

        self.xmin = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 273., 10 ** (-7), 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.xmax = np.array(
            [230., 100., 100., 100., 100., 50000., 25., 100., 110000., 313., 10 ** (-6), 10 ** 15, 10 ** 15,
             10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 20., 20., 200.,
             10000, 1., 10., 10000])
        self.umin = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.umax = np.array([300., 300., 100., 10., 10., 100., 10., 10000., 10000., 2., 500.])
        #  y = ax + b, a = 1./(xmax - xmin) , dydt = a*dxdt = af(x) = af(x)
        self.scale_grad = 1. / (self.xmax - self.xmin)

        # Parameter perturbation
        if para_perturb:
            self.para_perturb_flag = 0.3
        else:
            self.para_perturb_flag = 0.

        opt_sub_con_for_peni_bias = self.para_perturb_flag * 0.001 * (np.random.uniform(-1, 1, 1))
        opt_sub_con_for_peni_perturb = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        p_mu_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        p_k_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        p_i_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        x_mu_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        x_k_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        a_mu_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        s_consumption_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)
        sugar_perturbation = self.para_perturb_flag * self.truncated_normal_distribution(0, 1, -1, 1)

        # Possible additional perturbation
        std_of_penicillin_inhibition_curve_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)
        substrate_maintenance_k2_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)
        gamma2_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)
        k_s_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)
        s_p_consumption_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)
        k1a_perturbation = 0 * self.truncated_normal_distribution(0, 1, -1, 1)

        # System dynamics parameters (structured model)
        self.branching_mu = 0.13 + 0.03 * x_mu_perturbation  # 1/hr # 0.105 in paper # matlab use 0.4*extension mu
        self.branching_k0 = 0.05 + 0.005 * x_k_perturbation  # g S / L
        self.extension_mu = 0.32 + 0.07 * x_mu_perturbation  # 1/hr # 0.25 +- 0.01 in paper
        self.extension_k0 = 0.009  # g S / L # 0.05 in paper
        self.production_mu = 0.041 + 0.03 * p_mu_perturbation  # 0.00125*P_mu_perturbation # modified by 2015
        self.production_k0 = 0.0002 + 0.00002 * p_k_perturbation  # g S / L
        self.production_inhibition_k0 = 0.002 + 0.0002 * p_i_perturbation  # g S / L
        self.hydrolysis_mu = 0.003  # 1/hr
        self.autolysis_mu = 0.0035 + 0.0008 * a_mu_perturbation  # +- 0.0016 1/hr
        self.density_cytoplasm = 0.35  # g/mL
        self.substrate_maintenance_k1 = 0.05  # g S / L
        self.substrate_maintenance_k2 = 0.3 + 0.03 * substrate_maintenance_k2_perturbation  # +- 0.06 g S / L
        self.gamma_2 = 1.71 * 10 ** (-4) + 0.1 * 10 ** (-4) * gamma2_perturbation  # +-1.3*10**(-4) g S / L hr
        self.density_degeneration = 0.18  # g / mL

        # Parameter related to the vacuole formation,
        # Data from the paper in 1994 is missing, use parameter value from the matlab
        self.r_0 = 1.5 * 10 ** (-4)
        self.d_0 = 1.0 * 10 ** (-4)
        self.d_r = 0.75 * 10 ** (-4)
        self.k_s = 3.22 * 10 ** (-5) + 0.05 * 10 ** (-5) * k_s_perturbation  # +- 0.29*(10)**(-5) cm / hr
        self.diffusion_coefficient = 2.66 * 10 ** (-11)  # +- 0.65*(10)**(-11) cm2 / hr # do not change
        self.evaporate_coefficient = 5.24 * 10 ** (-4)  # L / hr matlab use 5.24*(10)**(-4)
        self.t0 = 273  # K
        self.tv = 373  # K
        self.area_of_wall = 105  # m2
        self.concentration_of_acid_base = 0.033  # mol / L
        self.r_1 = self.r_0 + self.d_r
        self.r_2 = self.r_1 + self.d_r
        self.r_3 = self.r_2 + self.d_r
        self.r_4 = self.r_3 + self.d_r
        self.r_5 = self.r_4 + self.d_r
        self.r_6 = self.r_5 + self.d_r
        self.r_7 = self.r_6 + self.d_r
        self.r_8 = self.r_7 + self.d_r
        self.r_9 = self.r_8 + self.d_r
        self.r_max = self.r_9 + self.d_r

        # System dynamics parameters - 2015 paper
        self.biomass_substrate_yield_coefficient = 1.85 + 0.1 * s_consumption_perturbation  # g/g
        self.penicillin_substrate_yield_coefficient = 0.9 + 0.05 * s_p_consumption_perturbation  # g/g
        self.substrate_maintenance_term = 0.029  # g/g hr
        self.oil_feed_concentration = 1000  # g/L
        self.sugar_feed_concentration = 600 + 20 * sugar_perturbation  # g/L
        self.biomass_oxygen_yield_coefficient = 650  # mg(O2)/g(X)
        self.penicillin_oxygen_yield_coefficient = 160  # mg(O2)/g(X)
        self.maintenance_oxygen_coefficient = 17.5  # mg(O2)/g(X)
        self.k1a_alpha = 85 + 8 * k1a_perturbation  # +- 14
        self.k1a_a = 0.38
        self.k1a_b = 0.34
        self.k1a_c = -0.38
        self.k1a_d = 0.25
        self.henry_o2 = 0.0251  # bar L / mg
        self.number_of_impellers = 3
        self.tank_radius = 2.1  # m
        self.impeller_radius = 0.85  # m
        self.gassed_to_ungassed_power_ratio = 0.5  # 0.4 - 0.6
        self.power_number = 5
        self.gas_hold_up = 0.1
        self.gravitational_constant = 9.81  # m/s2
        self.universal_gas_constant = 8.314  # J / K mol
        self.peni_cri_do_level = 0.3  # 30%
        self.biomass_cri_do_level = 0.1  # 10%
        # self.inhibition_constant = 1  # delete since it is 1
        self.temperature_substrate = 288
        self.temperature_cold_water = 288
        self.temperature_hot_water = 333
        self.temperature_air = 290
        self.specific_heat_capacity_substrate = 5.8  # kJ / kg K
        self.specific_heat_capacity_water = 4.18  # kJ / kg K
        self.specific_heat_capacity_broth = 4.18  # KJ / kg K
        self.heat_evaporation = 2430.7  # KJ / kg
        self.vessel_wall_heat_transfer_coefficient = 36  # kW m2
        self.area_of_cooling_coils = 105  # m2
        self.arrhenius_constant_for_cell_growth = 450  # J g X
        self.arrhenius_constant_for_cell_death = 0.25 * 10 ** 30  # J g X
        self.activation_energy_for_cell_growth = 1.488 * 10 ** 4  # J / mol
        self.activation_energy_for_cell_death = 1.7325 * 10 ** 5  # J / mol
        self.heat_yield_coefficients_biomass = 25  # KJ / g
        self.heat_yield_coefficients_penicillin = 25  # KJ / g
        self.acid_inlet_concentration = 0.033  # mol / L
        self.base_inlet_concentration = 0.033  # mol / L
        self.hydrogen_ion_production_term = 3.25 * 10 ** (-8)  # modified
        self.hydrogen_ion_process_inputs_term = 2.5 * 10 ** (-11)
        self.hydrogen_ion_maintenance_term = 0.0025
        self.hydrogen_inhibit_k1 = 1 * 10 ** (-5)  # mol / L
        self.hydrogen_inhibit_k2 = 2.5 * 10 ** (-8)  # mol / L
        self.nitrogen_concentration_in_oil_feed = 20000  # g / L
        self.nitrogen_concentration_in_paa_feed = 80000  # +- 28 g / L
        self.nitrogen_shots = 400000  # mg / kg
        self.nitrogen_penicillin_yield = 80  # mg / g
        self.nitrogen_biomass_yield = 10  # mg / g
        self.maintenance_n_term = 0.03  # mg / g hr
        self.critical_n_concentration_for_biomass = 150  # mg / L
        self.paa_feed_solution_concentration = 530000  # +- 35000 mg / L
        self.paa_biomass_yield_coefficient = 37.5 * 1.2  # mg / g # paper do not have 1.2
        self.paa_penicillin_yield_coefficient = 187.5  # mg / g
        self.maintenance_paa_term = 1.05  # mg / g hr  # 1.2 in paper
        self.critical_paa_concentration_for_biomass = 2000  # mg / L # 2400 in matlab
        self.critical_paa_concentration_for_penicillin = 200  # mg / L
        self.peni_hydrolysis_const1 = -64.29
        self.peni_hydrolysis_const2 = -1.825
        self.peni_hydrolysis_const3 = 0.3649
        self.peni_hydrolysis_const4 = 0.1280
        self.peni_hydrolysis_const5 = -4.9496 * 10 ** (-4)
        self.volumetric_mass_transfer_ratio = 0.89
        self.constant_related_to_filamentous_break_up = 0.05  # 0.005 in Matlab
        self.viscosity_coefficient_kin = 0.001
        self.viscosity_coefficient_kde = 0.0001
        self.viscosity_coefficient_tin = 1  # hr
        self.viscosity_coefficient_tde = 250  # hr
        self.penicillin_co2_yield_coefficient = 850  # mg (Co2) / g (X)
        self.maintenance_co2_coefficient = 66  # mg (Co2) / g (X) hr
        self.critical_co2_concentration_for_biomass = 35  # mg / L in paper # 7570 mg in matlab

        # Additional parameters for inner cell dynamics (supplement tables)
        self.biomass_growth_ratio = 0.40
        self.std_dev_of_penicillin_inhibition_curve = 0.0015 + 0.0002 * std_of_penicillin_inhibition_curve_perturbation
        self.opt_substrate_concentration_for_penicillin = 0.002 + opt_sub_con_for_peni_bias \
                                                          + 0.0004 * opt_sub_con_for_peni_perturb  # g / L
        self.vacuole_rate_constant = 1.71 * 10 ** (-4)  # / hr
        self.autolysis_rate_constant = 3.5 * 10 ** (-3)  # /hr
        self.differentiation_rate_constant_for_a0 = 5.36 * 10 ** (-3)  # +- 0.000024 g / L hr
        self.differentiation_constant_beta = 0.006
        self.branching_substrate_saturation_constant = 0.05  # g s / L
        self.differentiation_substrate_saturation_constant = 0.75  # g s / L
        self.extension_substrate_saturation_constant = 0.009  # g s / L
        self.vacuolation_substrate_saturation_constant = 0.005  # g s / L
        self.vacuole_radius_bin = 0.75 * 10 ** (-4)  # cm
        self.vacuole_growth_rate = 3.22 * 10 ** (-5)  # g / cm3
        self.vacuole_growth_dispersion_constant = 2.66 * 10 ** (-11)  # g / cm3
        self.density_of_a1_regions = 0.35  # g cm2
        self.density_of_a3_regions = 0.18  # g / cm3
        self.vacuole_radius_minimum = 1.5 * 10 ** (-4)  # cm
        self.vacuole_radius_bins_delta = 1 * 10 ** (-4)  # cm

        # Additional process parameters
        self.cooling_coils_constant_beta = 3.  # 2.88 in paper, avoid numerical error
        self.cooling_coils_constant_alpha = 6000  # 2451.8 kJ / m3 in paper
        self.soybean_oil_density = 900  # kg / m3
        self.density_of_water = 1000  # kg / m3
        self.density_of_substrate = 1320  # kg / m3
        self.oxygen_inlet_air_concentration = 0.21  # %
        self.co2_inlet_air_concentration = 0.038  # %
        self.nitrogen_inlet_air_concentration = 0.79  # %

        # Fixed input
        self.rpm = 100
        self.f_dis = 0

    @staticmethod
    def truncated_normal_distribution(mean, sigma, lower, upper):
        x = np.random.normal(mean, sigma)
        for i in range(1000):
            if x < lower or x > upper:
                x = np.random.normal(mean, sigma)
        if x < lower or x > upper:
            print('Not enough iteration for sampling')
        return x

    def reset(self):
        plant_state, plant_input = self.initial_state, self.initial_input
        return plant_state, plant_input

    def step(self, plant_state, plant_input):
        # Descaled in, Descaled out
        # Integrate ODE
        scaled_state = UtilityFunctions.scale(plant_state, self.xmin, self.xmax)
        scaled_input = UtilityFunctions.scale(plant_input, self.umin, self.umax)

        scaled_state = np.clip(scaled_state, 0, 1)
        scaled_input = np.clip(scaled_input, 0, 1)

        xdot, x, u = self.dynamics
        setting = {'x': x, 'p': u, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        integrate = ca.integrator('integrate', 'cvodes', setting, options)
        solution = integrate(x0=scaled_state, p=scaled_input)
        x_plus = np.squeeze(np.array(solution['xf']))

        descaled_x_plus = UtilityFunctions.descale(x_plus, self.xmin, self.xmax)
        descaled_x_plus = np.clip(descaled_x_plus, self.xmin, self.xmax)
        return descaled_x_plus

    @property
    def dynamics(self):

        # Scaled in, Scaled out
        plant_state = ca.SX.sym('state', self.state_dimension)
        plant_input = ca.SX.sym('input', self.input_dimension)

        xd = UtilityFunctions.descale(plant_state, self.xmin, self.xmax)
        ud = UtilityFunctions.descale(plant_input, self.umin, self.umax)

        time, a_0, a_1, a_3, a_4, integral_x, s, p, vol, temp, hydro_ion, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, \
        dissolved_o2, dissolved_co2, viscosity, paa, o2, co2, ni = ca.vertsplit(xd)
        f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3 = ca.vertsplit(ud)

        # Broth density
        density_b = 1100 + p + a_0 + a_1 + a_3 + a_4  # g / L

        # Liquid Height
        liquid_height = (1 - self.gas_hold_up) * (vol / 1000) / (np.pi * (self.tank_radius ** 2))  # m

        # Pressure
        pressure_bottom = 1 + pressure + ((density_b * liquid_height) * 9.81 * 10 ** (-5))  # bar
        pressure_top = 1 + pressure  # bar
        # lmp = Log mean pressure
        lmp = (pressure_bottom - pressure_top) / (np.log(pressure_bottom / pressure_top))

        # Production mu
        prod_mu_total = (1/4)*self.production_mu\
                        * (1 - np.tanh((self.peni_cri_do_level * ((lmp * o2) / self.henry_o2) - dissolved_o2))) \
                        * (1 + np.tanh((paa - self.critical_paa_concentration_for_penicillin))) \
                        * (2.5 * (np.exp(-(1 / 2) * ((s - self.opt_substrate_concentration_for_penicillin)
                                                     / self.std_dev_of_penicillin_inhibition_curve) ** 2) / (
                                      np.sqrt(2 * np.pi))))

        # Evaporation rate - use matlab equation
        f_evp = vol * self.evaporate_coefficient * (np.exp(2.5 * (temp - self.t0) / (self.tv - self.t0)) - 1)  # L / hr

        # Dilution
        dilution = f_s + f_a + f_b + f_w + f_paa - f_evp

        # Volume of vacuole, Note n is the density not a number
        # The equations in matlab may be wrong
        v_vol_0 = np.pi * (self.d_0 + self.r_0) ** 3 * n0 * (self.r_0 - self.d_0) / 6
        v_vol_1 = np.pi * (self.r_0 + self.r_1) ** 3 * n1 * self.d_r / 6
        v_vol_2 = np.pi * (self.r_1 + self.r_2) ** 3 * n2 * self.d_r / 6
        v_vol_3 = np.pi * (self.r_2 + self.r_3) ** 3 * n3 * self.d_r / 6
        v_vol_4 = np.pi * (self.r_3 + self.r_4) ** 3 * n4 * self.d_r / 6
        v_vol_5 = np.pi * (self.r_4 + self.r_5) ** 3 * n5 * self.d_r / 6
        v_vol_6 = np.pi * (self.r_5 + self.r_6) ** 3 * n6 * self.d_r / 6
        v_vol_7 = np.pi * (self.r_6 + self.r_7) ** 3 * n7 * self.d_r / 6
        v_vol_8 = np.pi * (self.r_7 + self.r_8) ** 3 * n8 * self.d_r / 6
        v_vol_9 = np.pi * (self.r_8 + self.r_9) ** 3 * n9 * self.d_r / 6
        v_2 = v_vol_0 + v_vol_1 + v_vol_2 + v_vol_3 + v_vol_4 + v_vol_5 + v_vol_6 + v_vol_7 + v_vol_8 + v_vol_9

        # Hydrolysis coefficients effected by Temp and PH
        hydrolysis_mu_total = np.exp(self.peni_hydrolysis_const1
                                     + self.peni_hydrolysis_const2 * (-np.log10(hydro_ion))
                                     + self.peni_hydrolysis_const3 * temp
                                     + self.peni_hydrolysis_const4 * (np.log10(hydro_ion)) ** 2
                                     + self.peni_hydrolysis_const5 * temp ** 2)

        # Density of A_1 & nongrowing cytoplasm in A_1
        density_a1 = a_1/(a_1/self.density_cytoplasm + v_2)  # g/mL
        v_1c = a_1/(2*density_a1) - v_2

        # Total_biomass
        x = a_0 + a_1 + a_3 + a_4

        # rate of branch formation
        r_b = self.branching_mu * a_1 * s / (self.branching_k0 + s)

        # rate of extension formation
        r_e = self.extension_mu * a_0 * s / (self.extension_k0 + s)

        a_t = integral_x / x
        k_diff = ca.fmax(0.09, 0.75 - self.differentiation_constant_beta * a_t)

        r_diff = self.differentiation_rate_constant_for_a0 * a_0 / (k_diff + s)

        # rate of production, modified version in 2015 paper
        r_p = prod_mu_total * self.density_cytoplasm * v_1c

        # rate of maintenance
        r_m = a_0 * s / (k_diff + s) + self.density_cytoplasm * v_1c * s / (self.substrate_maintenance_k1 + s)

        # rate of degeneration
        r_deg = np.pi * (self.r_9 + self.r_max) ** 3 * self.density_degeneration * self.k_s * n9 / 6

        # rate of autolysis
        r_a = self.autolysis_mu * a_3

        # rate of hydrolysis
        r_h = hydrolysis_mu_total * p

        # Saturated DO
        dissolved_o2_star = self.oxygen_inlet_air_concentration * lmp / self.henry_o2

        # Saturated CO2
        henry_co2 = (np.exp(11.25 - 395.9 / (temp - 175.9))) / (44 * 100)
        co2_star = co2 * lmp / henry_co2

        # Power required for agitation
        unaerated_power = self.number_of_impellers*self.power_number*density_b*(self.rpm / 60) ** 3 \
                                                  * (2*self.impeller_radius)**5
        p_g = 0.706*(((unaerated_power**2)*(self.rpm/60)*(2*self.impeller_radius)**3)
                     / (f_g**0.56))**0.45
        # p_g / unaerated_power : 0.4 ~ 0.6 < could use self.Gas_to_ungass_power_ratio
        p_agitation = (self.number_of_impellers*self.power_number*density_b*(self.rpm/60)**3
                       * (2*self.impeller_radius)**5*p_g)/(unaerated_power*1000)  # kW

        # Power required for aeration, use the equation in matlab
        p_air = (f_g/(np.pi*self.tank_radius**2))*self.universal_gas_constant*temp*(vol/1000)\
                * np.log(1 + density_b*self.gravitational_constant*liquid_height/(pressure_top*10**5))

        # Total power consumption
        p_w = p_agitation + p_air

        # Volumetric mass transfer coefficient
        volumetric_mass_transfer_coeff = self.k1a_alpha*(
                (f_g/(np.pi*(self.tank_radius**2)))**self.k1a_a)*((1000*p_w/vol)**self.k1a_b) \
                * ((viscosity/100)**self.k1a_c)*(1 - (f_oil/vol)**self.k1a_d)

        # Cooling coil - matlab use alpha/1000 typo? -> we use f_c/1000
        q_c = self.cooling_coils_constant_alpha*(f_c/1000)**self.cooling_coils_constant_beta \
              * (temp - self.temperature_cold_water)/(1/1000 + self.cooling_coils_constant_alpha
              * ((f_c/1000)**(self.cooling_coils_constant_beta - 1))
                                                      / (2*self.density_of_water*self.specific_heat_capacity_water))

        q_h = self.cooling_coils_constant_alpha*(f_h/1000)**self.cooling_coils_constant_beta \
              * (self.temperature_hot_water - temp)/(1/1000 + self.cooling_coils_constant_alpha
              * ((f_h/1000)**(self.cooling_coils_constant_beta - 1))
                                                     / (2*self.density_of_water*self.specific_heat_capacity_water))

        # pH_control, diff by ph level, use matlab code
        # Assume pH < 7
        c_base = -self.concentration_of_acid_base
        c_acid = self.concentration_of_acid_base
        pH_coeff = -(hydro_ion*vol + c_acid*f_a*self.time_interval + c_base*f_b*self.time_interval)\
                   / (vol + f_a*self.time_interval + f_b*self.time_interval)
        h_control = (-pH_coeff + np.sqrt(pH_coeff**2 + 4*10**(-14)))/2 - hydro_ion

        ######################    Differential equations     ######################

        dtdt = 1.

        # Vacuole formation - diff between 1996 paper and matlab, use paper (not matlab)
        dn0dt = ((self.gamma_2*v_1c/(self.substrate_maintenance_k2 + s))
                 * (6/(np.pi*(self.r_0 + self.d_0)**3)) - self.k_s*n0)/(self.r_0 - self.d_0)
        dn1dt = -self.k_s*(n2 - n0)/(2*self.d_r) + self.diffusion_coefficient*(n2 - 2*n1 + n0)/(self.d_r**2)
        dn2dt = -self.k_s*(n3 - n1)/(2*self.d_r) + self.diffusion_coefficient*(n3 - 2*n2 + n1)/(self.d_r**2)
        dn3dt = -self.k_s*(n4 - n2)/(2*self.d_r) + self.diffusion_coefficient*(n4 - 2*n3 + n2)/(self.d_r**2)
        dn4dt = -self.k_s*(n5 - n3)/(2*self.d_r) + self.diffusion_coefficient*(n5 - 2*n4 + n3)/(self.d_r**2)
        dn5dt = -self.k_s*(n6 - n4)/(2*self.d_r) + self.diffusion_coefficient*(n6 - 2*n5 + n4)/(self.d_r**2)
        dn6dt = -self.k_s*(n7 - n5)/(2*self.d_r) + self.diffusion_coefficient*(n7 - 2*n6 + n5)/(self.d_r**2)
        dn7dt = -self.k_s*(n8 - n6)/(2*self.d_r) + self.diffusion_coefficient*(n8 - 2*n7 + n6)/(self.d_r**2)
        dn8dt = -self.k_s*(n9 - n7)/(2*self.d_r) + self.diffusion_coefficient*(n9 - 2*n8 + n7)/(self.d_r**2)
        dn9dt = -self.k_s*(nm - n8)/(2*self.d_r) + self.diffusion_coefficient*(nm - 2*n9 + n8)/(self.d_r**2)
        # matlab use dn9dt instead of n9, dimension does not match in matlab
        dnmdt = self.k_s*n9/(self.r_max - self.r_9) - self.autolysis_mu*nm

        # Growing regions
        da0dt = (1/16)*(r_b - r_diff - dilution*a_0/vol)\
                * (1 - np.tanh(self.biomass_cri_do_level*((lmp*o2)/self.henry_o2)-dissolved_o2)) \
                * (1 - np.tanh(self.critical_co2_concentration_for_biomass - dissolved_co2*1000)) \
                * (1 - np.tanh(self.critical_n_concentration_for_biomass - ni)) \
                * (1 + np.tanh(self.critical_paa_concentration_for_biomass - paa)) \
                * (self.arrhenius_constant_for_cell_growth
                * np.exp(-self.activation_energy_for_cell_growth/(self.universal_gas_constant*temp))
                - self.arrhenius_constant_for_cell_death
                * np.exp(-self.activation_energy_for_cell_death/(self.universal_gas_constant*temp))) \
                * (1/(1 + hydro_ion/self.hydrogen_inhibit_k1 + self.hydrogen_inhibit_k2/hydro_ion))

        # Nongrowing regions
        da1dt = (1/16)*(r_e - r_b + r_diff - r_deg - dilution*a_1/vol)\
                * (1 - np.tanh(self.biomass_cri_do_level*((lmp*o2)/self.henry_o2) - dissolved_o2)) \
                * (1 - np.tanh(self.critical_co2_concentration_for_biomass - dissolved_co2*1000)) \
                * (1 - np.tanh(self.critical_n_concentration_for_biomass - ni)) \
                * (1 + np.tanh(self.critical_paa_concentration_for_biomass - paa)) \
                * (self.arrhenius_constant_for_cell_growth
                * np.exp(-self.activation_energy_for_cell_growth/(self.universal_gas_constant*temp))
                - self.arrhenius_constant_for_cell_death
                * np.exp(-self.activation_energy_for_cell_death/(self.universal_gas_constant*temp))) \
                * (1/(1 + hydro_ion/self.hydrogen_inhibit_k1 + self.hydrogen_inhibit_k2/hydro_ion))

        # Degenerated regions
        da3dt = (1/16)*(r_deg - r_a - dilution*a_3/vol)\
                * (1 - np.tanh(self.biomass_cri_do_level*((lmp*o2)/self.henry_o2) - dissolved_o2)) \
                * (1 - np.tanh(self.critical_co2_concentration_for_biomass - dissolved_co2*1000)) \
                * (1 - np.tanh(self.critical_n_concentration_for_biomass - ni)) \
                * (1 + np.tanh(self.critical_paa_concentration_for_biomass - paa)) \
                * (self.arrhenius_constant_for_cell_growth
                * np.exp(-self.activation_energy_for_cell_growth/(self.universal_gas_constant*temp))
                - self.arrhenius_constant_for_cell_death
                * np.exp(-self.activation_energy_for_cell_death/(self.universal_gas_constant*temp))) \
                * (1/(1 + hydro_ion/self.hydrogen_inhibit_k1 + self.hydrogen_inhibit_k2/hydro_ion))

        # Autolysed regions
        da4dt = (1/16)*(r_a - dilution*a_4/vol)\
                * (1 - np.tanh(self.biomass_cri_do_level*((lmp*o2)/self.henry_o2) - dissolved_o2)) \
                * (1 - np.tanh(self.critical_co2_concentration_for_biomass - dissolved_co2*1000)) \
                * (1 - np.tanh(self.critical_n_concentration_for_biomass - ni)) \
                * (1 + np.tanh(self.critical_paa_concentration_for_biomass - paa)) \
                * (self.arrhenius_constant_for_cell_growth
                * np.exp(-self.activation_energy_for_cell_growth/(self.universal_gas_constant*temp))
                - self.arrhenius_constant_for_cell_death
                * np.exp(-self.activation_energy_for_cell_death/(self.universal_gas_constant*temp))) \
                * (1/(1 + hydro_ion/self.hydrogen_inhibit_k1 + self.hydrogen_inhibit_k2/hydro_ion))

        # rate of differentiation
        dintegral_xdt = x

        # Product formation
        dpdt = r_p - r_h - dilution*p/vol

        # Substrate consumption
        dsdt = - self.biomass_substrate_yield_coefficient*r_e - self.biomass_substrate_yield_coefficient*r_b \
               - self.substrate_maintenance_term*r_m - self.penicillin_substrate_yield_coefficient*r_p \
               + f_s*self.sugar_feed_concentration/vol + f_oil*self.oil_feed_concentration/vol - dilution*s/vol

        # Volume change
        dvoldt = f_s + f_oil + f_paa + f_a + f_b + f_w - f_evp + self.f_dis*1000/density_b

        # Dissolved oxygen,
        ddissolved_o2dt = - (da0dt + da1dt + da3dt + da4dt)*self.biomass_oxygen_yield_coefficient \
                          - dpdt*self.penicillin_oxygen_yield_coefficient - self.maintenance_oxygen_coefficient*x \
                          + volumetric_mass_transfer_coeff*(dissolved_o2_star - dissolved_o2)\
                          - dissolved_o2*dilution/vol

        # Dissolved CO2
        ddissolved_co2dt = self.volumetric_mass_transfer_ratio*volumetric_mass_transfer_coeff \
                           * (co2_star - dissolved_co2) - dissolved_co2*dilution/vol

        # Nitrogen
        dnidt = (f_oil*self.nitrogen_concentration_in_oil_feed + f_paa*self.nitrogen_concentration_in_paa_feed
                + nh3*self.nitrogen_shots)/vol - (da0dt + da1dt + da3dt + da4dt)*self.nitrogen_biomass_yield \
                - dpdt*self.nitrogen_penicillin_yield - self.maintenance_n_term*x - ni*dilution/vol

        # Precursor addition
        dpaadt = f_paa*self.paa_feed_solution_concentration/vol - self.paa_penicillin_yield_coefficient*dpdt \
                 - self.paa_biomass_yield_coefficient*(da0dt + da1dt + da3dt + da4dt) \
                 - self.maintenance_paa_term*p - paa*dilution/vol

        # Viscosity (heuristic relation) Matlab use 3 times and minus on kde / take minus but not 3time
        dviscositydt = 3*a_0**(1/3)\
                       * (1/(1 + np.exp(-self.viscosity_coefficient_kin*(time - self.viscosity_coefficient_tin)))) \
                       * (1/(1 + np.exp(-self.viscosity_coefficient_kde*(time - self.viscosity_coefficient_tde)))) \
                       - self.constant_related_to_filamentous_break_up*f_w

        # Heat of reaction
        q_rxn = ((da0dt + da1dt + da3dt + da4dt)*self.heat_yield_coefficients_biomass*650
                 + dpdt*self.heat_yield_coefficients_penicillin*160)*(vol/1000)

        # Temperature
        dtempdt = (1/((vol/1000)*self.specific_heat_capacity_water*density_b))\
                  * (f_s*self.density_of_substrate*self.specific_heat_capacity_substrate
                    * (self.temperature_substrate - temp)/1000 + f_w*self.density_of_water
                     * self.specific_heat_capacity_water*(self.temperature_cold_water - temp)/1000
                     - self.heat_evaporation*self.density_of_water*f_evp/1000 + p_w
                     - self.vessel_wall_heat_transfer_coefficient*self.area_of_wall
                     * (temp - self.temperature_air) - q_c + q_h + q_rxn)
        # - f_evp*density_b*self.specific_heat_capacity_water/1000

        # PH - very sensitive (0.2~0.3 changes do affects the system. Should operated near 6.5 +- 0.2)
        # Take the equation from Matlab but modified
        # Assume pH < 7
        dhydro_iondt = self.hydrogen_ion_production_term*((r_b + r_e + r_diff + r_a)
                                                          + self.hydrogen_ion_maintenance_term*x + r_p) \
                       + self.hydrogen_ion_process_inputs_term*(f_s + f_oil + f_a + f_b + self.f_dis + f_w) + h_control

        # O2 off-gas, equation from Matlab
        mass_flow_rate_o2_in = 60*f_g*1000*32/22.4
        mass_flow_rate_o2_out = 60*f_g*(0.79/(1 - o2 - co2/100))*1000*32/22.4

        do2dt = (mass_flow_rate_o2_in*0.21 - mass_flow_rate_o2_out*o2 - volumetric_mass_transfer_coeff
                 * (dissolved_o2_star - dissolved_o2))/(vol*self.gas_hold_up*28.97/22.4)

        # CO2 off-gas, equation from Matlab
        dco2dt = (((60*f_g*44*1000)/22.4)*0.033 + (a_0 + a_1)*0.123*1.1*vol - ((60*f_g*44*1000)/22.4)*co2)\
                 / (vol*self.gas_hold_up*28.97/22.4)

        diff_values = ca.vertcat(dtdt, da0dt, da1dt, da3dt, da4dt, dintegral_xdt, dsdt, dpdt, dvoldt, dtempdt,
                                 dhydro_iondt, dn0dt, dn1dt, dn2dt, dn3dt, dn4dt, dn5dt, dn6dt, dn7dt, dn8dt, dn9dt,
                                 dnmdt, ddissolved_o2dt, ddissolved_co2dt, dviscositydt, dpaadt, do2dt, dco2dt, dnidt)
        diff_values = np.multiply(diff_values, self.scale_grad)  # Because of scaling

        return diff_values, plant_state, plant_input
