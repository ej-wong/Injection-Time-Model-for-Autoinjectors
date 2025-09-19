import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

## inputs
# fluid
fluid_viscosity = 14.9 # [cP] dynamic viscosity (14.9 cP for Entyvio)
fill_volume = 0.714 # [ml]
rho = 1050 # [kg/m^3] fluid density (1050 kg/m3 for Entyvio)
h_a0 = 0.002 # [m] initial air gap height above between stopper and fluid
# prefilled syringe
m_needle_barrel = 0.002 # [kg] mass of syringe barrel + needle
m_plunger_stopper = 0.005 # [kg] total mass of plunger + stopper (the stopper may or may not come with a plunger attached)
t_stopper0 = 0.005 # [m] uncompressed stopper axial thickness
r = 0.135/1000 # [m] needle radius
R = 3.175/1000 # [m] syringe barrel radius
L_n = 0.0127 # [m] needle length
K_L1 = 0.5 # loss coefficient due to contraction of cross-section (when the fluid flows from syringe barrel -> needle)
K_L2 = 1 # loss coefficient due to expansion of cross-section (when the fluid flows from needle -> syringe barrel)
# autoinjector
F_spring0 = 25 # [N] initial spring force
k_spring = 550 # [N/m] spring constant
m_eff_spring = 0.001 # [kg] effective spring mass
m_rod = 0.002 # [kg] mass of rod
m_container = 0.005 # [kg] mass of the container closure that drives the syringe forward
L_plunger = 0 # [m] plunger length
L_rod_flight = 0.005 # [m] rod flight distance before contact with plunger (or contact with stopper if no plunger is present)
L_barrel_flight = 0.01 # [m] barrel flight distance for full insertion
# environment
P_atm = 101325 # [Pa] atmospheric pressure
T_0 = 23 + 273.15 # [K] room temperature
cp = 1007 # [J/kgK] specific heat capacity of air at T_0
M_air = 0.02897 # [kg/mol] molar mass of air
R_g = 8.314 # [J/molK] universal gas constant

# Parameters
k1 = 6.17 # [N] (from Entyvio experinments)
k2 = 0.85 # [N⋅s/m] (Zhong et al. 2021) # not used
k3 = 4.2e5 # [W/(K⋅m2)] (Zhong et al. 2021)
k4 = 3.39 # [N] (Zhong et al. 2021)
k5 = 3.20 # [N] (Zhong et al. 2021)
k6 = 1.89 # [N⋅s/m] (Zhong et al. 2021)
k7 = 0.34 # [-] (Zhong et al. 2021)
k8 = 4.3e4  # [N/m] (Zhong et al. 2021)
k9 = 6.3e4 # [Pa⋅s] (Zhong et al. 2021)
k10 = 1.3e5 # [N/m] (Zhong et al. 2021)
k11 = 5.9e4 # [Pa⋅s] (Zhong et al. 2021)

## computational parameters
dt = 0.00001 # time step
simtime = 0.01 # [s] simulation time
start_time = 0 # [s] time to start plotting from
end_time = simtime # [s] time to end plotting at

## empty arrays
n_steps = int(simtime/dt) # number of steps
t = np.zeros(n_steps+1) # simulation time
F_spring = np.zeros(n_steps+1) # spring force
a_barrel = np.zeros(n_steps+1) # barrel acceleration
v_barrel = np.zeros(n_steps+1) # barrel velocity
s_barrel = np.zeros(n_steps+1) # barrel displacement
a_rod = np.zeros(n_steps+1) # rod acceleration
v_rod = np.zeros(n_steps+1) # rod velocity
s_rod = np.zeros(n_steps+1) # rod displacement
a_stopper = np.zeros(n_steps+1) # stopper acceleration
v_stopper = np.zeros(n_steps+1) # sotpper velocity
s_stopper = np.zeros(n_steps+1) # stopper displacement
t_stopper = np.zeros(n_steps+1) # stopper thickness
c_stopper = np.zeros(n_steps+1) # stopper compression
a_2 = np.zeros(n_steps+1) # fluid acceleration in needle
v_2 = np.zeros(n_steps+1) # fluid velocity in needle
v_1 = np.zeros(n_steps+1) # fluid velocity in barrel
h_1 = np.zeros(n_steps+1) # fluid height in barrel
h_2 = np.zeros(n_steps+1) # fluid height in needle
h_airgap = np.zeros(n_steps+1) # airgap height
P_airgap = np.zeros(n_steps+1) # airgap pressure
T_airgap = np.zeros(n_steps+1) # airgap temperature
dTdt = np.zeros(n_steps+1) # rate of change of airgap temperature
F_rod_plunger = np.zeros(n_steps+1) # contact force between rod and plunger (or between rod and stopper if plunger not present)
F_f = np.zeros(n_steps+1) # friction force between stopper and syringe barrel walls
F_m = np.zeros(n_steps+1) # friction force inside rod driving mechanism
F_barrel_stop = np.zeros(n_steps+1) # force on the syringe barrel due to mechanical stiop
c_barrelstop = np.zeros(n_steps+1) # compression of barrel mechanical stop

## initial conditions
P_airgap[0] = P_atm
T_airgap[0] = T_0
h_airgap[0] = h_a0
s_stopper[0] = 0
v_stopper[0] = 0
a_stopper[0] = 0
s_rod[0] = 0 # 
v_rod[0] = 0
a_rod[0] = 0
h_1[0] = (fill_volume/(1e6))/(np.pi*(R**2)) # initial fluid height 
v_1[0] = 0
h_2[0] = 0
v_2[0] = 0
a_2[0] = 0
s_rod_max = L_barrel_flight + L_rod_flight + h_airgap[0] + h_1[0] # rod maximum displacement
s_stopper_max = L_barrel_flight + h_airgap[0] + h_1[0] # stopper maximum displacement

### simulation start ###
V = fill_volume/(1e6) # [m^3] fill volume
A = np.pi*(R**2) # [m^2] syringe cross-section
mu = fluid_viscosity/1000 # [Pa⋅s] dynamic viscosity
m_fluid = rho*V
m_air = (P_atm*np.pi*(R**2)*h_a0*M_air) / (R_g*T_0) # [kg] mass of air in airgap
F_bl = k1 # [N] break loose force
H = k3 # [W/(K⋅m2)] heat convection coefficient
def smooth_sign(x, width): # smoothing function for numerical stability
    return np.tanh(x/width) # tanh is bounded in [-1, 1]
no_slip = True # no slip
complete_insertion = False # insertion not complete
for i in tqdm(range(n_steps), total=n_steps, desc="Simulating", unit="step", mininterval=0.3, smoothing=0.1):
    if h_1[i] <= 0:
        break # if no break, there is fluid in the barrel
    # spring force decay
    F_spring[i] = F_spring0 - k_spring*s_rod[i]
    # friction force between spring-rod system
    F_m[i] = k4*smooth_sign(v_rod[i], width = 1e-2)
    c_barrelstop[i] = s_barrel[i] - L_barrel_flight
    if c_barrelstop[i] <= 0: # no contact with barrel stop
        F_barrel_stop[i] = 0
        c_barrelstop[i] = 0
    else: # contact with barrel_stop
        F_barrel_stop[i] = c_barrelstop[i]*(k10 + k11*v_barrel[i]) # damping in both directions
        complete_insertion = True
    if not complete_insertion:
        a_barrel[i] = (F_spring[i] - F_m[i] - F_barrel_stop[i]) / (m_container + m_rod + m_eff_spring + m_needle_barrel + m_fluid + m_plunger_stopper)
        # updating barrel
        v_barrel[i+1] = v_barrel[i] + dt*a_barrel[i]
        s_barrel[i+1] = s_barrel[i] + dt*v_barrel[i+1]
        # rod rides along with barrel
        a_rod[i] = a_barrel[i]
        v_rod[i+1] = v_barrel[i+1]
        s_rod[i+1] = s_barrel[i+1]
        # stopper rides along with barrel (no slip)
        a_stopper[i]   = a_barrel[i]
        v_stopper[i+1] = v_barrel[i+1]
        s_stopper[i+1] = s_barrel[i+1]
        # no squeeze/flow during insertion (relative to barrel)
        K_L = K_L1 if v_2[i] >= 0.0 else K_L2
        if h_2[i] < L_n: # needle is partially filled
            a_2[i] = (P_airgap[i] - P_atm - (rho*a_barrel[i]*(h_1[i] + h_2[i])) - ((8*mu*v_2[i]*h_2[i])/(r**2)) + (rho/2)*((v_1[i]**2) - (1 + K_L)*(v_2[i]**2))) / (rho*(h_2[i] + h_1[i]*(r/R)**2))
        else: # needle is completely filled
            a_2[i] = (P_airgap[i] - P_atm - (rho*a_barrel[i]*(h_1[i] + L_n)) - (8*mu*v_2[i]*L_n)/(r**2) + (rho/2)*((v_1[i]**2) - (1 + K_L)*(v_2[i]**2))) / (rho*(L_n + h_1[i]*(r/R)**2))
        v_2[i+1] = v_2[i] + dt*a_2[i]
        h_2[i+1] = h_2[i] + dt*v_2[i+1]
        if h_2[i+1] <= 0:
            h_2[i+1] = 0.0
            v_2[i+1] = 0.0
        v_1[i+1] = v_2[i+1]*((r/R)**2)
        h_1[i+1] = h_1[i] - dt*v_1[i+1]
        # updating airgap
        dTdt[i] = -(A*H*(T_airgap[i] - T_0)) / (cp*m_air)
        T_airgap[i+1] = T_airgap[i] + dt*dTdt[i]
        h_airgap[i+1] = h_airgap[i]
        P_airgap[i+1] = ((m_air/M_air)*R_g*T_airgap[i+1]) / (h_airgap[i+1]*A)
        i_insertion = i
        t_insertion = i*dt
    else:
        v_rel_sb = v_stopper[i] - v_barrel[i]
        # rod-plunger force (parallel spring + damper model)
        c_stopper[i] = (s_rod[i] - L_rod_flight) - (s_stopper[i]) # compression
        if c_stopper[i] > 0:
            if c_stopper[i] > t_stopper0:
                c_stopper[i] = t_stopper0
            v_rel_rs = v_rod[i] - v_stopper[i]
            F_rod_plunger[i] = c_stopper[i]*(k8 + k9*max((v_rel_rs), 0))
        else:
            c_stopper[i] = 0
            F_rod_plunger[i] = 0.0
        t_stopper[i] = t_stopper0 - c_stopper[i] # stopper width
        # rod acceleration
        a_rod[i] = (F_spring[i] - F_m[i] - F_rod_plunger[i])/(m_eff_spring + m_rod)
        # stopper-barrel friction (4 cases)
        F_drive = F_rod_plunger[i] + (A*(P_atm - P_airgap[i]))
        if no_slip == True:
            # friction force between stopper and barrel
            if abs(F_drive) < F_bl: # case 1: no slip continues
                F_f[i] = F_drive
                a_barrel[i] = (F_rod_plunger[i] - F_barrel_stop[i]) / (m_needle_barrel + m_fluid + m_plunger_stopper)
                a_stopper[i]   = a_barrel[i]
                v_stopper[i+1] = v_stopper[i] + dt*a_stopper[i]
                s_stopper[i+1] = s_stopper[i] + dt * v_stopper[i+1]
            else: # case 2: transitioning from no slip to slip
                no_slip = False
                F_f[i] = (k5 + k6*np.abs(v_rel_sb) + k7*(A*P_airgap[i]+F_spring[i]))*smooth_sign(v_rel_sb, width = 1e-2)
                a_barrel[i] = (F_f[i] + A*(P_airgap[i] - P_atm) - F_barrel_stop[i]) / (m_needle_barrel + m_fluid)
                a_stopper[i] = (F_rod_plunger[i] - F_f[i] + A*(P_atm - P_airgap[i])) / m_plunger_stopper
                # updating stopper kinematics
                v_stopper[i+1] = v_stopper[i] + dt*a_stopper[i]
                s_stopper[i+1] = s_stopper[i] + dt*v_stopper[i+1]
                if s_stopper[i+1] >= s_stopper_max:
                    s_stopper[i+1] = s_stopper_max
                    v_stopper[i+1] = 0
                    a_stopper[i] = 0
        elif abs(v_rel_sb) < 1e-6 and (abs(F_drive) <= F_bl): # case 3: transitioning from slip to no slip
            F_f[i] = F_drive
            a_barrel[i] = (F_rod_plunger[i] - F_barrel_stop[i]) / (m_needle_barrel + m_fluid + m_plunger_stopper)
            a_stopper[i]   = a_barrel[i]
            v_stopper[i+1] = v_barrel[i+1]
            s_stopper[i+1] = s_stopper[i] + dt * v_stopper[i+1]
            no_slip = True
        else: # case 4: slip continues
            F_f[i] = (k5 + k6*np.abs(v_rel_sb) + k7*(A*P_airgap[i]+F_spring[i]))*smooth_sign(v_rel_sb, width = 1e-1)
            a_barrel[i] = (F_f[i] + A*(P_airgap[i] - P_atm) - F_barrel_stop[i]) / (m_needle_barrel + m_fluid)
            a_stopper[i] = (F_rod_plunger[i] - F_f[i] + A*(P_atm - P_airgap[i])) / m_plunger_stopper
            v_stopper[i+1] = v_stopper[i] + dt*a_stopper[i]
            s_stopper[i+1] = s_stopper[i] + dt*v_stopper[i+1]
            if s_stopper[i+1] >= s_stopper_max:
                s_stopper[i+1] = s_stopper_max
                v_stopper[i+1] = 0
                a_stopper[i] = 0
        # updating barrel kinematics
        v_barrel[i+1] = v_barrel[i] + dt*a_barrel[i]
        s_barrel[i+1] = s_barrel[i] + dt*v_barrel[i+1]
        # acceleration of fluid in the needle
        K_L = K_L1 if v_2[i] >= 0.0 else K_L2
        if h_2[i] < L_n: # needle is partially filled
            a_2[i] = (P_airgap[i] - P_atm - (rho*a_barrel[i]*(h_1[i] + h_2[i])) - ((8*mu*v_2[i]*h_2[i])/(r**2)) + (rho/2)*((v_1[i]**2) - (1 + K_L)*(v_2[i]**2))) / (rho*(h_2[i] + h_1[i]*(r/R)**2))
        else: # needle is completely filled
            a_2[i] = (P_airgap[i] - P_atm - (rho*a_barrel[i]*(h_1[i] + L_n)) - (8*mu*v_2[i]*L_n)/(r**2) + (rho/2)*((v_1[i]**2) - (1 + K_L)*(v_2[i]**2))) / (rho*(L_n + h_1[i]*(r/R)**2))
        # updating rod kinematics
        v_rod[i+1] = v_rod[i] + dt*a_rod[i]
        s_rod[i+1] = s_rod[i] + dt*v_rod[i+1]
        if s_rod[i+1] >= s_rod_max: # maximum displacement hit
            s_rod[i+1] = s_rod_max
            v_rod[i+1] = 0
            a_rod[i] = 0
        # updating fluid kinematics
        v_2[i+1] = v_2[i] + dt*a_2[i]
        v_1[i+1] = v_2[i+1]*((r/R)**2)
        h_1[i+1] = h_1[i] - dt*v_1[i+1]
        if h_1[i+1] < 0: # barrel is empty
            h_1[i+1] = 0
            v_1[i+1] = 0
        h_2[i+1] = h_2[i] + dt*v_2[i+1]
        if h_2[i+1] > L_n: # needle is fully filled
            h_2[i+1] = L_n
        if h_2[i+1] < 0: # needle is empty
            h_2[i+1] = 0
            v_2[i+1] = 0
        # updating airgap properties
        v_rel_sb = v_stopper[i] - v_barrel[i]
        dTdt[i] = ( (A*P_airgap[i]*(v_rel_sb - v_1[i])) - (A*H*(T_airgap[i] - T_0)) ) / (cp*m_air)
        T_airgap[i+1] = T_airgap[i] + dt*dTdt[i]
        v_rel_sb1 = v_stopper[i+1] - v_barrel[i+1]
        h_airgap[i+1] = h_airgap[i] - (dt*(v_rel_sb1 - v_1[i+1]))
        h_min = 1e-4 # [m] minimum air gap height ()
        if h_airgap[i+1] < h_min:
            h_airgap[i+1] = h_min
            alpha = (h_airgap[i] - h_min) / ((v_rel_sb1 - v_1[i+1])*dt) # fraction of timestep until hitting min height (height left to go/expected height change) 
            alpha = max(0.0, min(1.0, alpha))
            # fluid kinematics at contact h = h_min
            v_2_hit = v_2[i] + alpha*dt*a_2[i] # updating needle velocity to time where h = h_min
            v_1_hit = v_2_hit*((r/R)**2)
            h_1_hit = h_1[i] - alpha*dt*v_1_hit # updating barrel height to time where h = h_min
            # stopper kinematics at contact h = h_min
            s_stopper_hit = s_stopper[i] + alpha*dt*v_stopper[i+1] # stopper position at time where h = h_min
            # kinematics for the remaining fraction of the timestep (1 - alpha)*dt
            dt2 = (1 - alpha)*dt
            v_stopper[i+1] = v_barrel[i+1] + v_1_hit # stopper velocity matches fluid velocity
            v_1[i+1] = v_1_hit
            v_2[i+1] = v_2_hit
            # now advanving with constraints for the remaining timestep
            s_stopper[i+1] = s_stopper_hit + dt2*v_stopper[i+1] # stopper position at end of timestep
            h_1[i+1] = h_1_hit - dt2*v_1[i+1] # barrel fluid height at end of timestep
            h_2[i+1] = h_2[i] + dt*v_2[i+1]
            if h_2[i+1] > L_n: # needle is fully filled
                h_2[i+1] = L_n
            if h_2[i+1] < 0: # needle is empty
                h_2[i+1] = 0
                v_2[i+1] = 0
            # airgap temperature update
            T_hit = T_airgap[i] + alpha*dt*dTdt[i] # airgap temperature at time where h = h_min
            dTdt_post = -(A*H*(T_hit - T_0)) / (cp*m_air) # no compression work after contact since v_stopper = v_1
            T_airgap[i+1] = T_hit + dt2*dTdt_post
        P_airgap[i+1] = ((m_air/M_air)*R_g*T_airgap[i+1]) / (h_airgap[i+1]*A)
    # updating time
    t[i + 1] = t[i] + dt
### simulation end ###

## debugging
tqdm.write(f"initial fluid height = {h_1[0]*1000:.6f} mm")
tqdm.write(f"initial fluid height + initial airgap height = {(h_1[0]+h_a0)*1000:.6f} mm")
tqdm.write(f"stopper displacement = {s_stopper[i]*1000:.6f} mm")
tqdm.write(f"barrel flight distance + rod flight distance + initial airgap height + initial fluid height = {(L_barrel_flight + L_rod_flight + h_a0 + h_1[0])*1000:.6f} mm")
tqdm.write(f"rod displacement = {s_rod[i]*1000:.6f} mm")
tqdm.write(f"F_f = {F_f[i]:.6f} N")
tqdm.write(f"F_bl = {F_bl:.6f} N")
tqdm.write(f"a_stopper = {a_stopper[i]:.6f} m/s^2")
tqdm.write(f"a_rod = {a_rod[i]:.6f} m/s^2")
tqdm.write(f"stopper compression = {c_stopper[i]:.6f} m")
tqdm.write(f"h_airgap = {h_airgap[i]*1000:.6f} mm")
tqdm.write(f"average injection velocity = {np.mean(v_2[i_insertion:i]):.6f} m/s")
vol_injected = (V - (A*h_1[i]))
percent_injected = (vol_injected/V)*100
tqdm.write(f"volume injected = {vol_injected*1e6:.6f} ml ({percent_injected:.2f}%)")
tqdm.write(f"total injection time t = {t[i]:.6f} s")
tqdm.write(f"insertion time t = {t_insertion:.6f} s")

## indexing
N = i
t_valid = t[:N]
t_end = min(end_time, t_valid[-1])
# indices for [start_time, t_end] inside the valid region
start_index = np.searchsorted(t_valid, start_time, side='left')
final_index = np.searchsorted(t_valid, t_end,      side='right')
# setting stride length for plotting (stride = 1 plots all points)
stride = 1 
sl = slice(start_index, final_index, stride)

# subplots
fig, axs = plt.subplots(8, 2, figsize=(14, 10))
axs = axs.flatten()
axs[0].plot(t_valid[:i], F_spring[:N]);                                            axs[0].set_title("Spring Force [N]")
axs[1].plot(t_valid[sl], a_rod[:N][sl], label="a_rod");
axs[1].plot(t_valid[sl], a_stopper[:N][sl], label="a_stopper");  
axs[1].plot(t_valid[sl], a_barrel[:N][sl], lw=1.5, ls='--', alpha=0.8, label="a_barrel");                       axs[1].set_title("Rod, Stopper, Barrel Accelerations [m/s²]"); axs[1].legend(loc="upper left")
axs[2].plot(t_valid[sl], v_rod[:N][sl], label="v_rod")
axs[2].plot(t_valid[sl], v_stopper[:N][sl], label="v_stopper");
axs[2].plot(t_valid[sl], v_barrel[:N][sl], lw=1.5, ls='--', alpha=0.8, label="v_barrel");                       axs[2].set_title("Rod, Stopper, Barrel, Velocities [m/s]"); axs[2].legend(loc="upper left")
axs[3].plot(t_valid[sl], (s_rod[:N][sl]-L_rod_flight)*1000, label="s_rod")
axs[3].plot(t_valid[sl], (s_stopper[:N][sl])*1000, label="s_stopper");                
axs[3].plot(t_valid[sl], (s_barrel[:N][sl])*1000, lw=1.5, ls='--', alpha=0.8, label="s_barrel");                axs[3].set_title("Rod, Stopper, Barrel Displacements [mm]"); axs[3].legend(loc="upper left")
axs[4].plot(t_valid[sl], ((V - (np.pi*R**2*h_1[sl])) / V)*100);                     axs[4].set_title("Percentage of Volume Injected")
axs[5].plot(t_valid[sl], (h_2[:N]/L_n)[sl]*100);                                    axs[5].set_title("Percentage of Needle Filled")
axs[6].plot(t_valid[sl], v_2[:N][sl]);                                              axs[6].set_title("Fluid Velocity Through Needle [m/s]")
axs[7].plot(t_valid[sl], a_2[:N][sl]);                                              axs[7].set_title("Fluid Acceleration Through Needle [m/s²]")
axs[8].plot(t_valid[sl], (P_airgap[:N]/1e5)[sl]);                                   axs[8].set_title("Air Gap Pressure [bar]")
axs[9].plot(t_valid[sl], T_airgap[:N][sl]);                                         axs[9].set_title("Air Gap Temperature [K]")
axs[10].plot(t_valid[sl], h_airgap[:N][sl]*1000);                                   axs[10].set_title("Air Gap Height [mm]")
axs[11].plot(t_valid[sl], F_rod_plunger[:N][sl], label="F_rod_plunger");            
axs[11].plot(t_valid[sl], F_barrel_stop[:N][sl], label="F_barrel_stop");            axs[11].set_title("Rod-Plunger and Barrel Stop [N]"); axs[11].legend(loc="upper left")
axs[12].plot(t_valid[sl], F_f[:N][sl]);                                             axs[12].set_title("Stopper-Barrel Friction [N]")
axs[13].plot(t_valid[sl], F_m[:N][sl]);                                             axs[13].set_title("Spring-Rod Friction [N]")
axs[14].plot(t_valid[sl], c_barrelstop[:N][sl]*1000, label="c_barrelstop");       axs[14].set_title("barrel stop compression [mm]")
axs[15].plot(t_valid[:i], c_stopper[:N]*1000);                                              axs[15].set_title("Stopper Compression [mm]")
for ax in axs:
    ax.set_xlabel("Time (s)")
    ax.grid(True)
plt.tight_layout(); plt.savefig("full_subplots.png", dpi=300)

# Barrel, Rod, Stopper Accelerations ### INCLUDE ZOOMED IN VERSION IN THE PRESENTATION TO SEE NON ZERO ACCELERATIONS ###
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(t_valid[sl], a_barrel[:N][sl], color="#1b9e77", label="a_barrel", linestyle='-', linewidth=3)
ax.plot(t_valid[sl], a_rod[:N][sl], color="#d95f02", label="a_rod", linestyle='--', linewidth=3)
ax.plot(t_valid[sl], a_stopper[:N][sl], color="#0072B2", label="a_stopper", linestyle=':', linewidth=3)
ax.set_title("Barrel, Rod, Stopper Accelerations")
ax.set_ylabel("acceleration (m/s²)")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("acceleration_plot.png", dpi=300)

# Barrel, Rod, Stopper Velocities ### INCLUDE ZOOMED IN VERSION IN THE PRESENTATION TO SEE NON ZERO VELOCITIES ###
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(t_valid[sl], v_barrel[:N][sl], color="#1b9e77", label="v_barrel", linestyle='-', linewidth=3)
ax.plot(t_valid[sl], v_rod[:N][sl], color="#d95f02", label="v_rod", linestyle='--', linewidth=3)
ax.plot(t_valid[sl], v_stopper[:N][sl], color="#0072B2", label="v_stopper", linestyle=':', linewidth=3)
ax.set_title("Barrel, Rod, Stopper Velocities")
ax.set_ylabel("Velocity (m/s)")
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True)
plt.savefig("velocity_plot.png", dpi=300)

# Barrel, Rod, Stopper Displacements
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(t_valid[sl], (s_barrel[:N][sl])*1000, color="#1b9e77", label="s_barrel", linestyle='-', linewidth=3)
ax.plot(t_valid[sl], (s_rod[:N][sl])*1000, color="#d95f02", label="s_rod", linestyle='--', linewidth=3)
ax.plot(t_valid[sl], (s_stopper[:N][sl])*1000, color="#0072B2", label="s_stopper", linestyle=':', linewidth=3)
ax.set_title("Barrel, Rod, Stopper Displacements")
ax.set_ylabel("displacement (mm)")
ax.set_xlabel("time (s)")
ax.legend(loc="lower right")
ax.grid(True)
plt.savefig("displacement_plot.png", dpi=300)