import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import scipy.stats as st
from src.plotting_convention import mark_subplots, simplify_axes
import neuron
import LFPy
from lfpykit.models import CurrentDipoleMoment
from lfpykit.eegmegcalc import FourSphereVolumeConductor
from mpi4py import MPI
from src import cell_models


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'Rank {rank} of {size}')

np.random.seed(123456)

h = neuron.h
cell_models_folder = os.path.abspath(cell_models.__path__[0])
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")


default_pop_params = {
    'pop_radius': 250,  # population radius in µm
    'num_cells': 500,  # Out-degree, K_out
    'in_degree': 1,  # num synaptic inputs onto each postsynaptic cell
    'soma_z_scale': 100,  # SD of somas in z-direction
    'syn_z_scale': 100,  # SD of input in z-direction
    'syn_z_mean': -1270,  # mean input location in z-direction
    'tau_s_mean': 1,  # mean synaptic time constant
    'tau_s_scale': 0.2,  # SD of synaptic time constant
    'syn_delay_mean': 1,  # mean synaptic time constant
    'syn_delay_scale': 0.2,  # SD of synaptic time constant
    'weight_mean': 0.1,  # synaptic input strength mean in nA
    'weight_distribution': 'lognormal',
    'weight_scale': 0.02,  # SD of synaptic input strength in nA
    'weight_scale_lognormal': 0.4,
}


num_elecs = 16
elec_x = np.zeros(num_elecs)
elec_y = np.zeros(num_elecs)
elec_z = np.arange(num_elecs) * (-100) - 100
elec_params = {
    'sigma': 0.3,
    'x': elec_x,
    'y': elec_y,
    'z': elec_z,
}
dz = np.abs(elec_z[1] - elec_z[0])

synapse_params = {
    'syntype' : 'ExpSynI',      #conductance based exponential synapse
    'record_current' : False,    #record synaptic currents
}

sim_names = ['default',
             'small_population', 'large_population',
             'small_radius', 'large_radius',
             'uniform', 'apical',
             'similar_synapses', 'variable_synapses',
             'broad_input_region', 'narrow_input_region']

cond_clr = {sim_name: plt.cm.tab10(0.0 + idx / (len(sim_names) - 2))
            for idx, sim_name in enumerate(sim_names[1:])}

cond_clr["default"] = 'k'

head_radii = [89000., 90000., 95000., 100000.]  # (µm)
head_sigmas = [0.276, 1.65, 0.01, 0.465]  # (S/m)
r_elecs = np.array([[0.], [0.], [head_radii[-1] - 1e-2]]).T
dipole_loc = np.array([0., 0., head_radii[0] - 1000.])  # (µm)
sphere_model = FourSphereVolumeConductor(r_elecs, head_radii, head_sigmas)
M_eeg = sphere_model.get_transformation_matrix(dipole_loc)[:, 2] * 1e6 # z-axis, nV

dt = 2**-4

syn_clr =  'darkgreen'
soma_clr = 'k'
elec_clr = 'darkred'
eeg_clr = '0.6'
cell_clr = '0.9'
brain_clr = '#ffb380'
csf_clr = '#74abff'

mechs_loaded = neuron.load_mechanisms(cell_models_folder)
if not mechs_loaded:
    print("Attempting to compile mod mechanisms.")
    os.system('''
              cd {}
              nrnivmodl
              cd -
              '''.format(cell_models_folder))
    mechs_loaded = neuron.load_mechanisms(cell_models_folder)
    if not mechs_loaded:
        raise RuntimeError("Could not load mechanisms")


def calc_K_equal(kernels):
    """
    Takes numpy array with num_kernels * num_electrodes * num_kernel_tsteps
    :param kernels:
    :return:
    """

    # K_equal = np.zeros([kernels.shape[1], kernels.shape[2], kernels.shape[2]])
    # for elec in range(kernels.shape[1]):
    #     for k in range(kernels.shape[0]):
    #         K_equal[elec] += np.outer(kernels[k, elec], kernels[k, elec])
    K_equal = np.einsum('bai,baj->aij', kernels, kernels)

    return K_equal / kernels.shape[0]


def calc_K_nonequal(kernels):
    # K_nonequal = np.zeros([kernels.shape[1], kernels.shape[2], kernels.shape[2]])
    # for elec in range(kernels.shape[1]):
    #     for ki in range(kernels.shape[0]):
    #         for kj in range(kernels.shape[0]):
    #             if ki != kj:
    #                 K_nonequal[elec] += np.outer(kernels[ki, elec], kernels[kj, elec])
    K_nonequal = np.einsum('bai,caj->aij', kernels, kernels) - np.einsum('bai,baj->aij', kernels, kernels)

    return K_nonequal / (kernels.shape[0] * (kernels.shape[0] - 1))


def load_mechs_from_folder(mod_folder):

    if hasattr(neuron.h, "ISyn"):
        return
    mechs_loaded = neuron.load_mechanisms(mod_folder)

    if not mechs_loaded:
        print("Attempting to compile mod mechanisms.")
        os.system('''
                  cd {}
                  nrnivmodl
                  cd -
                  '''.format(mod_folder))
        mechs_loaded = neuron.load_mechanisms(mod_folder)
        if not mechs_loaded:
            raise RuntimeError("Could not load mechanisms")


def download_hay_model():
    load_mechs_from_folder(mod_folder)

    print("Downloading Hay model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://modeldb.science/download/139653',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def return_hay_cell(tstop, dt, make_passive=True):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    if make_passive:
        cell_params = {
            'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': True,
            'passive_parameters': {"g_pas": 1 / 30000,
                                   "e_pas": -70.},
            'nsegs_method': "lambda_f",
            "Ra": 150,
            "cm": 1.0,
            "lambda_f": 100,
            'dt': dt,
            'tstart': 0,
            'tstop': tstop,
            'v_init': -70,
            'pt3d': True,
        }

        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)

        return cell
    else:
        if not hasattr(neuron.h, "CaDynamics_E2"):
            neuron.load_mechanisms(join(hay_folder, 'mod'))
        cell_params = {
            'morphology': join(hay_folder, "morphologies", "cell1.asc"),
            'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                             join(hay_folder, 'models', 'L5PCtemplate.hoc')],
            'templatename': 'L5PCtemplate',
            'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': -200,
            'tstop': tstop,
            'v_init': -75,
            'celsius': 34,
            'pt3d': True,
        }

        cell = LFPy.TemplateCell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)
        return cell


def plot_syn_histograms(cell, syn_idxs, syn_weights, syn_taus, syn_delays):

    fig = plt.figure(figsize=[6, 3])
    ax1 = fig.add_subplot(151, aspect=1, frameon=False, xticks=[])
    ax2 = fig.add_subplot(152, title="syn loc")
    ax3 = fig.add_subplot(153, title=r"$w_{\rm syn}$")
    ax4 = fig.add_subplot(154, title=r"$\tau_{\rm syn}$")
    ax5 = fig.add_subplot(155, title=r"synaptic delay")

    ax1.plot(cell.x.T, cell.z.T, c='k')
    ax1.plot(cell.x[syn_idxs].mean(axis=1),
             cell.z[syn_idxs].mean(axis=1),
             'o', c=syn_clr, ms=2)
    ax2.hist(cell.z[syn_idxs].mean(axis=1))
    ax3.hist(syn_weights)
    ax4.hist(syn_taus)
    ax5.hist(syn_delays)

    simplify_axes(fig.axes)
    fig.savefig("syn_histograms.png")
    plt.close("all")


def insert_synaptic_input(cell, syn_idxs,
                      syn_weights, syn_taus,
                      syn_delays, input_times):

    for idx in range(len(syn_idxs)):
        synapse_params["tau"] = syn_taus[idx]
        synapse_params["weight"] = syn_weights[idx]
        synapse_params["idx"] = syn_idxs[idx]

        syn = LFPy.Synapse(cell, **synapse_params)
        syn.set_spike_times(np.array([input_times[idx] + syn_delays[idx]]))


def return_syn_weight_dist(pop_params):
    if pop_params['weight_distribution'] == 'gaussian':
        syn_weight_func = st.norm
        syn_weight_params = dict(
            loc=pop_params['weight_mean'],
            scale=pop_params['weight_scale']
        )
    elif pop_params['weight_distribution'] == 'lognormal':
        syn_weight_func = st.lognorm
        mu_normal = pop_params['weight_mean']
        sigma_lognormal = pop_params['weight_scale_lognormal']
        mu_lognormal = np.log(mu_normal) - (sigma_lognormal ** 2 / 2)
        syn_weight_params = dict(
            s=sigma_lognormal,
            scale=np.exp(mu_lognormal)
        )
    else:
        raise RuntimeError("weight distribution not recognized")

    return syn_weight_func(**syn_weight_params)


def return_syn_delay_dist(pop_params):

    syn_delay_func = st.truncnorm

    syn_delay_params = dict(
        loc=pop_params['syn_delay_mean'],
        scale=pop_params['syn_delay_scale'],
        a=(dt - pop_params['syn_delay_mean']) / pop_params['syn_delay_scale'],
        b=np.inf
    )
    return syn_delay_func(**syn_delay_params)


def return_soma_loc_dist(pop_params, soma_z_mean, max_soma_loc):

    rs = pop_params['pop_radius'] * np.sqrt(np.random.rand(
        pop_params['num_cells']))
    theta = np.random.uniform(0, 2 * np.pi,
                              pop_params['num_cells'])
    soma_xs = rs * np.cos(theta)
    soma_ys = rs * np.sin(theta)

    z_rots = np.random.uniform(0, 2 * np.pi,
                               size=pop_params['num_cells'])

    soma_loc_func = st.truncnorm
    soma_z_scale = pop_params['soma_z_scale']
    soma_loc_params = dict(
        loc=soma_z_mean,
        scale=soma_z_scale,
        a=-np.inf,
        b=(max_soma_loc - soma_z_mean) / soma_z_scale
    )
    
    soma_loc_dist = soma_loc_func(**soma_loc_params)
    soma_zs = soma_loc_dist.rvs(pop_params['num_cells'])

    
    return soma_loc_dist, soma_xs, soma_ys, soma_zs, z_rots


def return_syn_loc_dist(pop_params):
    syn_loc_func = st.norm
    syn_loc_params = dict(
        loc=pop_params['syn_z_mean'],
        scale=pop_params['syn_z_scale']
    )
    return syn_loc_func(**syn_loc_params), syn_loc_func, syn_loc_params


def return_syn_tau_dist(pop_params):
    syn_tau_func = st.truncnorm
    syn_tau_params = dict(
        loc=pop_params['tau_s_mean'],
        scale=pop_params['tau_s_scale'],
        a=(dt - pop_params['tau_s_mean']) / pop_params['tau_s_scale'],
        b=np.inf
    )
    return syn_tau_func(**syn_tau_params)


def calculate_kernel_instance(soma_z_mean, input_time, pop_params,
                              trial_idx, sim_name, tstop, save_folder,
                              plot_results=False):

    np.random.seed(123456 + trial_idx * 23)
    cell = return_hay_cell(tstop=tstop, dt=dt)
    num_tsteps = round((cell.tstop - cell.tstart) / cell.dt + 1)
    cell.tvec = np.arange(num_tsteps) * cell.dt
    cell.set_pos(x=0, y=0, z=soma_z_mean)

    cell_z_top = np.max(cell.z)
    max_soma_loc = -cell_z_top + cell.z[0].mean() - 10

    syn_z_mean = pop_params['syn_z_mean']
    syn_z_scale = pop_params['syn_z_scale']

    num_syns = pop_params['in_degree']
    num_plot_cells = np.min([pop_params['num_cells'], 100])

    syn_weight_dist = return_syn_weight_dist(pop_params)
    soma_loc_dist, soma_xs, soma_ys, soma_zs, z_rots = \
        return_soma_loc_dist(pop_params, soma_z_mean, max_soma_loc)
    syn_delay_dist = return_syn_delay_dist(pop_params)
    syn_loc_dist, syn_loc_func, syn_loc_params = return_syn_loc_dist(pop_params)
    syn_tau_dist = return_syn_tau_dist(pop_params)

    if plot_results:
        ylim = [-1600, 300]
        plt.close('all')
        fig = plt.figure(figsize=[6, 3])
        ax1 = fig.add_axes([0.01, 0.01, 0.3, 0.94], frameon=False, aspect=1,
                           xticks=[], yticks=[], ylim=ylim)
        ax1.axhspan(0, 100, color=csf_clr)
        ax1.axhspan(ylim[0], 0, zorder=-1e9, color=brain_clr)
        ax1.scatter(soma_xs, soma_zs, c='k', s=2)

        # Draw EEG electrode
        ax1.plot([0, 0, 0], [125, 150, 175], 'k.', ls='none', ms=1)
        ax1.plot(elec_x, elec_z, c=elec_clr, marker='o', ls='none', ms=4)
        collection = PatchCollection([mpatches.Ellipse([0, 250], 1000, 100)],
                                     color=eeg_clr)
        ax1.add_collection(collection)
        ax1.text(-200, 330, "EEG electrode", color=eeg_clr)

        # Draw soma outline box
        patches = []
        ellipse_top = mpatches.Ellipse([0, soma_z_mean +
                                        pop_params['soma_z_scale']],
                                        pop_params['pop_radius'] * 2, 50)
        ellipse_bottom = mpatches.Ellipse([0, soma_z_mean -
                                           pop_params['soma_z_scale']],
                                        pop_params['pop_radius'] * 2, 50)
        patches.append(ellipse_top)
        patches.append(ellipse_bottom)
        collection = PatchCollection(patches,
                                     edgecolors='k', facecolors='none',
                                     ls='--', lw=0.5)
        ax1.add_collection(collection)
        ax1.plot([-pop_params['pop_radius']] * 2,
                 [soma_z_mean + pop_params['soma_z_scale'],
                  soma_z_mean - pop_params['soma_z_scale']],
                 ls='--', c='k', lw=0.5)
        ax1.plot([pop_params['pop_radius']] * 2,
                 [soma_z_mean + pop_params['soma_z_scale'],
                  soma_z_mean - pop_params['soma_z_scale']],
                 ls='--', c='k', lw=0.5)

        ax1.plot([0, pop_params['pop_radius']],
                 [soma_z_mean - pop_params['soma_z_scale']] * 2,
                 ls='--', c='k', lw=0.5)
        ax1.text(pop_params['pop_radius'] / 2,
                 soma_z_mean - pop_params['soma_z_scale'] - 30,
                 r'$R_{\rm pop}$', va='top', ha='left')

        # Draw synaptic input box
        patches = []
        ellipse_top = mpatches.Ellipse([0, syn_z_mean + syn_z_scale],
                                        pop_params['pop_radius'] * 4, 100)
        ellipse_bottom = mpatches.Ellipse([0, syn_z_mean - syn_z_scale],
                                        pop_params['pop_radius'] * 4, 100)
        patches.append(ellipse_top)
        patches.append(ellipse_bottom)
        collection = PatchCollection(patches,
                                     edgecolors='k', facecolors='none',
                                     ls='--', lw=0.5)
        ax1.add_collection(collection)
        ax1.plot([-pop_params['pop_radius'] * 2] * 2,
                 [syn_z_mean + syn_z_scale,
                  syn_z_mean - syn_z_scale],
                 ls='--', c='k', lw=0.5)
        ax1.plot([pop_params['pop_radius'] * 2] * 2,
                 [syn_z_mean + syn_z_scale,
                  syn_z_mean - syn_z_scale],
                 ls='--', c='k', lw=0.5)

        # ax1.plot([0, pop_params['pop_radius']],
        #          [syn_z_mean - syn_z_scale] * 2,
        #          ls='--', c='k', lw=0.5)
        # ax1.text(pop_params['pop_radius'] / 2,
        #          soma_z_mean - syn_z_scale - 30,
        #          r'$R_{\rm pop}$', va='top', ha='left')

        ax1.text(pop_params['pop_radius'] * 1.05,
                 soma_z_mean,
                 r'$\Delta{z}$', va='center', ha='left')

        ax2 = fig.add_axes([0.35, 0.91, 0.15, 0.05],
                           xlabel="soma location (µm)", yticks=[])
        ax3 = fig.add_axes([0.35, 0.71, 0.15, 0.05],
                           xlabel="synaptic locations (µm)", yticks=[])
        ax4 = fig.add_axes([0.35, 0.51, 0.15, 0.05],
                           xlabel=r"$w_{\rm syn}$ (nA)", yticks=[])
        ax5 = fig.add_axes([0.35, 0.31, 0.15, 0.05],
                           xlabel=r"$\tau_{\rm syn}$ (ms)", yticks=[])
        ax6 = fig.add_axes([0.35, 0.11, 0.15, 0.05],
                           xlabel=r"synaptic delay (ms)", yticks=[])

        x_ = np.linspace(-1600, 0, 100)
        ax2.plot(x_, soma_loc_dist.pdf(x_), c='k', clip_on=False)

        x_ = np.linspace(-1400, 0, 100)
        ax3.plot(x_, syn_loc_dist.pdf(x_), c='k', clip_on=False)

        x_ = np.linspace(0, pop_params['weight_mean'] * 4, 100)
        ax4.plot(x_, syn_weight_dist.pdf(x_), c='k', clip_on=False)

        x_ = np.linspace(0, 4, 100)
        ax5.plot(x_, syn_tau_dist.pdf(x_), c='k', clip_on=False)

        x_ = np.linspace(0, 2, 100)
        ax6.plot(x_, syn_delay_dist.pdf(x_), c='k', clip_on=False)

    lfp_kernel = np.zeros((num_elecs, len(cell.tvec)))
    cdm = np.zeros((3, len(cell.tvec)))
    # single_lfp_kernels = np.zeros((pop_params['num_cells'],
    #                                num_elecs,
    #                                num_tsteps))
    tstop = cell.tstop

    for cell_idx in range(pop_params['num_cells']):
        if cell_idx % 10 == 0:
            print(f"Cell idx: ", cell_idx)
        # t0 = time.time()
        # if cell_idx % 100 == 0:
        #     print(f'{cell_idx} of {pop_params["num_cells"]}')
        del cell
        cell = return_hay_cell(tstop, dt, make_passive=True)
        cell.set_pos(x=soma_xs[cell_idx],
                     y=soma_ys[cell_idx],
                     z=soma_zs[cell_idx],
                     )
        cell.set_rotation(z=z_rots[cell_idx])
        syn_idxs = cell.get_rand_idx_area_and_distribution_norm(nidx=num_syns,
                                                                fun=syn_loc_func,
                                                                funargs=syn_loc_params)
        syn_weights = syn_weight_dist.rvs(pop_params['in_degree'])
        syn_taus = syn_tau_dist.rvs(pop_params['in_degree'])
        syn_delays = syn_delay_dist.rvs(pop_params['in_degree'])
        if cell_idx == 0:
            plot_syn_histograms(cell, syn_idxs, syn_weights,
                                syn_taus, syn_delays)

        assert np.max(cell.z) < 0, f"CELL EXTRUDING FROM BRAIN: {np.max(cell.z)}"
        input_times = np.ones(num_syns) * input_time
        insert_synaptic_input(cell, syn_idxs,
                                      syn_weights, syn_taus,
                                      syn_delays, input_times)
        elec = LFPy.RecExtElectrode(cell, **elec_params)
        cell.simulate(rec_imem=True, rec_vmem=False, probes=[elec])

        # cell = get_membrane_currents(cell, imem_delta_mapping, syn_idxs,
        #                              syn_weights, syn_taus,
        #                              syn_delays, input_time)

        # cell_lfp = elec.get_transformation_matrix() @ cell.imem * 1000
        cell_lfp = elec.data * 1000
        # print("MAX Ve:", np.max(np.abs(elec.data * 1000)))

        # single_lfp_kernels[cell_idx] = cell_lfp
        lfp_kernel += cell_lfp
        cdm += CurrentDipoleMoment(cell).get_transformation_matrix() @ cell.imem

        # print("Time for single sim: ", time.time() - t0)

        if plot_results and (cell_idx <= num_plot_cells):
            cell_clr_ = plt.cm.Greys(0.1 + cell_idx / num_plot_cells / 5)
            ax1.plot(cell.x.T, cell.z.T, c=cell_clr_, zorder=-1e9)
            ax1.plot(cell.x[syn_idxs].mean(axis=1),
                     cell.z[syn_idxs].mean(axis=1), 'o', c=syn_clr, ms=2,
                     zorder=-1e9)

    if plot_results:
        lfp_norm = np.max(np.abs(lfp_kernel)) * 1.1

        ax_lfp = fig.add_axes([0.65, 0.1, 0.3, 0.8], xlim=[0, 40],
                              xlabel="time (ms)",
                              ylabel="electrode location (µm)")
        for elec_idx in range(num_elecs):
            ax_lfp.plot(cell.tvec,
                        lfp_kernel[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                        c='k', lw=1.)
            # for cell_idx in range(pop_params['num_cells']):
            #     ax_lfp.plot(cell.tvec,
            #                 single_lfp_kernels[cell_idx, elec_idx] / lfp_norm * dz + elec_z[elec_idx],
            #                 c='gray', lw=0.5)

        ax_lfp.plot([32, 32], [-100, -300], c='k', lw=1, clip_on=False)
        ax_lfp.text(31.6, -200, f'{lfp_norm:0.2f}\nµV', va='center', ha='right')
        simplify_axes(fig.axes)
        mark_subplots(ax1, "A")
        fig.savefig(join(save_folder, f'kernel_instance_{sim_name}_{trial_idx}.pdf'))
        fig.savefig(join(save_folder, f'kernel_instance_{sim_name}_{trial_idx}.png'))
        plt.close("all")
    del cell
    return lfp_kernel, cdm


def run_through_trials(sim_name, pop_params, tstop, num_trials, soma_z_mean,
                       input_time, save_folder):


    if rank == 0:
        print(sim_name, pop_params)
    for trial_idx in range(num_trials):
        if trial_idx % size == rank:

            if not os.path.isfile(join(save_folder, f"kernels_case_{sim_name}_{trial_idx}.npy")):
                print(f'Trial number: {trial_idx}')
                kernel_instance, cdm = calculate_kernel_instance(soma_z_mean,
                                                            input_time,
                                                            pop_params, trial_idx,
                                                            sim_name,
                                                            tstop,
                                                            save_folder,
                                                            plot_results=trial_idx <= 5)

                np.save(join(save_folder, f"kernels_case_{sim_name}_{trial_idx}.npy"), kernel_instance)
                np.save(join(save_folder, f"kernels_cdm_case_{sim_name}_{trial_idx}.npy"), cdm)
            # else:
            #     kernel_instance = np.load(join(save_folder, f"kernels_case_{sim_name}_{trial_idx}.npy"),
            #                               allow_pickle=True)[()]['kernel_trial']


    trial_kernels = []
    trial_cdms = []
    comm.Barrier()
    if rank == 0:
        print("Collecting, and calculating K_eq and K_neq")
        for trial_idx in range(num_trials):
            kernel_instance = np.load(join(save_folder,
                                           f"kernels_case_{sim_name}_{trial_idx}.npy"))
            cdm_instance = np.load(join(save_folder,
                                           f"kernels_cdm_case_{sim_name}_{trial_idx}.npy"))
            trial_kernels.append(kernel_instance)
            trial_cdms.append(cdm_instance)

        k_ = np.array(trial_kernels)
        cdm_ = np.array(trial_cdms)

        lfp_kernels = np.zeros((k_.shape[0], k_.shape[1], k_.shape[2] * 2 + 1))
        lfp_kernels[:, :, -k_.shape[-1]:] = k_

        cdm_kernels = np.zeros((cdm_.shape[0], cdm_.shape[1], cdm_.shape[2] * 2 + 1))
        cdm_kernels[:, :, -cdm_.shape[-1]:] = cdm_

        K_equal = calc_K_equal(lfp_kernels)
        K_nonequal = calc_K_nonequal(lfp_kernels)

        save_dict = {'parameters':  pop_params,
                     'kernel_trials': lfp_kernels,
                     'cdm_trials': cdm_kernels,
                     'K_equal': K_equal,
                     'K_nonequal': K_nonequal}

        np.save(join(save_folder, f"kernels_case_{sim_name}.npy"), save_dict)


def return_pop_param_dict(sim_name):

    ###
    # sim_name = 'low_variability'
    # pop_params = default_pop_params.copy()
    # pop_params['pop_radius'] = 25
    # pop_params['weight_scale'] = 1e-9
    # pop_params['syn_delay_scale'] = 1e-9
    # pop_params['tau_s_scale'] = 1e-9
    # pop_params['soma_z_scale'] = 10
    # run_through_trials(sim_name, pop_params, num_trials, cell, soma_z_mean,
    #                    input_time, imem_delta_mapping, sim_data_folder)

    pop_params = default_pop_params.copy()
    if sim_name == 'default':
        pass
    elif sim_name == 'apical':
        pop_params['syn_z_mean'] = -200
    elif sim_name ==  'uniform':
        pop_params['syn_z_mean'] = -600
        pop_params['syn_z_scale'] = 1e8
    elif sim_name == 'small_radius':
        pop_params['pop_radius'] /= 2
    elif sim_name == 'large_radius':
        pop_params['pop_radius'] *= 2
    elif sim_name == 'small_population':
        pop_params['num_cells'] = int(pop_params['num_cells'] / 2)
    elif sim_name == 'large_population':
        pop_params['num_cells'] = int(pop_params['num_cells'] * 2)
    elif sim_name == 'similar_synapses':
        pop_params['weight_scale'] /= 2
        pop_params['weight_scale_lognormal'] /= 2
        pop_params['syn_delay_scale'] /= 2
        pop_params['tau_s_scale'] /= 2
    elif sim_name == 'variable_synapses':
        pop_params['weight_scale'] *= 2
        pop_params['weight_scale_lognormal'] *= 2
        pop_params['syn_delay_scale'] *= 2
        pop_params['tau_s_scale'] *= 2
    elif sim_name == 'narrow_input_region':
        pop_params['syn_z_scale'] /= 2
    elif sim_name == 'broad_input_region':
        pop_params['syn_z_scale'] *= 2
    return pop_params


def kernel_heterogeneities(sim_data_folder):

    num_trials = 100
    tstop = 50
    input_time = 0

    os.makedirs(sim_data_folder, exist_ok=True)
    cell = return_hay_cell(tstop, dt, make_passive=True)

    cell_z_top = np.max(cell.z)
    soma_z_mean = -cell_z_top - 150
    del cell

    sim_names = ['default', 'apical', 'uniform',
                 'small_radius', 'large_radius',
                 'small_population', 'large_population',
                 'similar_synapses', 'variable_synapses',
                 'narrow_input_region', 'broad_input_region']

    for sim_name in sim_names:
        pop_params = return_pop_param_dict(sim_name)
        run_through_trials(sim_name, pop_params, tstop, num_trials, soma_z_mean,
                           input_time, sim_data_folder)


def make_neural_pop_setup_fig(sim_data_folder):

    tstop = 50

    cell = return_hay_cell(tstop, dt, make_passive=True)

    cell_z_top = np.max(cell.z)
    soma_z_mean = -cell_z_top - 150
    del cell

    np.random.seed(123456)
    cell = return_hay_cell(tstop=tstop, dt=dt)

    cell.set_pos(x=0, y=0, z=soma_z_mean)

    cell_z_top = np.max(cell.z)
    max_soma_loc = -cell_z_top + cell.z[0].mean() - 10

    sim_name_a = 'apical'
    sim_name_b = 'default'
    sim_name_u = 'uniform'
    pop_params = default_pop_params.copy()
    pop_params['syn_z_mean'] = -200

    syn_weight_dist = return_syn_weight_dist(pop_params)
    soma_loc_dist, soma_xs, soma_ys, soma_zs, z_rots = \
        return_soma_loc_dist(pop_params, soma_z_mean, max_soma_loc)
    syn_delay_dist = return_syn_delay_dist(pop_params)
    syn_loc_dist, syn_loc_func, syn_loc_params = return_syn_loc_dist(pop_params)
    syn_tau_dist = return_syn_tau_dist(pop_params)

    data_dict_a = np.load(join(sim_data_folder, f"kernels_case_{sim_name_a}.npy"),
                        allow_pickle=True)[()]
    data_dict_b = np.load(join(sim_data_folder, f"kernels_case_{sim_name_b}.npy"),
                        allow_pickle=True)[()]
    data_dict_u = np.load(join(sim_data_folder, f"kernels_case_{sim_name_u}.npy"),
                        allow_pickle=True)[()]

    lfp_kernel_a = np.array(data_dict_a['kernel_trials'])
    lfp_mean_a = lfp_kernel_a.mean(axis=0)
    lfp_std_a = np.max(lfp_kernel_a.std(axis=0), axis=1)
    eeg_a = np.array(data_dict_a['cdm_trials'])[:, 2, :] * M_eeg
    eeg_mean_a = eeg_a.mean(axis=0)

    lfp_kernel_b = np.array(data_dict_b['kernel_trials'])
    lfp_mean_b = lfp_kernel_b.mean(axis=0)
    lfp_std_b = np.max(lfp_kernel_b.std(axis=0), axis=1)
    eeg_b = np.array(data_dict_b['cdm_trials'])[:, 2, :] * M_eeg
    eeg_mean_b = eeg_b.mean(axis=0)


    lfp_kernel_u = np.array(data_dict_u['kernel_trials'])
    lfp_mean_u = lfp_kernel_u.mean(axis=0)
    lfp_std_u = np.max(lfp_kernel_u.std(axis=0), axis=1)
    eeg_u = np.array(data_dict_u['cdm_trials'])[:, 2, :] * M_eeg
    eeg_mean_u = eeg_u.mean(axis=0)

    num_kernels = lfp_kernel_a.shape[0]
    tvec_k = np.arange(lfp_kernel_a.shape[2]) * dt
    tvec_k -= tvec_k[int(len(tvec_k) / 2)]
    cell.set_pos(x=0, y=0, z=soma_z_mean)

    num_syns = pop_params['in_degree']

    num_plot_cells = 100

    ylim = [-1820, 300]
    xlim = [-500, 500]
    plt.close('all')
    fig = plt.figure(figsize=[6, 3])

    ax_spks = fig.add_axes([0.01, 0.85, 0.1, 0.03], frameon=False,
                           title="spike train",
                       xticks=[], yticks=[], ylim=[-3, 3], xlim=[0, 20], zorder=1)

    ax1 = fig.add_axes([0.08, 0.02, 0.3, 0.93], frameon=False, aspect=1,
                       xticks=[], yticks=[], ylim=ylim, xlim=xlim,
                       rasterized=True, zorder=0)

    ax2 = fig.add_axes([0.34, 0.02, 0.03, 0.93],
                       xticks=[], sharey=ax1, frameon=False)

    ax2.set_title("soma depth\ndistribution", c=soma_clr,
                  y=0.1, x=1.6, rotation=-90)

    ax3 = fig.add_axes([0.34, 0.02, 0.03, 0.93],
                       xticks=[], sharey=ax1, frameon=False)

    ax3.set_title("synapse depth\ndistribution", c=syn_clr,
                  y=0.57, x=1.6, rotation=-90)
    ax4 = fig.add_axes([0.02, 0.59, 0.09, 0.09],
                       xlabel=r"$J$ (pA)", yticks=[])
    ax5 = fig.add_axes([0.02, 0.35, 0.09, 0.09],
                       xlabel=r"$\tau_{\rm syn}$ (ms)", yticks=[])
    ax6 = fig.add_axes([0.02, 0.11, 0.09, 0.09],
                       xlabel="delay (ms)", yticks=[])

    ax_lfp_a = fig.add_axes([0.44, 0.02, 0.11, 0.93], xlim=[0, 15],
                            xticks=[],
                            frameon=False,
                            yticks=[],
                            sharey=ax1,)

    ax_lfp_b = fig.add_axes([0.57, 0.02, 0.11, 0.93], xlim=[0, 15],
                            xticks=[],
                            frameon=False,
                            yticks=[],
                            sharey = ax1,
                            )

    ax_lfp_u = fig.add_axes([0.7, 0.02, 0.11, 0.93], xlim=[0, 15],
                            xticks=[],
                            frameon=False,
                            yticks=[],
                            sharey=ax1,
                            )
    ax_amp = fig.add_axes([0.83, 0.02, 0.15, 0.93],
                          xlabel="µV",
                          sharey=ax1,
                          xlim=[-6, 6]
                          )

    mark_subplots(ax_spks, "A", ypos=4., xpos=0.05)
    mark_subplots(ax_lfp_a, "B", ypos=1.01)
    mark_subplots(ax_lfp_b, "C", ypos=1.01)
    mark_subplots(ax_lfp_u, "D", ypos=1.01)
    mark_subplots(ax_amp, "E", ypos=1.01)


    ax_amp.spines['left'].set_visible(False)
    ax_amp.spines['bottom'].set_position(('axes', 0.1))
    ax_lfp_a.set_title("apical\ninput", y=0.94)
    ax_lfp_b.set_title('basal\ninput', y=0.94)
    ax_lfp_u.set_title("uniform\ninput", y=0.94)

    ax_amp.plot([0, 0], [-1600, 0], lw=0.5, ls='--', c='gray')

    max_a = lfp_mean_a[np.arange(num_elecs), np.argmax(np.abs(lfp_mean_a), axis=1)]
    max_b = lfp_mean_b[np.arange(num_elecs), np.argmax(np.abs(lfp_mean_b), axis=1)]
    max_u = lfp_mean_u[np.arange(num_elecs), np.argmax(np.abs(lfp_mean_u), axis=1)]

    la, = ax_amp.plot(max_a, elec_z, c=cond_clr['apical'])
    ax_amp.fill_betweenx(elec_z, max_a-lfp_std_a,
                         max_a+lfp_std_a, fc=cond_clr['apical'], alpha=0.3)
    lb, = ax_amp.plot(max_b, elec_z, c=cond_clr['default'])
    ax_amp.fill_betweenx(elec_z, max_b-lfp_std_b,
                         max_b+lfp_std_b, fc=cond_clr['default'], alpha=0.3)

    lu, = ax_amp.plot(max_u, elec_z, c=cond_clr['uniform'])
    ax_amp.fill_betweenx(elec_z, max_u-lfp_std_u,
                         max_u+lfp_std_u, fc=cond_clr['uniform'], alpha=0.3)

    ax_amp.legend([la, lb, lu], ["apical input",
                                 "basal input",
                                 "uniform input"], frameon=False, loc=(-0.05, 0.89))

    dummy_spikes = [0, 2, 8, 9, 11, 17,]
    ax_spks.plot(dummy_spikes, np.zeros(len(dummy_spikes)),
                 marker='|', ls='', clip_on=False, c='k', lw=0.1)

    ax_spks.plot([18, 21, 21, 25], [0, 0, -25, -25], clip_on=False, c='k', lw=0.5)

    ax_spks.plot([25, 25], [-25, -15], clip_on=False, c='k', lw=0.5)
    ax_spks.plot([25, 25], [-25, -35], clip_on=False, c='k', lw=0.5)

    ax_spks.arrow(25, -15, dx=2, dy=0, clip_on=False, lw=0.5, head_width=1, fc='k')
    ax_spks.arrow(25, -20, dx=2, dy=0, clip_on=False, lw=0.5, head_width=1, fc='k')
    ax_spks.arrow(25, -25, dx=2, dy=0, clip_on=False, lw=0.5, head_width=1, fc='k')
    ax_spks.arrow(25, -30, dx=2, dy=0, clip_on=False, lw=0.5, head_width=1, fc='k')
    ax_spks.arrow(25, -35, dx=2, dy=0, clip_on=False, lw=0.5, head_width=1, fc='k')
    ax_spks.text(21, -25, "{", fontsize=25, va='center', ha="right")
    ax_spks.text(15, -25, r"$K_{\rm out}$", va='center', ha="right")

    ax1.axhspan(0, 100, color=csf_clr)
    ax1.axhspan(ylim[0], 0, zorder=-1e9, color=brain_clr)
    ax1.scatter(soma_xs, soma_zs, c='k', s=2)

    # Draw EEG electrode
    ax1.plot([0, 0, 0], [125, 150, 175], 'k.', ls='none', ms=1)
    ax1.plot(elec_x, elec_z, c=elec_clr, marker='o', ls='none', ms=3)
    collection = PatchCollection([mpatches.Ellipse([0, 250],
                                                   1000, 100)],
                                 color=eeg_clr, clip_on=False)
    ax1.add_collection(collection)
    ax1.text(-200, 330, "EEG electrode", color=eeg_clr)

    ax1.plot([300, 300], [-900, -800], lw=1, c='k', zorder=1e10)
    ax1.text(310, -850, "100\nµm", va="center", ha="left", color='k', zorder=1e10)

    x_ = np.linspace(-1600, -1000, 250)
    ax2.plot(soma_loc_dist.pdf(x_) / np.max(soma_loc_dist.pdf(x_)) * 100 + 600,
             x_, c='k', clip_on=False)

    x_ = np.linspace(-600, 0, 250)
    ax3.plot(syn_loc_dist.pdf(x_)/ np.max(syn_loc_dist.pdf(x_)) * 100 + 600,
             x_ , c=syn_clr, clip_on=False)

    x_ = np.linspace(0, pop_params['weight_mean'] * 4, 100)
    ax4.plot(x_ * 1e3, syn_weight_dist.pdf(x_), c='k', clip_on=False)

    x_ = np.linspace(0, 4, 100)
    ax5.plot(x_, syn_tau_dist.pdf(x_), c='k', clip_on=False)

    x_ = np.linspace(0, 2, 100)
    ax6.plot(x_, syn_delay_dist.pdf(x_), c='k', clip_on=False)

    ax1.plot([0, pop_params["pop_radius"]],
             [ylim[0] + 150]*2, c='k', lw=1, ls='-', clip_on=False, marker='|')
    ax1.text(pop_params["pop_radius"]/2, ylim[0] + 100,
             r"$R_{\rm pop}$", va="top", ha="center")

    for k_idx in range(num_plot_cells):
        if k_idx % 100 == 0:
            print(f'{k_idx} of {pop_params["num_cells"]}')
        cell.set_pos(x=soma_xs[k_idx],
                     y=soma_ys[k_idx],
                     z=soma_zs[k_idx],
                     )
        cell.set_rotation(z=z_rots[k_idx])
        syn_idxs = cell.get_rand_idx_area_and_distribution_norm(nidx=num_syns,
                                                                fun=syn_loc_func,
                                                                funargs=syn_loc_params)

        assert np.max(cell.z) < 0, f"CELL EXTRUDING FROM BRAIN: {np.max(cell.z)}"

        cell_clr_ = plt.cm.Greys(0.1 + k_idx / num_plot_cells / 5)
        ax1.plot(cell.x.T, cell.z.T, c=cell_clr_, zorder=-1e9)
        ax1.plot(cell.x[syn_idxs].mean(axis=1),
                 cell.z[syn_idxs].mean(axis=1), 'o', c=syn_clr, ms=2,
                 zorder=-1e6)

    norm_scale = 1
    max_lfp = 4#np.max(np.abs(lfp_mean_a))
    max_eeg = 1  # nV
    eeg_plot_loc = 120
    lfp_norm = max_lfp / norm_scale
    for elec_idx in range(num_elecs)[::2]:
        for k_idx in range(num_kernels):
            ax_lfp_a.plot(tvec_k,
                        lfp_kernel_a[k_idx, elec_idx] / lfp_norm * dz
                        + elec_z[elec_idx],
                        c='0.8', lw=0.5, zorder=0)
            ax_lfp_b.plot(tvec_k,
                        lfp_kernel_b[k_idx, elec_idx] / lfp_norm * dz
                        + elec_z[elec_idx],
                        c='0.8', lw=0.5, zorder=0)

            ax_lfp_u.plot(tvec_k,
                        lfp_kernel_u[k_idx, elec_idx] / lfp_norm * dz
                        + elec_z[elec_idx],
                        c='0.8', lw=0.5, zorder=0)


        ax_lfp_a.plot(tvec_k,
                    lfp_mean_a[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                    c='k', lw=1., zorder=1)
        ax_lfp_b.plot(tvec_k,
                    lfp_mean_b[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                    c='k', lw=1., zorder=1)
        ax_lfp_u.plot(tvec_k,
                    lfp_mean_u[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                    c='k', lw=1., zorder=1)

        ax_lfp_a.plot(elec_x[elec_idx], elec_z[elec_idx], 'o',
                      c=elec_clr, ms=3., clip_on=False)
        ax_lfp_b.plot(elec_x[elec_idx], elec_z[elec_idx], 'o',
                      c=elec_clr, ms=3., clip_on=False)
        ax_lfp_u.plot(elec_x[elec_idx], elec_z[elec_idx], 'o',
                      c=elec_clr, ms=3., clip_on=False)

    ax_lfp_a.text(1, np.max(elec_z) + 20, "LFP", color=elec_clr,
                  ha='right', va="bottom")

    # Plot EEG kernels
    ax_lfp_a.text(1, eeg_plot_loc + 25, "EEG", color=eeg_clr,
                  ha='right', va="bottom")
    for k_idx in range(num_kernels):
        ax_lfp_a.plot(tvec_k,
                    eeg_a[k_idx] / max_eeg * dz + eeg_plot_loc,
                    c='0.8', lw=1., zorder=1)
        ax_lfp_b.plot(tvec_k,
                    eeg_b[k_idx] / max_eeg * dz + eeg_plot_loc,
                    c='0.8', lw=1., zorder=1)
        ax_lfp_u.plot(tvec_k,
                    eeg_u[k_idx] / max_eeg * dz + eeg_plot_loc,
                    c='0.8', lw=1., zorder=1)

    ax_lfp_a.plot(tvec_k,
                eeg_mean_a / max_eeg * dz + eeg_plot_loc,
                c='k', lw=1., zorder=1)

    ax_lfp_b.plot(tvec_k,
                eeg_mean_b / max_eeg * dz + eeg_plot_loc,
                c='k', lw=1., zorder=1)
    ax_lfp_u.plot(tvec_k,
                eeg_mean_u / max_eeg * dz + eeg_plot_loc,
                c='k', lw=1., zorder=1)

    ax_lfp_a.plot(0, eeg_plot_loc, 'o', c=eeg_clr, clip_on=False)
    ax_lfp_b.plot(0, eeg_plot_loc, 'o', c=eeg_clr, clip_on=False)
    ax_lfp_u.plot(0, eeg_plot_loc, 'o', c=eeg_clr, clip_on=False)

    ax_lfp_a.plot([14, 14], [eeg_plot_loc, eeg_plot_loc + dz], lw=1,
                  clip_on=False, color=eeg_clr)
    ax_lfp_a.text(13.5, eeg_plot_loc + dz/2, f'{max_eeg:0.0f} nV',
                  va='center', ha='right', color=eeg_clr)

    ax_lfp_a.plot([17, 17], [-1650, -1650 - dz * norm_scale], c='k', lw=1,
                  clip_on=False)
    ax_lfp_a.text(16, -1650 - dz/2, f'{max_lfp:0.0f} µV',
                  va='center', ha='right')

    ax_lfp_b.plot([5, 15], [-1650, -1650], c='k', lw=1, clip_on=False)
    ax_lfp_b.text(10, -1670, f'10 ms', va='top', ha='center')

    simplify_axes(fig.axes)
    mark_subplots(ax1, "A")
    fig.savefig(f'neural_pop_setup_fig_eeg.pdf')
    plt.close("all")


if __name__ == '__main__':
    sim_data_folder = 'simulated_pop_kernels'

    kernel_heterogeneities(sim_data_folder)
    make_neural_pop_setup_fig(sim_data_folder)

