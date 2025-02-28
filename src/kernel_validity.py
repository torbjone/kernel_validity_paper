import os
import sys
from os.path import join
import numpy as np
import scipy.fftpack as ff
import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from elephant.spike_train_generation import (single_interaction_process,
                                             homogeneous_poisson_process,
                                             inhomogeneous_poisson_process)
from scipy import interpolate
import neo
from quantities import s, Hz, ms
from elephant.conversion import BinnedSpikeTrain
from src.plotting_convention import mark_subplots, simplify_axes
from src.correlation_toolbox import correlation_analysis as ca
from src import neural_simulations

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'Rank {rank} of {size}')


elec_x = neural_simulations.elec_x
elec_z = neural_simulations.elec_z
dz = np.abs(elec_z[1] - elec_z[0])
np.random.seed(1234)
M_eeg = neural_simulations.M_eeg

class KernelValidity:

    def __init__(self, num_kernels, kernels_param_dict,
                 signal_length, correlation, correlation_type, dt, rate,
                 jitter, figure_folder, calc_error_from_theory=False, **kwargs):

        self.num_kernels = num_kernels

        self.dt = dt
        self.cutoff_idxs = int(100 / dt)  # cut off 100 ms at each end
        self.kernels_param_dict = kernels_param_dict
        self.make_kernels()

        self.tvec_kernel = np.arange(self.kernel_length) * self.dt

        self.correlation = correlation
        self.correlation_type = correlation_type
        self.jitter = jitter

        self.signal_length = signal_length + self.kernel_length

        self.tvec_signal = np.arange(self.signal_length) * self.dt

        self.t_stop = self.tvec_signal[-1] + self.dt
        self.t_start = self.tvec_signal[0]
        self.rate = rate # Hz
        try:
            self.avrg_num_spikes_per_kernel = int(self.t_stop / 1000 * self.rate)
        except:
            self.avrg_num_spikes_per_kernel = None
        self.kwargs = kwargs
        self.figure_folder = figure_folder
        self.data_folder = kwargs["data_folder"]
        os.makedirs(self.figure_folder, exist_ok=True)

        self.make_spike_times()

        self.sim_name = self.return_sim_name()
        print(self.sim_name)

        self.make_kernel_signal()

        self.make_ground_truth_signal()

        # plt.close("all")
        # plt.subplot(311)
        # plt.plot(self.tvec_signal, self.avrg_firingrate, lw=3)
        #
        # plt.plot(self.tvec_signal, self.firing_rate[0], lw=2)
        # plt.plot(self.tvec_signal, self.firing_rate[1], lw=1)
        # plt.plot(self.tvec_signal, self.firing_rate[2], lw=0.5)
        #
        # plt.subplot(312)
        # elec_idx = 12
        # print(self.kernels.shape)
        # plt.plot(self.tvec_kernel, self.kernels[0, elec_idx], lw=1)
        # plt.plot(self.tvec_kernel, self.kernels[1, elec_idx], lw=1)
        # plt.plot(self.tvec_kernel, self.kernels[2, elec_idx], lw=1)
        # plt.plot(self.tvec_kernel, self.avrg_kernel[elec_idx], 'k', lw=1)
        #
        # plt.show()
        #
        # sys.exit()
        self.cutoff_signal_ends()

        self.calc_firing_rate_error_measures(kwargs['firing_rate_folder'])
        self.evaluate_difference()

        if calc_error_from_theory:
            self.calc_error_from_theory()

    def cutoff_signal_ends(self):
        if hasattr(self, 'num_elecs'):
            self.phi = self.phi[:, self.cutoff_idxs:-self.cutoff_idxs]
            self.phi_tilde = self.phi_tilde[:, self.cutoff_idxs:-self.cutoff_idxs]
        else:
            self.phi = self.phi[self.cutoff_idxs:-self.cutoff_idxs]
            self.phi_tilde = self.phi_tilde[self.cutoff_idxs:-self.cutoff_idxs]

        if hasattr(self, 'cdm'):
            self.cdm = self.cdm[:, self.cutoff_idxs:-self.cutoff_idxs]
            self.cdm_tilde = self.cdm_tilde[:, self.cutoff_idxs:-self.cutoff_idxs]

        self.firing_rate = self.firing_rate[:, self.cutoff_idxs:-self.cutoff_idxs]
        self.avrg_firingrate = self.avrg_firingrate[self.cutoff_idxs:-self.cutoff_idxs]
        self.std_firingrate = self.std_firingrate[self.cutoff_idxs:-self.cutoff_idxs]

        self.tvec_signal = np.arange(self.firing_rate.shape[1]) * self.dt

    def calc_firing_rate_error_measures(self, firing_rate_folder):

        Df = neural_simulations.dt
        # This gives covariances:
        fr__ = ca.cthlp.centralize(self.firing_rate, time=True, units=False)
        freq, p_p = ca.powerspec(fr__, self.dt, Df=Df, units=True)
        self.t_a, self.a_mean = ca.autocorrfunc(freq, p_p)
        freq, p_c = ca.crossspec(fr__, self.dt, Df=Df, units=True)
        self.t_c, self.c_mean = ca.crosscorrfunc(freq, p_c)

        # This gives auto/cross-correlations:
        freq2, p_p2 = ca.powerspec(self.firing_rate, self.dt, Df=Df, units=True)
        t_a2, a_mean_cor = ca.autocorrfunc(freq2, p_p2)

        freq2, p_c2 = ca.crossspec(self.firing_rate, self.dt, Df=Df, units=True)
        t_c2, c_mean_cor = ca.crosscorrfunc(freq2, p_c2)

        fr_part_name = f"corrtype:{self.correlation_type}"
        if not 'brunel' in self.correlation_type:
            fr_part_name += f'_{self.correlation}_{self.rate}_{self.jitter}'
        fr_dict = {
            'firing_rate': self.firing_rate,
            't_a': self.t_a,
            'a_mean': self.a_mean,
            'a_mean_cor': a_mean_cor,
            't_c': self.t_c,
            'c_mean': self.c_mean,
            'c_mean_cor': c_mean_cor,
        }
        os.makedirs(firing_rate_folder, exist_ok=True)
        np.save(join(firing_rate_folder, f'{fr_part_name}.npy'), fr_dict)

    @staticmethod
    def calc_K_equal(kernels):
        """
        Takes numpy array with num_kernels * num_electrodes * num_kernel_tsteps
        :param kernels:
        :return:
        """
        K_equal = np.einsum('bai,baj->aij', kernels, kernels)
        return K_equal / kernels.shape[0]

    @staticmethod
    def calc_K_nonequal(kernels):
        K_nonequal = (np.einsum('bai,caj->aij', kernels, kernels) -
                      np.einsum('bai,baj->aij', kernels, kernels))
        return K_nonequal / (kernels.shape[0] * (kernels.shape[0] - 1))

    def return_sim_name(self):
        if self.kernels_param_dict['kernel_type'] == 'toy_Exp2Syn':

            k_part = "amp:{:1.2f}({:1.2f})_tau1:{:1.2f}({:1.2f})_tau2:{:1.2f}({:1.2f})".format(
                self.amp_mean, self.amp_std, self.tau_1_mean, self.tau_1_std,
                self.tau_2_mean, self.tau_2_std, )
        elif self.kernels_param_dict['kernel_type'] == 'toy_ExpSyn':
            k_part = "amp:{:1.2f}({:1.2f})_tau1:{:1.2f}({:1.2f})".format(
                self.amp_mean, self.amp_std, self.tau_1_mean, self.tau_1_std)
        else:
            k_part = f'{self.kernels_param_dict["case_name"]}'

        if 'brunel' in self.correlation_type:
            fr_part = f'corrtype:{self.correlation_type}'
        else:
            fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(self.rate,
                                                           self.correlation_type,
                                                           self.correlation,
                                                           self.jitter)

        sim_name = "num_kernels:{}_".format(self.num_kernels) + k_part + '_' + fr_part
        return sim_name

    def calc_error_from_theory(self):

        if not hasattr(self, 'K_equal'):
            if hasattr(self, 'num_elecs'):
                print("Finding K_equal ...")
                K_equal = self.calc_K_equal(self.kernels[:, :, :])
                print("Finding K_nonequal ...")
                K_nonequal = self.calc_K_nonequal(self.kernels[:, :, :])
            else:
                # None because the code assumes several electrodes
                K_equal = self.calc_K_equal(self.kernels[:, None, :])[0, :, :]
                K_nonequal = self.calc_K_nonequal(self.kernels[:, None, :])[0, :, :]
        else:
            K_equal = self.K_equal
            K_nonequal = self.K_nonequal


        if hasattr(self, 'num_elecs'):
            E = np.zeros(self.num_elecs)
            E2 = np.zeros(self.num_elecs)
            f_ = self.a_mean - self.c_mean
            f2_ = np.zeros(f_.shape)
            K_diff = K_equal - K_nonequal

            idx_0 = np.argmin(np.abs(self.t_a))#[0]
            f2_[idx_0] = np.max(self.a_mean)
            for elec_idx in range(self.num_elecs):
                for k_idx in range(self.kernel_length):
                    for l_idx in range(self.kernel_length):
                        if l_idx - k_idx + idx_0 >= 0:
                            if l_idx - k_idx + idx_0 < len(f_):
                                E[elec_idx] += K_diff[elec_idx, k_idx, l_idx] * f_[l_idx - k_idx + idx_0]
                    E2[elec_idx] += K_diff[elec_idx, k_idx, k_idx] * f2_[idx_0]
            self.error_theory = E * (self.num_kernels - 1) * (self.dt / 1000)**2

        else:
            E = 0
            f_ = self.a_mean - self.c_mean
            K_diff = K_equal - K_nonequal
            idx_0 = np.where(self.t_a == 0.0)[0][0]
            for k_idx in range(self.kernel_length):
                for l_idx in range(self.kernel_length):
                    if l_idx - k_idx + idx_0 >= 0:
                        if l_idx - k_idx + idx_0 < len(f_):
                            E += K_diff[k_idx, l_idx] * f_[l_idx - k_idx + idx_0]

            self.error_theory = E * (self.num_kernels - 1) * (self.dt / 1000) **2

    def make_spike_times(self):
        self.firing_rate = np.zeros((self.num_kernels, self.signal_length))

        if self.correlation_type.startswith("brunel_"):
            self._load_brunel_spikes()
        elif self.correlation_type == "mip":
            self._make_spiketimes_mip()
        else:
            raise RuntimeError("Unrecognized correlation_type")

        self.firing_rate /= (self.dt*1e-3)
        self.avrg_firingrate = np.average(self.firing_rate, axis=0)
        self.std_firingrate = np.std(self.firing_rate, axis=0)

    def _load_brunel_spikes(self):

        data_folder = os.path.join("brunel_sim_delta", )
        brunel_name_root = self.correlation_type.replace("brunel_", "")
        self.firing_rate_name = brunel_name_root
        data_files = [f for f in os.listdir(data_folder)
                      if f.endswith("_ex.npy") and
                      f.startswith(brunel_name_root)]

        if len(data_files) > 1:
            raise RuntimeError("Ambiguity in data file names")
        filename = data_files[0]
        spiketrains = np.load(join(data_folder, filename), allow_pickle=True)[()]

        if type(spiketrains) == dict:
            spiketrains = spiketrains["spikes"]
        if len(spiketrains) < self.num_kernels:
            print(len(spiketrains))
            raise RuntimeError("Too few spiketrains available!")


        for s_idx in range(self.num_kernels):
            st = spiketrains[s_idx]
            st = st[np.where(st < self.t_stop)]
            st = st[np.where(st > self.t_start)]
            spike_train = neo.SpikeTrain(st * ms,
                                               t_start=self.t_start*ms,
                                               t_stop=self.t_stop * ms)
            self.firing_rate[s_idx] = BinnedSpikeTrain(spike_train,
                                                       bin_size=self.dt * ms,
                                                       t_start=self.t_start * ms,
                                                       t_stop=self.t_stop * ms).to_array()[0]

    def _make_spiketimes_mip(self):
        if not hasattr(self, "jitter"):
            self.jitter = 0

        mother_spike_train = homogeneous_poisson_process(
            rate=self.rate*Hz, t_start=self.t_start*ms, t_stop=self.t_stop * ms)
        spike_trains = []

        for st_idx in range(self.num_kernels):
            if "rate_std" in self.kwargs:
                rate_ = np.random.normal(self.rate, self.kwargs["rate_std"])
                if rate_ <= 0.0:
                    rate_ = 0.0
                print("Using variable rate: ", rate_)
            else:
                rate_ = self.rate
            independent_rate = rate_ * (1. - self.correlation)
            correlation_rate = rate_ - independent_rate

            if np.abs(independent_rate) < 1e-6:
                child_spike_train = []
                corr_spike_times = mother_spike_train.copy()
            else:
                child_spike_train = list(homogeneous_poisson_process(
                    rate=independent_rate*Hz, t_start=self.t_start*ms,
                    t_stop=self.t_stop * ms))

                num_corr_spikes = np.min([int(correlation_rate * self.t_stop / 1000.),
                                         len(mother_spike_train)])

                corr_spike_times = np.random.choice(mother_spike_train, num_corr_spikes,
                                                    replace=False)

            if self.jitter > 0:
                print("Adding jitter!")
                corr_spike_times += np.random.normal(0, self.jitter, size=num_corr_spikes)
                corr_spike_times = corr_spike_times[np.where(corr_spike_times < self.t_stop)]
                corr_spike_times = corr_spike_times[np.where(corr_spike_times > self.t_start)]

            child_spike_train.extend(corr_spike_times)
            child_spike_train = neo.SpikeTrain(np.sort(child_spike_train) * ms,
                                               t_start=self.t_start*ms,
                                               t_stop=self.t_stop * ms)
            spike_trains.append(child_spike_train)

        for k_idx in range(self.num_kernels):
            self.firing_rate[k_idx] = BinnedSpikeTrain(spike_trains[k_idx],
                                                       bin_size=self.dt*ms,
                                                       t_start=self.t_start * ms,
                                                       t_stop=self.t_stop * ms).to_array()[0]

    def make_ground_truth_signal(self):

        if hasattr(self, 'num_elecs') and self.num_elecs > 1:
            self.phi = np.zeros((self.num_elecs, self.signal_length))
            for elec_idx in range(self.num_elecs):
                for k_idx in range(self.num_kernels):
                    s_ = np.convolve(self.kernels[k_idx, elec_idx], self.firing_rate[k_idx],
                                     mode="same") * (self.dt*1e-3)
                    self.phi[elec_idx] += s_
        else:

            self.phi = np.zeros(self.signal_length)
            for k_idx in range(self.num_kernels):
                s_ = np.convolve(self.kernels[k_idx, ], self.firing_rate[k_idx],
                                 mode="same")  * (self.dt*1e-3)
                self.phi += s_

        if hasattr(self, 'cdm_kernels'):
            self.cdm = np.zeros((3, self.signal_length))
            for axis_idx in range(3):
                for k_idx in range(self.num_kernels):

                    s_ = np.convolve(self.cdm_kernels[k_idx, axis_idx], self.firing_rate[k_idx],
                                     mode="same") * (self.dt*1e-3)
                    self.cdm[axis_idx] += s_

    def make_kernel_signal(self):
        self.avrg_kernel = np.average(self.kernels[:self.num_kernels], axis=0)
        self.std_kernel = np.std(self.kernels, axis=0)
        if hasattr(self, 'cdm_kernels'):
            self.avrg_cdm_kernel = np.average(self.cdm_kernels, axis=0)
            self.std_cdm_kernel = np.std(self.cdm_kernels, axis=0)

        if hasattr(self, 'num_elecs') and self.num_elecs > 1:
            self.phi_tilde = np.zeros((self.num_elecs, self.signal_length))
            for elec_idx in range(self.num_elecs):

                self.phi_tilde[elec_idx] = np.convolve(self.avrg_kernel[elec_idx, :],
                                                       self.avrg_firingrate,
                                         mode="same") * self.num_kernels  * (self.dt*1e-3)
        else:
            self.phi_tilde = np.convolve(self.avrg_kernel, self.avrg_firingrate,
                                         mode="same") * self.num_kernels  * (self.dt*1e-3)

        if hasattr(self, 'cdm_kernels'):
            self.cdm_tilde = np.zeros((3, self.signal_length))
            for axis_idx in range(3):
                self.cdm_tilde[axis_idx, :] = np.convolve(self.avrg_cdm_kernel[axis_idx, :],
                            self.avrg_firingrate,
                            mode="same") * self.num_kernels * (self.dt * 1e-3)

    def make_kernels(self):
        # print("Making kernels ...")
        if self.kernels_param_dict['kernel_type'] == 'toy_Exp2Syn':
            self._make_toy_kernels_exp2syn()
        elif self.kernels_param_dict['kernel_type'] == 'toy_ExpSyn':
            self._make_toy_kernels_expsyn()
        elif self.kernels_param_dict['kernel_type'] == 'neural_sim':
            self._load_simulated_kernels()

    def _load_simulated_kernels(self):
        data_folder = self.kernels_param_dict['data_folder']
        self.case_name = self.kernels_param_dict['case_name']
        data_dict = np.load(join(data_folder, f"kernels_case_{self.case_name}.npy"),
                            allow_pickle=True)[()]
        params = data_dict['parameters']

        #self.kernel_params = params

        k_ = np.array(data_dict['kernel_trials'])
        cdm_ = np.array(data_dict['cdm_trials'])
        self.K_equal = np.array(data_dict['K_equal'])
        self.K_nonequal = np.array(data_dict['K_nonequal'])

        self.kernels = k_
        self.cdm_kernels = cdm_

        num_kernels, self.num_elecs, self.kernel_length = self.kernels.shape
        if num_kernels < self.num_kernels:
            print(f"NOTE: Only {num_kernels} available, not {self.num_kernels}")
            sys.exit()

    def _make_toy_kernels_exp2syn(self):

        self.amp_std = self.kernels_param_dict['amp_std']
        self.amp_mean = self.kernels_param_dict['amp_mean']
        self.tau_1_mean = self.kernels_param_dict['tau_1_mean']
        self.tau_1_std = self.kernels_param_dict['tau_1_std']
        self.tau_2_mean = self.kernels_param_dict['tau_2_mean']
        self.tau_2_std = self.kernels_param_dict['tau_2_std']
        self.kernel_length = self.kernels_param_dict['kernel_length']

        self.syn_amps = np.random.normal(self.amp_mean, self.amp_std,
                                         size=self.num_kernels)
        self.tau1s = np.random.normal(self.tau_1_mean, self.tau_1_std,
                                         size=self.num_kernels)
        self.tau1s[np.where(self.tau1s < 0.01)] = 0.01

        self.tau2s = np.random.normal(self.tau_2_mean, self.tau_2_std,
                                         size=self.num_kernels)
        self.tau2s[np.where(self.tau2s < 0.1)] = 0.1

        self.kernels = np.zeros((self.num_kernels, self.kernel_length))
        tvec = np.arange(self.kernel_length) * self.dt

        for k_idx in range(self.num_kernels):
            t_ = tvec[:int(self.kernel_length/2)]
            tau1_ = self.tau1s[k_idx]
            tau2_ = self.tau2s[k_idx]
            k_ = (-np.exp(-t_ / tau1_) + np.exp(-t_ / tau2_))

            k_ = np.abs(k_) / np.max(np.abs(k_)) * self.syn_amps[k_idx]
            self.kernels[k_idx, int(self.kernel_length/2):] = k_

    def _make_toy_kernels_expsyn(self):

        self.amp_std = self.kernels_param_dict['amp_std']
        self.amp_mean = self.kernels_param_dict['amp_mean']
        self.tau_1_mean = self.kernels_param_dict['tau_1_mean']
        self.tau_1_std = self.kernels_param_dict['tau_1_std']

        self.kernel_length = self.kernels_param_dict['kernel_length']

        self.syn_amps = np.random.normal(self.amp_mean, self.amp_std,
                                         size=self.num_kernels)

        self.tau1s = np.random.normal(self.tau_1_mean, self.tau_1_std,
                                         size=self.num_kernels)
        self.tau1s[np.where(self.tau1s < 0.1)] = 0.1

        self.kernels = np.zeros((self.num_kernels, self.kernel_length))
        tvec = np.arange(self.kernel_length) * self.dt

        for k_idx in range(self.num_kernels):
            t_ = tvec[:int(self.kernel_length/2)]
            tau1_ = self.tau1s[k_idx]
            k_ = (+np.exp(-t_ / tau1_))

            k_ = np.abs(k_) / np.max(np.abs(k_)) * self.syn_amps[k_idx]
            self.kernels[k_idx, int(self.kernel_length/2):] = k_

    def evaluate_difference(self):

        if hasattr(self, 'num_elecs'):
            self.difference = (self.phi - self.phi_tilde)

            self.phi_var = np.var(self.phi, axis=1)
            self.var_diff = np.var(self.difference, axis=1)
            self.rel_var_diff =  self.var_diff / self.phi_var
        else:
            self.difference = (self.phi - self.phi_tilde)
            self.phi_var = np.var(self.phi)
            self.var_diff = np.var(self.difference)
            self.rel_var_diff =  self.var_diff / self.phi_var

        if hasattr(self, 'cdm_kernels'):
            self.difference_cdm = (self.cdm - self.cdm_tilde)
            self.var_cdm = np.var(self.cdm, axis=1)
            self.var_diff_cdm = np.var(self.difference_cdm, axis=1)
            self.rel_var_diff_cdm = self.var_diff_cdm / self.var_cdm

    def make_simple_plot(self, include_info_in_plot=True):
        print("Plotting ...")
        num_kernels = self.num_kernels
        xlim = [self.tvec_signal[-1] - 300, self.tvec_signal[-1]]

        try:
            st_title = "rate: {} Hz;  correlation type: {};  correlation: {:1.3f};  jitter: {:1.1f} ms".format(
                self.rate, self.correlation_type, self.correlation, self.jitter)
        except:
            st_title = "correlation type: {}".format(self.correlation_type)
        sig_title =  r"var($\phi$-$\tilde \phi$):{:1.3f};  E$_t$$_h$$_e$$_o$$_r$$_y$:{:1.3f};  var($\phi$-$\tilde \phi$)/var($\phi$):{:1.3f}".format(
            self.var_diff, self.error_theory, self.rel_var_diff)

        plt.close("all")
        fig = plt.figure(figsize=[18, 10])
        fig.subplots_adjust(hspace=0.6, top=0.95, right=0.98, left=0.07, wspace=0.2)
        ax_k = fig.add_subplot(421, title="kernels (N={:d})".format(self.num_kernels),
                               xlabel="time (ms)", ylabel="µV")
        ax_sp = fig.add_subplot(423, ylabel="spike trains",
                                ylim=[-1, self.num_kernels ], xlim=xlim,
                                xlabel="time (ms)")
        ax_sp.set_title(st_title, fontsize=12)
        ax_fr = fig.add_subplot(425, ylabel="mean\nfiring rate", xlim=xlim,
                                xlabel="time (ms)")
        ax_sig = fig.add_subplot(427, xlim=xlim, xlabel="time (ms)",
                                 ylabel="µV")
        ax_sig.set_title(sig_title, fontsize=12)

        for k_idx in range(num_kernels):
            ax_k.plot(self.tvec_kernel, self.kernels[k_idx, ], lw=0.5)
            spike_time_idxs = np.where(self.firing_rate[k_idx] > 0.1)[0]
            ax_sp.plot(self.tvec_signal[spike_time_idxs],
                       np.ones(len(spike_time_idxs)) * k_idx, '|', ms=5)

        l, = ax_k.plot(self.tvec_kernel, self.avrg_kernel, lw=2, c='k')
        ax_k.legend([l], ["mean"], frameon=False, loc=(.7, 0.05))

        ax_fr.plot(self.tvec_signal, self.avrg_firingrate, lw=2, c='k')

        l1, = ax_sig.plot(self.tvec_signal, self.phi_tilde, lw=2, c='k', ls='-')
        l2, = ax_sig.plot(self.tvec_signal, self.phi, c='gray')
        l3, = ax_sig.plot(self.tvec_signal, self.difference, c='r')


        fig.legend([l1, l2, l3,# l4
                    ], [r"population kernel ($\tilde \phi$)",
                                  r"reference ($\phi$)",
                                  r"$\phi$ - $\tilde \phi$",
                        #          r"E$_{\rm FFT}?$"
                        ],
                      frameon=False, ncol=4, loc=(0.02, 0.01))
        simplify_axes(fig.axes)
        mark_subplots(fig.axes, xpos=-0.05)
        if include_info_in_plot:
            fig.text(0.35, 0.93, "amplitudes: {:1.2f} (±{:1.2f})".format(self.amp_mean,
                                                           self.amp_std), fontsize=12)
            fig.text(0.35, 0.9, "tau_1: {:1.2f} (±{:1.2f})".format(self.tau_1_mean,
                                                           self.tau_1_std), fontsize=12)

        plt.savefig(join(self.figure_folder, "kernel_illustration_%s_cov.png" % self.return_sim_name()), dpi=50)
        plt.savefig("kernel_illustration_recent.png", dpi=50)

    def make_multiple_elecs_plot(self):

        num_kernels = self.num_kernels
        # psd_max = np.max(self.phi_psd[0, 1:])
        # xlim = [self.tvec_kernel[-1], self.tvec_signal[-1]]
        # xlim = [self.tvec_signal[-1] - 900, self.tvec_signal[-1] - 100]
        # xlim = [self.t_stop - 1000, self.t_stop -800]
        xlim = [350, 400]


        plt.close("all")
        fig = plt.figure(figsize=[18, 10])
        fig.subplots_adjust(hspace=0.6, top=0.95, right=0.98, left=0.25, wspace=0.2)
        ax_k = fig.add_axes([0.04, 0.1, 0.15, 0.8], title="kernels (N={:d})".format(self.num_kernels),
                               xlabel="time (ms)", ylabel="µV")
        ax_sp = fig.add_subplot(321, ylabel="spike trains",
                                ylim=[-1, self.num_kernels ], xlim=xlim,
                                xlabel="time (ms)")
        # ax_sp.set_title(st_title, fontsize=12)
        ax_fr = fig.add_subplot(323, ylabel="mean\nfiring rate", xlim=xlim,
                                xlabel="time (ms)")
        ax_sig = fig.add_subplot(325, xlim=xlim, xlabel="time (ms)",
                                 ylabel="µV")

        ax_err_depth = fig.add_axes([0.6, 0.1, 0.15, 0.8], title="error")
        ax_rel_err_depth = fig.add_axes([0.8, 0.1, 0.15, 0.8], title="relative error")

        le_, = ax_err_depth.plot(self.var_diff, elec_z, c='orange')
        lt_, = ax_err_depth.plot(self.error_theory, elec_z, c='blue')

        ax_err_depth.legend([le_, lt_], ["error simulated", "error from theory"],
                            frameon=False)

        le_, = ax_rel_err_depth.plot(self.var_diff / self.phi_var, elec_z, c='orange')
        lt_, = ax_rel_err_depth.plot(self.error_theory / self.phi_var, elec_z, c='blue')

        kernel_norm = np.max(np.abs(self.kernels)) * 1.5
        for k_idx in range(num_kernels):
            for elec_idx in range(self.num_elecs):
                y_ = self.kernels[k_idx, elec_idx, ] / kernel_norm * dz + elec_z[elec_idx]
                ax_k.plot(self.tvec_kernel, y_, lw=0.5, c='gray')
                spike_time_idxs = np.where(self.firing_rate[k_idx] > 0.1)[0]
                ax_sp.plot(self.tvec_signal[spike_time_idxs],
                           np.ones(len(spike_time_idxs)) * k_idx, '|', ms=5)

        for elec_idx in range(self.num_elecs):
            l, = ax_k.plot(self.tvec_kernel, self.avrg_kernel[elec_idx] / kernel_norm * dz + elec_z[elec_idx], lw=2, c='k')
        ax_k.legend([l], ["mean"], frameon=False, loc=(.7, 0.05))

        ax_fr.plot(self.tvec_signal, self.avrg_firingrate, lw=2, c='k')

        lfp_norm = np.max(np.abs(self.phi)) * 1.
        for elec_idx in range(self.num_elecs):
            l1, = ax_sig.plot(self.tvec_signal, self.phi_tilde[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                              lw=2, c='k', ls='-')
            l2, = ax_sig.plot(self.tvec_signal, self.phi[elec_idx] / lfp_norm * dz + elec_z[elec_idx], c='gray')
            l3, = ax_sig.plot(self.tvec_signal, self.difference[elec_idx] / lfp_norm * dz + elec_z[elec_idx], c='r')

        fig.legend([l1, l2, l3,], [r"population kernel ($\tilde \phi$)",
                                  r"reference ($\phi$)",
                                  r"$\phi$ - $\tilde \phi$",
                                  r"E$_{\rm FFT}?$"],
                      frameon=False, ncol=4, loc=(0.02, 0.01))
        simplify_axes(fig.axes)
        mark_subplots(fig.axes, xpos=-0.05)

        plt.savefig(join(self.figure_folder, "kernel_illustration_%s.png" % self.sim_name), dpi=90)
        plt.savefig("kernel_illustration_recent_remade.png", dpi=90)
        # plt.show()

    def save_signals(self):
        np.save(join(self.data_folder, f'phi_{self.sim_name}.npy'), self.phi)
        np.save(join(self.data_folder, f'phi_tilde_{self.sim_name}.npy'), self.phi_tilde)
        np.save(join(self.data_folder, f'cdm_{self.sim_name}.npy'), self.cdm)
        np.save(join(self.data_folder, f'cdm_tilde_{self.sim_name}.npy'), self.cdm_tilde)

    def save_errors(self):
        np.save(join(self.data_folder, f'error_observed_{self.sim_name}.npy'), self.var_diff)
        np.save(join(self.data_folder, f'error_theory_{self.sim_name}.npy'), self.error_theory)
        np.save(join(self.data_folder, f'cdm_error_observed_{self.sim_name}.npy'), self.var_diff_cdm)


def run_illustrative_examples(figure_folder):

    num_kernels = 1000
    amp_stds = np.array([0, 0.5, 0.5, 0.5])
    correlations = np.array([0.0, 0.1, 0.0, 1.0])

    dt = 0.1
    kernel_length = 200
    signal_length = 100000
    kernels_param_dict = dict(
        amp_mean = 1.0,
        kernel_length = kernel_length,
        tau_1_mean = 0.2,
        tau_1_std = 0.00,
        tau_2_mean = 1.0,
        tau_2_std = 0.0,
        kernel_type = 'toy_Exp2Syn'
    )

    default_param_dict = dict(
        num_kernels = num_kernels,
        figure_folder = figure_folder,
        signal_length = signal_length,
        kernels_param_dict = kernels_param_dict,
        data_folder = figure_folder,
        firing_rate_folder=figure_folder,
        dt = dt,
        rate = 25,
        jitter = 0,
        correlation_type = "mip"
    )

    column_titles =[
        "homogeneous kernels\nuncorrelated spikes",
        "heterogeneous kernels\nlow spike correlation",
        "heterogeneous kernels\nuncorrelated spikes",
        "heterogeneous kernels\nfully correlated spikes"
        ]

    xlim = [1000, 2000]
    plt.close("all")
    fig = plt.figure(figsize=[6, 4.3])
    fig.subplots_adjust(hspace=0.9, top=0.9, wspace=0.5, left=0.1,
                        right=0.97, bottom=0.13)

    cols = len(amp_stds)
    rows = 3
    for i in range(cols):
        np.random.seed(12345)
        param_dict = default_param_dict.copy()

        param_dict["correlation"] = correlations[i]
        param_dict['kernels_param_dict']["amp_std"] = amp_stds[i]

        toy_kernel = KernelValidity(**param_dict)
        t_kernels = toy_kernel.tvec_kernel
        t_mid_idx = int(len(t_kernels) / 2)
        t_kernels -= t_kernels[t_mid_idx]
        t_signal = toy_kernel.tvec_signal
        avrg_kernel = toy_kernel.avrg_kernel
        sd_kernel = toy_kernel.std_kernel
        fr = toy_kernel.avrg_firingrate
        frs = toy_kernel.firing_rate
        fr_std =  toy_kernel.std_firingrate
        rel_diff = np.std(toy_kernel.difference) / np.std(toy_kernel.phi)

        ax_k = fig.add_subplot(rows, cols, 0 * cols + 1 + i,
                               xlabel="time (ms)",
                               #ylabel="µV",
                               ylim=[-0.1, 2])

        col_title = column_titles[i] + '\n\n' + r"$A_{\rm SD}$" + "={:1.1f} µV".format(amp_stds[i])

        ax_k.set_title(col_title, y=0.8)

        ax_fr = fig.add_subplot(rows, cols, 1 * cols + 1 + i,
                                xlim=xlim,
                                xlabel="time (ms)"
                                )
        ax_fr.set_title("$f$={:1.1f}".format(correlations[i]), y=0.9)

        ax_sig = fig.add_subplot(rows, cols, 2 * cols + 1 + i, xlim=xlim,
                                 xlabel="time (ms)",
                                 )
        ax_sig.set_title(r"$E_{\rm rel}$" + "={:1.1f}".format(rel_diff),
                         pad=0.8)

        if i == 0:
            ax_fr.set_ylabel("population rate\n(spikes/$\\Delta{}t$)")
            ax_k.set_ylabel(r"$V$ (µV)")
            ax_sig.set_ylabel(r"$V$ (µV)")


        ax_k.fill_between(t_kernels, avrg_kernel - sd_kernel, avrg_kernel + sd_kernel, fc='0.7')
        ax_k.plot(t_kernels, avrg_kernel, c='k', lw=2)

        ax_fr.plot(t_signal, fr_std  * dt*1e-3, c='0.7', lw=0.5)
        ax_fr.plot(t_signal, fr * dt*1e-3, c='k', lw=2)

        l2, = ax_sig.plot(t_signal[:], toy_kernel.phi_tilde, lw=2, c='k', ls='-')
        l1, = ax_sig.plot(t_signal[:], toy_kernel.phi, c='gray')
        l3, = ax_sig.plot(t_signal[:], toy_kernel.difference, c='r')

    fig.legend([l1, l2, l3], ["ground truth", "population kernel", "difference"],
               ncol=3, frameon=False, loc="lower center")
    simplify_axes(fig.axes)

    plt.savefig(join(figure_folder, "illustrative_toy_examples.png"))
    plt.savefig(join(figure_folder, "illustrative_toy_examples.pdf"))


def run_parameter_scan(results_folder):

    amp_stds = np.linspace(0, 2, 11)
    correlations = np.r_[0, np.logspace(-2, 0, 11)]
    # firing_rates = np.r_[0.1, 1, 10, 100]

    num_kernels = 1000
    signal_length = 100000

    figure_folder = results_folder

    kernels_param_dict = dict(amp_mean = 1.0,
                              #amp_std = 1,
                              kernel_length = 200,
                              tau_1_mean = 0.2,
                              tau_1_std = 0.00,
                              tau_2_mean = 1.0,
                              tau_2_std = 0.0,
                              kernel_type = 'toy_Exp2Syn'
                              )

    default_param_dict = dict(num_kernels = num_kernels,
                      figure_folder = figure_folder,
                      signal_length = signal_length,
                      dt = 0.1,
                      rate = 25,
                      # rate_std = 15,
                      correlation = 0.0,
                      jitter = 0,
                      correlation_type = "mip",
                      kernels_param_dict = kernels_param_dict,
                      data_folder = results_folder,
                      firing_rate_folder = results_folder,
                      calc_error_from_theory=True)
    #
    var_dict = {"amp_stds": amp_stds,
                "correlations": correlations}
    diff_var_dict = {"amp_stds": amp_stds,
                "correlations": correlations}
    diff_theory_dict = {"amp_stds": amp_stds,
                "correlations": correlations}
    rel_diff_var_dict = {"amp_stds": amp_stds,
                "correlations": correlations}

    # var_dict = {"amp_stds": amp_stds,
    #             "firing_rates": firing_rates}
    # diff_var_dict = {"amp_stds": amp_stds,
    #             "firing_rates": firing_rates}
    # diff_theory_dict = {"amp_stds": amp_stds,
    #             "firing_rates": firing_rates}
    # rel_diff_var_dict = {"amp_stds": amp_stds,
    #             "firing_rates": firing_rates}

    for i, amp_std in enumerate(amp_stds):
        for j, correlation in enumerate(correlations):
            # print(amp_std, correlation)
            param_dict = default_param_dict.copy()
            param_dict['kernels_param_dict']["amp_std"] = amp_std
            param_dict["correlation"] = correlation
            # param_dict["rate"] = firing_rate
            np.random.seed(1234)
            toy_kernel = KernelValidity(**param_dict)
            toy_kernel.make_simple_plot()

            var_dict[amp_std, correlation] = np.var(toy_kernel.phi)
            diff_var_dict[amp_std, correlation] = toy_kernel.var_diff
            diff_theory_dict[amp_std, correlation] = toy_kernel.error_theory
            rel_diff_var_dict[amp_std, correlation] = toy_kernel.rel_var_diff
            print(amp_std, correlation, toy_kernel.error_theory, toy_kernel.rel_var_diff)

    os.makedirs(results_folder, exist_ok=True)
    np.save(join(results_folder, "var_dict.npy"), var_dict)
    np.save(join(results_folder, "diff_var_dict.npy"), diff_var_dict)
    np.save(join(results_folder, "diff_theory_dict.npy"), diff_theory_dict)
    np.save(join(results_folder, "rel_diff_var_dict.npy"), rel_diff_var_dict)


def plot_parameter_scan_results(results_folder):
    var_dict = np.load(join(results_folder, "var_dict.npy"), allow_pickle=True).item()
    diff_var_dict = np.load(join(results_folder, "diff_var_dict.npy"), allow_pickle=True).item()
    diff_theory_dict = np.load(join(results_folder, "diff_theory_dict.npy"), allow_pickle=True).item()
    rel_diff_var_dict = np.load(join(results_folder, "rel_diff_var_dict.npy"), allow_pickle=True).item()

    amp_stds = var_dict["amp_stds"]
    correlations = var_dict["correlations"]
    # firing_rates = var_dict["firing_rates"]

    var_matrix = np.zeros((len(amp_stds), len(correlations)))
    diff_var_matrix = np.zeros((len(amp_stds), len(correlations)))
    diff_theory_matrix = np.zeros((len(amp_stds), len(correlations)))
    rel_diff_var_matrix = np.zeros((len(amp_stds), len(correlations)))
    rel_diff_theory_matrix = np.zeros((len(amp_stds), len(correlations)))

    # var_matrix = np.zeros((len(amp_stds), len(firing_rates)))
    # diff_var_matrix = np.zeros((len(amp_stds), len(firing_rates)))
    # diff_theory_matrix = np.zeros((len(amp_stds), len(firing_rates)))
    # rel_diff_var_matrix = np.zeros((len(amp_stds), len(firing_rates)))
    # rel_diff_theory_matrix = np.zeros((len(amp_stds), len(firing_rates)))

    for i, amp_std in enumerate(amp_stds):
        for j, correlation in enumerate(correlations):
        # for j, firing_rate in enumerate(firing_rates):

            var_matrix[i, j] = np.sqrt(var_dict[amp_std, correlation])
            diff_var_matrix[i, j] = np.sqrt(diff_var_dict[amp_std, correlation])
            diff_theory_matrix[i, j] = np.sqrt(np.abs(diff_theory_dict[amp_std, correlation]))
            rel_diff_var_matrix[i, j] = np.sqrt(rel_diff_var_dict[amp_std, correlation])
            rel_diff_theory_matrix[i, j] = np.sqrt(np.abs(diff_theory_dict[amp_std, correlation] / var_dict[amp_std, correlation]))

    var_max = np.max(var_matrix)
    diff_var_max = 10#np.max(diff_var_matrix)

    print(diff_var_max)

    fig = plt.figure(figsize=[6, 4.2])
    fig.subplots_adjust(hspace=0.9, top=0.92, left=0.075, wspace=0.7, right=.92, bottom=0.07)

    xlabel = "correlation $f$"
    # xlabel = "firing rate"

    ax_var = fig.add_subplot(3, 3, 1, xlabel=xlabel, title="simulated\nSD",
                             ylabel="amplitude SD (µV)")

    ax_err = fig.add_subplot(3, 3, 2, title="simulated\nabsolute error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )
    ax_rel = fig.add_subplot(3, 3, 3, title="simulated\nrelative error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )
    ax_err_theory = fig.add_subplot(3, 3, 5, title="theory\nabsolute error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )
    ax_rel_theory = fig.add_subplot(3, 3, 6, title="theory\nrelative error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )

    ax_err_diff = fig.add_subplot(3, 3, 8, title="difference\nabsolute error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )
    ax_rel_diff = fig.add_subplot(3, 3, 9, title="difference\nrelative error",
                                 xlabel=xlabel,
                                 ylabel="amplitude SD (µV)"
                                 )


    img_var = ax_var.imshow(var_matrix, #norm=LogNorm(vmin=1e-1 * var_max, vmax=var_max),
                            origin="lower")
    img_diff = ax_err.imshow(diff_var_matrix, vmin=0, vmax=diff_var_max, origin="lower")
    img_diff_theory = ax_err_theory.imshow(diff_theory_matrix, vmin=0, vmax=diff_var_max, origin="lower")
    img_rel = ax_rel.imshow(rel_diff_var_matrix, vmin=0, vmax=1, origin="lower")
    img_rel_theory = ax_rel_theory.imshow(rel_diff_theory_matrix,
                                          vmin=0, vmax=1, origin="lower")

    img_err_diff = ax_err_diff.imshow(np.abs(diff_var_matrix - diff_theory_matrix), vmin=0,
                                      vmax=0.25, origin="lower")
    img_rel_diff = ax_rel_diff.imshow(np.abs(rel_diff_var_matrix - rel_diff_theory_matrix),
                                      vmin=0, vmax=0.025, origin="lower")

    cor_ticks = np.array([1, 6, 11])
    # cor_ticks = np.array([0, 1, 2, 3])

    amp_ticks = np.array([0, 5, 10])
    for ax in [ax_var, ax_err, ax_rel, ax_err_theory, ax_rel_theory, ax_err_diff, ax_rel_diff]:
        ax.set_xticks(cor_ticks)
        ax.set_xticklabels(["{:1.2f}".format(c) for c in correlations[cor_ticks]], rotation=0)
        # ax.set_xticklabels(["{:1.1f}".format(c) for c in firing_rates[cor_ticks]], rotation=90)

        ax.set_yticks(amp_ticks)
        ax.set_yticklabels(["{:1.1f}".format(a) for a in amp_stds[amp_ticks]])

    cax_1 = fig.add_axes([0.245, 0.74, 0.01, 0.18])
    cax_2 = fig.add_axes([0.57, 0.74, 0.01, 0.18])
    cax_3 = fig.add_axes([0.9, 0.74, 0.01, 0.18])

    cax_2b = fig.add_axes([0.57, 0.405, 0.01, 0.18])
    cax_3b = fig.add_axes([0.9, 0.405, 0.01, 0.18])

    cax_2c = fig.add_axes([0.57, 0.07, 0.01, 0.18])
    cax_3c = fig.add_axes([0.9, 0.07, 0.01, 0.18])

    plt.colorbar(img_var, cax=cax_1, label="µV")#, label=r"var($\phi$)")
    plt.colorbar(img_diff, cax=cax_2, label="µV")#, label=r"var($\phi$ - $\tilde \phi$)")
    plt.colorbar(img_rel, cax=cax_3)#, label=r"var($\phi$ - $\tilde \phi$) / var($\phi$)")

    plt.colorbar(img_diff_theory, cax=cax_2b, label="µV")#, label=r"var($\phi$ - $\tilde \phi$)")
    plt.colorbar(img_rel_theory, cax=cax_3b)#, label=r"var($\phi$ - $\tilde \phi$) / var($\phi$)")

    plt.colorbar(img_err_diff, cax=cax_2c, label="µV")#, label=r"var($\phi$ - $\tilde \phi$)")
    plt.colorbar(img_rel_diff, cax=cax_3c)#, label=r"var($\phi$ - $\tilde \phi$) / var($\phi$)")

    mark_subplots(fig.axes[:7])

    plt.savefig(join(results_folder, "toy_kernels_error.png"))
    plt.savefig(join(results_folder, "toy_kernels_error.pdf"))


def run_simpop_example(figure_folder, data_folder, kernel_name,
                       correlation_type, firing_rate_folder):

    kernels_param_dict = dict(
        kernel_type="neural_sim",
        data_folder=data_folder,
        case_name=kernel_name,
    )

    if "brunel" in correlation_type:
       jitter = None
       correlation = None
       rate = None
    else:
        fr_params = correlation_type.split('_')
        correlation = float(fr_params[1])
        rate = int(fr_params[2])
        jitter = float(fr_params[3])
        correlation_type = fr_params[0]

    param_dict = dict(num_kernels=100,
                      figure_folder=figure_folder,
                      kernels_param_dict=kernels_param_dict,
                      signal_length=250000,
                      dt=neural_simulations.dt,
                      rate=rate,
                      correlation=correlation,
                      jitter=jitter,
                      correlation_type=correlation_type,
                      firing_rate_folder=firing_rate_folder,
                      calc_error_from_theory=True,
                      data_folder = data_folder,
                      )

    kernel_instance = KernelValidity(**param_dict)
    kernel_instance.save_signals()
    kernel_instance.save_errors()
    kernel_instance.make_multiple_elecs_plot()


def compare_errors(correlation_type):

    case_studies = {
        'input region': ['apical', 'default', 'uniform'],
        'population size': ['small_population', 'default', 'large_population'],
        'synaptic variability': ['similar_synapses', 'default', 'variable_synapses'],
        'input spread': ['narrow_input_region', 'default', 'broad_input_region'],
        'population radius': ['small_radius', 'default', 'large_radius'],
    }

    var_max = 4000 ** 0.5
    error_max = 8 ** 0.5
    rel_error_max = 0.3

    cond_clr = neural_simulations.cond_clr
    plt.close("all")
    fig = plt.figure(figsize=[6, 8])

    fig.subplots_adjust(bottom=0.06, top=0.97, right=0.97, wspace=0.2,
                        left=0.1, hspace=0.6)

    full_corr_name = correlation_type
    if "brunel" in correlation_type:
        jitter = None
        correlation = None
        rate = None
    else:
        fr_params = correlation_type.split('_')
        correlation = float(fr_params[1])
        rate = int(fr_params[2])
        jitter = float(fr_params[3])
        correlation_type = fr_params[0]
    # rate = 25
    # correlation = 0.0
    # jitter = 0.0
    num_kernels = 100

    axes_var = []
    axes_var_diff = []
    max_vars = 0
    max_var_diffs = 0

    for c_idx, case_study in enumerate(case_studies):
        ax_var = fig.add_subplot(len(case_studies), 3, c_idx * 3 + 1,
                                 xlabel="LFP amplitude (µV)",
                                 ylabel='depth (µm)',
                                 xlim=[-var_max/50, var_max])
        axes_var.append(ax_var)
        ax_err = fig.add_subplot(len(case_studies), 3, c_idx * 3 + 2,
                                 xlabel="error (µV)",
                                 #ylabel='depth (µm)',
                                 yticklabels=[],
                                 xlim=[-error_max/50, error_max],
                                 )
        axes_var_diff.append(ax_err)
        #ax_err.set_title(case_study, fontsize=14, pad=15)
        ax_rel_err = fig.add_subplot(len(case_studies), 3, c_idx * 3 + 3,
                                 xlabel="relative error",
                                 #ylabel='depth (µm)',
                                 yticklabels=[],
                                 xlim=[-0.0005, rel_error_max])

        lines = []
        line_names = []
        for kernel_name in case_studies[case_study]:
             # for kernel_name in kernel_names:
            k_part = f'{kernel_name}'

            if 'brunel' in correlation_type:
                fr_part = f'corrtype:{correlation_type}'
            else:
                fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(rate,
                                                                         correlation_type,
                                                                         correlation,
                                                                         jitter)

            sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

            phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
            phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))

            var_diff = np.sqrt(np.load(join(data_folder, f'error_observed_{sim_name}.npy')))
            #var_diff = np.std(phi - phi_tilde, axis=1)
            max_vars = np.max([max_vars, np.max(np.std(phi, axis=1))])
            max_var_diffs = np.max([max_var_diffs, np.max(var_diff)])

            error_theory = np.sqrt(np.load(join(data_folder,
                                                f'error_theory_{sim_name}.npy')))
            #lw = 1.5 if kernel_name == "default" else 1.5

            l, = ax_var.plot(np.std(phi, axis=1), elec_z,
                             c=cond_clr[kernel_name], lw=1., ls='-', clip_on=False)
            ax_var.plot(np.std(phi_tilde, axis=1), elec_z,
                        c=cond_clr[kernel_name], lw=1.5, ls='--', clip_on=False)
            l1, = ax_var.plot([], [], c='gray', lw=1., ls='-')
            l2, = ax_var.plot([], [], c='gray', lw=1., ls='--')
            ax_var.legend([l1, l2], ['ground truth', "population kernel"], frameon=False,
                          loc=(0.4, 0.4))

            ax_err.plot(error_theory, elec_z, c=cond_clr[kernel_name], lw=1.5, ls=':')
            ax_err.plot(var_diff, elec_z, c=cond_clr[kernel_name], lw=1, ls='-')
            l1, = ax_err.plot([], [], c='gray', lw=1., ls='-')
            l2, = ax_err.plot([], [], c='gray', lw=1.5, ls=':')
            ax_err.legend([l1, l2], ['simulated', "theory"], frameon=False,
                           loc=(0.5, 0.4))

            rel_norm = np.max(np.std(phi, axis=1))
            if np.all(error_theory / rel_norm > rel_error_max):
                mean_rel_error = np.mean(error_theory / rel_norm)
                ax_rel_err.text(rel_error_max, -500, r"{}$\rightarrow$".format(
                    kernel_name) + "\n(mean: {:1.2f})".format(mean_rel_error),
                                c=cond_clr[kernel_name], ha='right')

            ax_rel_err.plot(error_theory / rel_norm, elec_z,
                             c=cond_clr[kernel_name], lw=1.5, ls=':')
            ax_rel_err.plot(var_diff / rel_norm, elec_z,
                            c=cond_clr[kernel_name], lw=1., ls='-')

            #l1, = ax_rel_err.plot([], [], c='gray', lw=1.5, ls='-')
            #l2, = ax_rel_err.plot([], [], c='gray', lw=1.5, ls=':')
            #ax_rel_err.legend([l1, l2], ['simulated', "theory"], frameon=False,
            #               loc=(0.4, 0.5))

            line_name = kernel_name.replace('_', ' ')
            line_name = line_name.replace('population', r'$K_{\rm out}$')
            print(line_name)
            lines.append(l)
            line_names.append(line_name)
            ax_rel_err.legend(lines, line_names, frameon=False, ncol=3, loc=(-1.5, -0.47))
        ax_marks = ["ABCDE"[c_idx] + str(i) for i in range(1, 4)]
        mark_subplots([ax_var, ax_err, ax_rel_err], ax_marks, ypos=1.03)

    for ax in axes_var:
        ax.set_xlim(-max_vars / 50, max_vars)

    for ax in axes_var_diff:
        ax.set_xlim(-max_var_diffs / 50, max_var_diffs)

    simplify_axes(fig.axes)

    fig.savefig(f"compare_cases_error_{full_corr_name}.png")
    fig.savefig(f"compare_cases_error_{full_corr_name}.pdf")


def error_with_correlation_type(data_folder):
    correlation_types_mip = [
        "mip_0.0_10_0.0",
        "mip_0.1_10_0.0",
        "mip_0.0_50_0.0",
        "mip_0.1_50_0.0",
    ]
    correlation_types_brunel = [
        "brunel_AI_slow",
        "brunel_SI_slow",
    ]

    correlation_types = correlation_types_mip + correlation_types_brunel
    lws = [2, .75, .75, 2, 1, 1, 1, 1]

    num_corrs = len(correlation_types)

    cond_clr = {sim_name: plt.cm.tab10(idx / (num_corrs))
                for idx, sim_name in enumerate(correlation_types)}

    var_max = 500 ** 0.5
    error_max = 15 ** 0.5
    rel_error_max = 0.25


    kernel_names = ['default',
                 # 'small_population', 'large_population',
                 # 'small_radius', 'large_radius',
                 # 'uniform', 'apical',
                 # 'similar_synapses', 'variable_synapses',
                 # 'broad_input_region', 'narrow_input_region'
                 ]

    for kernel_name in kernel_names:
        k_part = f'{kernel_name}'
        plt.close("all")
        fig = plt.figure(figsize=[6, 2.6])

        fig.subplots_adjust(bottom=0.28, top=0.95, right=0.9, wspace=0.2,
                            left=0.1, hspace=0.6)

        num_kernels = 100

        axes_var = []
        axes_var_diff = []
        max_vars = 0
        max_var_diffs = 0

        ax_var = fig.add_subplot(1, 4, 1,
                                 xlabel="LFP amplitude (µV)",
                                 ylabel='depth (µm)',
                                 xlim=[-var_max/50, var_max*1.1])
        axes_var.append(ax_var)
        ax_err = fig.add_subplot(1, 4, 2,
                                 xlabel="error (µV)",
                                 #ylabel='depth (µm)',
                                 yticklabels=[],
                                 xlim=[-error_max/50, error_max],
                                 )
        axes_var_diff.append(ax_err)
        #ax_err.set_title(case_study, fontsize=14, pad=15)
        ax_rel_err = fig.add_subplot(1, 4, 3,
                                 xlabel="relative error",
                                 #ylabel='depth (µm)',
                                 yticklabels=[],
                                 xlim=[-rel_error_max / 50, rel_error_max*1.01])

        ax_abs_scan = fig.add_axes([0.78, 0.63, 0.15, 0.29],
                                 xlabel=r"$\nu$ (s$^{-1}$)",
                                 ylabel='$f$',
                                 title="error"
                                 )
        ax_rel_scan = fig.add_subplot([0.78, 0.13, 0.15, 0.29], xlabel=r"$\nu$ (s$^{-1}$)", ylabel='$f$',
                                      title="relative error")
        lines = []
        line_names = []
        idx = 0

        for correlation_type in correlation_types:
            # if correlation_type == "spikepool":
            #     for corr in [0.]:
            #         for rate in [25, ]:
            corr_full_name = correlation_type
            if "brunel" in correlation_type:
                jitter = None
                correlation = None
                rate = None
            else:

                fr_params = correlation_type.split('_')
                correlation = float(fr_params[1])
                rate = int(fr_params[2])
                jitter = float(fr_params[3])
                correlation_type = fr_params[0]


            if 'brunel' in correlation_type:
                fr_part = f'corrtype:{correlation_type}'
            else:
                fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(rate,
                                                                         correlation_type,
                                                                         correlation,
                                                                         jitter)

            sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

            phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
            phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))


            var_diff = np.sqrt(np.load(join(data_folder, f'error_observed_{sim_name}.npy')))
            #var_diff = np.std(phi - phi_tilde, axis=1)
            max_vars = np.max([max_vars, np.max(np.std(phi, axis=1))])
            max_var_diffs = np.max([max_var_diffs, np.max(var_diff)])

            error_theory = np.sqrt(np.load(join(data_folder, f'error_theory_{sim_name}.npy')))
            lw = lws[idx]#np.random.uniform(0.5, 1.5) #if correlation_type == "default" else 1.5
            clr = cond_clr[corr_full_name]

            l, = ax_var.plot(np.std(phi, axis=1), elec_z,
                             c=clr, lw=lw, ls='-', clip_on=False, zorder=-lw)
            ax_var.plot(np.std(phi_tilde, axis=1), elec_z,
                        c=clr, lw=lw, ls='--', clip_on=False, zorder=-lw)
            l1, = ax_var.plot([], [], c='gray', lw=lw, ls='-')
            l2, = ax_var.plot([], [], c='gray', lw=lw, ls='--')
            ax_var.legend([l1, l2], ['ground truth', "kernel method"], frameon=False,
                          loc=(0.18, 0.85), handlelength=1.7)

            ax_err.plot(var_diff, elec_z, c=clr,
                        lw=lw, ls='-', zorder=-lw)
            ax_err.plot(error_theory, elec_z, c=clr,
                        lw=lw, ls=':', zorder=-lw)
            l1, = ax_err.plot([], [], c='gray', lw=lw, ls='-')
            l2, = ax_err.plot([], [], c='gray', lw=lw, ls=':')
            ax_err.legend([l1, l2], ['simulated', "theory"], frameon=False,
                           loc=(0.3, 0.85))

            rel_norm = np.max(np.std(phi, axis=1))

            ax_rel_err.plot(var_diff / rel_norm, elec_z,
                            c=clr, lw=lw, ls='-', zorder=-lw)
            ax_rel_err.plot(error_theory / rel_norm, elec_z,
                            c=clr, lw=lw, ls=':', zorder=-lw)

            line_name = corr_full_name.replace('_', ' ')
            if line_name == 'spikepool':
                line_name = "Poisson, uncorrelated"
            line_name = line_name.replace("brunel", "Brunel,").replace("AI slow", "AI")
            line_name = line_name.replace("mip", "MIP")
            if "MIP" in line_name:
                params = line_name.split(' ')
                line_name = params[0] + f', $f$={float(params[1]):0.1f}, ' + f'$ν$={params[2]} s⁻¹'

            lines.append(l)
            line_names.append(line_name)
            idx += 1

        ax_rel_err.legend(lines, line_names, frameon=False, ncol=3, loc=(-2.6, -0.38))

        fr_values =  [5, 10, 50, 100]
        corr_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
        rel_error_scan = np.zeros((len(fr_values), len(corr_values)))
        abs_error_scan = np.zeros((len(fr_values), len(corr_values)))
        # Add results from MIP parameter scan
        for idx_fr, fr in enumerate(fr_values):
            for idx_c, corr_value in enumerate(corr_values):

                fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(fr,
                                                                         "mip",
                                                                         corr_value,
                                                                         0.0)

                sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

                phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
                phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))
                var_ = np.max(np.std(phi - phi_tilde, axis=1) / np.std(phi, axis=1))
                var_diff = np.sqrt(np.load(join(data_folder, f'error_observed_{sim_name}.npy')))
                rel_norm = np.max(np.std(phi, axis=1))
                rel_error = np.max(var_diff / rel_norm)

                rel_error_scan[idx_fr, idx_c] = rel_error
                abs_error_scan[idx_fr, idx_c] = np.max(var_diff)
                # print(sim_name, rel_error, var_)

                # if corr_value == 1.0:
                #     plt.close("all")
                #     elec_idx = 12
                #     plt.plot(phi[elec_idx], 'k')
                #     plt.plot(phi_tilde[elec_idx], 'r')
                #     plt.plot((phi[elec_idx] - phi_tilde[elec_idx]), c='gray')
                #     plt.show()

        img1 = ax_abs_scan.imshow(abs_error_scan.T, origin="lower", vmin=0, vmax=4)
        img2 = ax_rel_scan.imshow(rel_error_scan.T, origin="lower", vmin=0, vmax=0.2)

        ax_rel_scan.set_yticks(np.arange(len(corr_values)))
        ax_rel_scan.set_yticklabels(corr_values)
        ax_rel_scan.set_xticks(np.arange(len(fr_values)))
        ax_rel_scan.set_xticklabels(fr_values)

        ax_abs_scan.set_yticks(np.arange(len(corr_values)))
        ax_abs_scan.set_yticklabels(corr_values)
        ax_abs_scan.set_xticks(np.arange(len(fr_values)))
        ax_abs_scan.set_xticklabels(fr_values)
        ax_abs_scan.axis("auto")
        ax_rel_scan.axis("auto")

        cax1 = fig.add_axes([0.935, 0.635, 0.01, 0.28])
        cax2 = fig.add_axes([0.935, 0.135, 0.01, 0.28])
        cbar1 = plt.colorbar(img1, cax=cax1, label="µV")
        cbar2 = plt.colorbar(img2, cax=cax2)

        mark_subplots([ax_var, ax_err, ax_rel_err,  ax_abs_scan, ax_rel_scan], ypos=1.03)
        simplify_axes(fig.axes[:-2])

        fig.savefig(f"correlation_type_error_{kernel_name}.png")
        fig.savefig(f"correlation_type_error_{kernel_name}.pdf")
        # plt.show()


def error_summary_figure(data_folder):
    if rank != 0:
        return
    correlation_types = [
        "mip_0.0_50_0.0",
        "mip_0.0_10_0.0",
        "brunel_AI_slow",
        "mip_0.1_50_0.0",
        "mip_0.1_10_0.0",
        "brunel_SI_slow",
         ]

    kernel_names = [
        'large_population',
        'narrow_input_region',
        'similar_synapses',
        'default',
        'small_population',
        'variable_synapses',
        'broad_input_region',
        'uniform',  # 'apical',
                    ]
    num_kernel_types = len(kernel_names)
    num_corrtypes = len(correlation_types)


    plt.close("all")
    fig = plt.figure(figsize=[6, 3.5])

    fig.subplots_adjust(bottom=0.0, top=0.94, right=0.7, wspace=0.05,
                        left=0.2, hspace=0.6)

    # rate = 25
    # correlation = 0.0
    # jitter = 0.0
    num_kernels = 100

    ax_var = fig.add_subplot(1, 2, 1, #title="max SD"
                             )
    # ax_err_obs = fig.add_subplot(1, 3, 2, title="error")
    ax_rel_err_obs = fig.add_subplot(1, 2, 2, #title="relative error"
                                     )

    ax_relation = fig.add_axes([0.77, 0.535, 0.20, 0.4],
                               xlabel="max signal amplitude\n(norm.)",
                               ylabel="relative error")
    x__ = np.array([0.01, 1])
    ax_relation.plot(x__, 0.01/x__, lw=0.5, ls='--', c='k')
    # ax_err_theory = fig.add_subplot(2, 3, 5)
    # ax_rel_err_theory = fig.add_subplot(2, 3, 6)

    mean_var = np.zeros((num_kernel_types, num_corrtypes, ))
    mean_error_obs = np.zeros((num_kernel_types, num_corrtypes, ))
    mean_error_theory = np.zeros((num_kernel_types, num_corrtypes, ))

    mean_rel_error_obs = np.zeros((num_kernel_types, num_corrtypes, ))
    mean_rel_error_theory = np.zeros((num_kernel_types, num_corrtypes, ))

    all_STDs = []
    all_rel_errors = []

    all_STDs_eeg = []
    all_rel_errors_eeg = []
    M_eeg = neural_simulations.M_eeg * 1e-3

    for c_idx, correlation_type in enumerate(correlation_types):

        if "brunel" in correlation_type:
            jitter = None
            correlation = None
            rate = None
        else:
            fr_params = correlation_type.split('_')
            correlation = float(fr_params[1])
            rate = int(fr_params[2])
            jitter = float(fr_params[3])
            correlation_type = fr_params[0]

        for k_idx, kernel_name in enumerate(kernel_names):

            k_part = f'{kernel_name}'

            if 'brunel' in correlation_type:
                fr_part = f'corrtype:{correlation_type}'
            else:
                fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(rate,
                                                                         correlation_type,
                                                                         correlation,
                                                                         jitter)

            sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

            phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
            phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))

            var_diff = np.sqrt(np.load(join(data_folder, f'error_observed_{sim_name}.npy')))
            rel_norm = np.max(np.std(phi, axis=1))
            # var_diff = np.std(phi - phi_tilde, axis=1)

            eeg = np.load(join(data_folder, f'cdm_{sim_name}.npy'))[2,:] * M_eeg
            eeg_tilde = np.load(join(data_folder, f'cdm_tilde_{sim_name}.npy'))[2,:] * M_eeg

            rel_norm_eeg = np.std(eeg)
            var_diff_eeg = np.std(eeg - eeg_tilde)


            mean_var[k_idx, c_idx] = rel_norm

            mean_rel_error_obs[k_idx, c_idx] = np.max(var_diff) / rel_norm
            if kernel_name == 'default':
                print(sim_name, mean_rel_error_obs[k_idx, c_idx])
            # mean_rel_error_theory[k_idx, c_idx] = np.mean(error_theory) / rel_norm
            all_STDs.append(rel_norm)
            all_rel_errors.append(np.max(var_diff) / rel_norm)

            all_STDs_eeg.append(rel_norm_eeg)
            all_rel_errors_eeg.append(np.max(var_diff_eeg) / rel_norm_eeg)


    img1 = ax_var.imshow(mean_var, vmin=0, vmax=2000**0.5)
    cbar1 = plt.colorbar(img1, ax=ax_var, shrink=0.78, label="max LFP amplitude (µV)",
                         orientation="horizontal", pad=.42, extend="max")

    img3 = ax_rel_err_obs.imshow(mean_rel_error_obs, vmin=0.075, vmax=0.3)
    cbar3 = plt.colorbar(img3, ax=ax_rel_err_obs, shrink=0.78, extend='both',
                         orientation="horizontal", pad=.42,
                         label="relative error")

    for ax in [ax_var, ax_rel_err_obs]:
        ax.set_xticks(np.arange(num_corrtypes))
        ax.set_yticks(np.arange(num_kernel_types))
        xlabels = []
        for c_ in correlation_types:
            n_ = c_.replace('_', ' ').replace(
                'mip', 'MIP')
            n_ = n_.replace('brunel', 'Brunel,').replace("AI slow", "AI")
            if "MIP" in n_:
                params = n_.split(' ')
                n_ = (params[0] + f', $f$={float(params[1]):0.1f}, ' +
                      f'$ν$={params[2]} s⁻¹S')
                print(n_)

            xlabels.append(n_)
        ax.set_xticklabels(xlabels, rotation=-90)
        ax.set_yticklabels([])

    for ax in [ax_var]:

        ax.set_yticklabels([k_.replace('_', ' ').replace("population", r'$K_{\rm out}$')
                            for k_ in kernel_names])

    all_STDs = np.array(all_STDs) / np.max(all_STDs)
    all_STDs_eeg = np.array(all_STDs_eeg) / np.max(all_STDs_eeg)
    l_lfp, = ax_relation.loglog(all_STDs, all_rel_errors, 'k.', clip_on=False)
    l_eeg, = ax_relation.loglog(all_STDs_eeg, all_rel_errors_eeg, '.',
                                c='gray', clip_on=False)

    ax_relation.legend([l_lfp, l_eeg], ["LFP", "EEG"],
                       frameon=False, loc=(0.7, .9), ncol=1)

    simplify_axes([ax_var, ax_rel_err_obs, ax_relation])
    mark_subplots([ax_var, ax_rel_err_obs, ax_relation], xpos=0., ypos=1.08)
    fig.savefig(f"error_summary_figure_eeg.png")
    fig.savefig(f"error_summary_figure_eeg.pdf")


def plot_all_signals_and_kernels(data_folder, firing_rate_folder):
    if rank != 0:
        return

    correlation_types = [
                         "mip_0.0_10_0.0",
                         # "mip_0.1_12_0.0",
                         # "mip_0.0_50_0.0",
                         # "mip_0.1_50_0.0",
                         # "brunel_AI_slow",
                         # "brunel_SI_slow",
                         ]


    kernel_names = [
        'default',
        'apical',
        'uniform',
        'small_population',
        'large_population',
        'narrow_input_region',
        'broad_input_region',
        'similar_synapses',
        'variable_synapses',
        ]

    num_rows = 2
    num_cols = len(kernel_names) + 1

    for c_idx, correlation_type in enumerate(correlation_types):
        #fr_part = f'corrtype:{correlation_type}'
        if "brunel" in correlation_type:
            fr_part = f'corrtype:{correlation_type}'
        else:
            fr_params = correlation_type.split('_')
            correlation = float(fr_params[1])
            rate = int(fr_params[2])
            jitter = float(fr_params[3])
            # correlation_type = fr_params[0]
            fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(rate,
                                                                     fr_params[0],
                                                                     correlation,
                                                                     jitter)

        fr_dict = np.load(join(firing_rate_folder, f'corrtype:{correlation_type}.npy'),
                          allow_pickle=True)[()]
        firing_rate = fr_dict['firing_rate']
        #avrg_firingrate = np.average(firing_rate, axis=0)

        tvec = np.arange(firing_rate.shape[1]) * neural_simulations.dt
        # print(firing_rate.shape, tvec.shape)
        tlim = [2000, 2500]
        t0 = np.argmin(np.abs(tvec - tlim[0]))
        t1 = np.argmin(np.abs(tvec - tlim[1]))

        plt.close("all")
        fig = plt.figure(figsize=[6, 2.7])

        fig.subplots_adjust(bottom=0.04, top=0.87, right=0.96, wspace=0.05,
                            left=0.05, hspace=0.15)

        ax_pop = fig.add_axes([0.01, 0.54, 0.08, 0.37], aspect=1,
                                frameon=False, xticks=[], ylim=[-1700, 10],
                              yticks=[], title=f"population", rasterized=True)

        cell = neural_simulations.return_hay_cell(tstop=100,
                                                  dt=neural_simulations.dt,
                                                  make_passive=True)
        cell_z_top = np.max(cell.z)
        soma_z_mean = -cell_z_top - 150
        max_soma_loc = -cell_z_top + cell.z[0].mean() - 10
        data_dict = np.load(join(data_folder, f"kernels_case_default.npy"),
                                allow_pickle=True)[()]
        pop_params = data_dict['parameters']
        np.random.seed(1234)
        soma_loc_dist, soma_xs, soma_ys, soma_zs, z_rots = \
            neural_simulations.return_soma_loc_dist(pop_params, soma_z_mean, max_soma_loc)


        num_plot_cells = 100
        for k_idx in range(num_plot_cells):

            cell.set_pos(x=soma_xs[k_idx],
                         y=soma_ys[k_idx],
                         z=soma_zs[k_idx],
                         )
            cell.set_rotation(z=z_rots[k_idx])
            cell_clr_ = plt.cm.Greys(0.1 + k_idx / num_plot_cells / 5)
            ax_pop.plot(cell.x.T, cell.z.T, c=cell_clr_, zorder=-1e9)
        ax_pop.plot(elec_x, elec_z, 'o', c=neural_simulations.elec_clr,
                    mec='none', ms=3)
        ax_fr = fig.add_axes([0.01, 0.04, 0.08, 0.37], xlim=tlim,
                                frameon=False, xticks=[], yticks=[],
                              title=f"spikes")

        ax_fr.plot([tlim[1] - 200, tlim[1]], [-2, -2], c='k', lw=1, clip_on=False)
        ax_fr.text(tlim[1] - 100, -4, f'200 ms', va='top', ha='center')

        for t_idx in range(100):
            spks = tvec[np.where(firing_rate[t_idx] > 0.)]
            ax_fr.plot(spks, t_idx + np.zeros(len(spks)), '|',
                    ms=2, c='k')

        for k_case_idx, kernel_name in enumerate(kernel_names):
            data_dict = np.load(join(data_folder, f"kernels_case_{kernel_name}.npy"),
                                allow_pickle=True)[()]
            pop_params = data_dict['parameters']

            k_part = f'{kernel_name}'
            # lfp_kernels = np.array(data_dict['kernel_trials'])
            lfp_kernels = np.array(data_dict['kernel_trials'])

            num_kernels, num_elecs, num_tsteps = lfp_kernels.shape

            tvec_k = np.arange(num_tsteps) * neural_simulations.dt
            tvec_k -= tvec_k[int(len(tvec_k) / 2)]
            lfp_mean_kernel = lfp_kernels.mean(axis=0)
            kernels_norm = 8#np.max(np.abs(lfp_mean_kernel))
            lfp_norm = 100
            sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

            phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
            phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))

            baseline = np.mean(phi[:, t0:t1], axis=1)
            phi -= baseline[:, None]
            phi_tilde -= baseline[:, None]

            ax1 = fig.add_subplot(num_rows, num_cols, k_case_idx + 2, frameon=False,
                                  ylim=[-1700, 10],
                                  yticks=[], xlim=[0, 20], xticks=[])


            ax2 = fig.add_subplot(num_rows, num_cols, num_cols + k_case_idx + 2,
                                  frameon=False, ylim=[-1700, 10], xticks=[],
                                  xlim=tlim,
                                  yticks=[])

            if k_case_idx == 0:
                ax1.set_ylabel("LFP kernels")
                ax2.set_ylabel("LFP signals")
            ax1.set_title(kernel_name.replace('_', '\n').replace("population", r"$K_{\rm out}$"),
                          fontsize=6)
            for elec_idx in range(num_elecs):
                for k_idx in range(num_kernels):
                    l0, = ax1.plot(tvec_k, lfp_kernels[k_idx, elec_idx] / kernels_norm * dz + elec_z[elec_idx],
                             c='gray', lw=0.5, zorder=-1, rasterized=True)
                l1, = ax1.plot(tvec_k, lfp_mean_kernel[elec_idx] / kernels_norm * dz + elec_z[elec_idx],
                         c='k', lw=1.0, zorder=-1)

                l2, = ax2.plot(tvec, phi[elec_idx, :] / lfp_norm * dz + elec_z[elec_idx],
                         c='gray', lw=1., zorder=-1)
                l3, = ax2.plot(tvec, phi_tilde[elec_idx, :] / lfp_norm * dz + elec_z[elec_idx],
                         c='k', lw=0.5, zorder=-1)
                # ax2.plot(tvec, (phi[elec_idx] - phi_tilde[elec_idx]) / lfp_norm * dz + elec_z[elec_idx],
                #          c='r', lw=0.5, zorder=-1)

            if k_case_idx == (len(kernel_names) - 1):

                ax1.plot([20.5, 20.5], [-1000, -1000 - dz], c='k', lw=1,
                      clip_on=False)
                ax1.text(21, -1000 - dz/2, f'{kernels_norm:0.0f}\nµV',
                      va='center', ha='left')
                ax2.plot([tlim[1] + 5, tlim[1] + 5], [-1000, -1000 - dz], c='k', lw=1,
                              clip_on=False)
                ax2.text(tlim[1] + 10, -1000 - dz/2, f'{lfp_norm:0.0f}\nµV',
                              va='center', ha='left')

                ax1.plot([10, 20], [-1650, -1650], c='k', lw=1, clip_on=False)
                ax1.text(15, -1690, f'10 ms', va='top', ha='center')

                ax2.plot([tlim[1] - 200, tlim[1]], [-1650, -1650], c='k', lw=1, clip_on=False)
                ax2.text(tlim[1] - 100, -1690, f'200 ms', va='top', ha='center')

                ax1.legend([l0, l1],
                           ["single-cell", "population kernel"],
                           frameon=False, ncol=2, loc=(-6, -0.1))
                ax2.legend([l2, l3],
                           ["ground truth", "population kernel"],
                           frameon=False, ncol=2, loc=(-6, -0.1))

        plt.savefig(f'kernels_and_signals_{correlation_type}.pdf')
        plt.savefig(f'kernels_and_signals_{correlation_type}.png')


def illustrate_firing_rates(firing_rate_folder):
    correlation_types = [
        "mip_0.0_10_0.0",
        "mip_0.1_10_0.0",
        "brunel_AI_slow",
        "brunel_SI_slow",
        ]

    plt.close("all")

    fig2 = plt.figure(figsize=[10, 10])
    fig2.subplots_adjust(left=0.075, right=0.98, hspace=0.12,
                        wspace=0.8, bottom=0.12, top=0.88)

    dt = neural_simulations.dt
    fig = plt.figure(figsize=[6, 4])
    fig.subplots_adjust(left=0.12, right=0.98, hspace=0.1,
                        wspace=0.4, bottom=0.12, top=0.88)
    xlim = [3750, 4250]

    axes_to_mark = []
    for c_idx, correlation_type in enumerate(correlation_types):
        title = correlation_type.replace('_', ' ').replace(
            'brunel ', 'Brunel\n').replace("AI slow", "AI")
        # title = title.replace("mip", "MIP\n")
        if "mip" in title:
            params = correlation_type.split('_')
            title = f"MIP\n$f$={params[1]}\n$ν$={params[2]} s⁻¹"

        shifter = 0# if c_idx > 1 else 0
        col_idx = 0
        ax = fig.add_subplot(2, 4, c_idx + 1 + shifter,
                             xlim=xlim,
                             yticks=[], frameon=False, xticks=[])
        axes_to_mark.append(ax)
        ax_a = fig.add_subplot(2, 4,
                               1 * int(len(correlation_types))
                               + c_idx + 1 + shifter,
                               xlim=[-100, 100],
                               ylim=[-0.0025, 0.013],
                               #title=r'a$_{\rm mean}$',
                               xlabel="$t$ (ms)")
        if c_idx == 0:
            ax_a.set_ylabel('normalized correlation')
        else:
            ax_a.set_yticklabels([])

        ax.set_title(title)
        fr_part_name = f"corrtype:{correlation_type}"
        # if not 'brunel' in correlation_type:
        #     fr_part_name += f'_{correlation}_{rate}_{jitter}'
        fr_dict = np.load(join(firing_rate_folder, f'{fr_part_name}.npy'),
                          allow_pickle=True)[()]
        firing_rate = fr_dict['firing_rate']
        t = np.arange(firing_rate.shape[1]) * dt

        print(firing_rate.shape)
        print(t[-1])

        spiketimes = [np.where(firing_rate[idx] > 0.5)[0]
                      for idx in range(len(firing_rate))]
        sc_corr_coeff = ca.spike_count_correlation_coefficient(spiketimes, 10, t[-1])
        avrg_num_spikes = np.sum(firing_rate * dt / 1000) / firing_rate.shape[0]

        pop_firing_rate = np.mean(firing_rate, axis=0)

        sim_dur = dt * firing_rate.shape[1]
        # print(sim_dur)
        avrg_fr = avrg_num_spikes / sim_dur * 1000

        print(f"{correlation_type} \t mean FR: {np.mean(pop_firing_rate):0.1f}\t cc: {sc_corr_coeff}")

        ax__1 = fig2.add_subplot(2, len(correlation_types), c_idx + 1,
                             xlim=xlim, title=correlation_type, ylim=[-100, 2500])

        ax__2 = fig2.add_subplot(2, len(correlation_types), c_idx + 1 + len(correlation_types),
                             xlim=xlim, title=correlation_type, ylim=[-100, 2500])
        ax__2.plot(t, pop_firing_rate, lw=0.5, c='k')

        ax__1.plot(t, firing_rate[0])
        ax__1.plot(t[spiketimes[0]],
                   np.zeros(len(spiketimes[0])), '*')

        # low-pass filter firing rate for visualization:
        sigma_g = 2
        gx = np.arange(-3 * sigma_g, 3 * sigma_g, neural_simulations.dt)
        gaussian = np.exp(-(gx / sigma_g) ** 2 / 2)
        gaussian /= np.sum(gaussian)
        # plt.plot(gx, gaussian)
        # plt.show()
        fr_lp = np.convolve(pop_firing_rate, gaussian, mode="same")

        ax__2.plot(t, fr_lp, 'r')

        t_a  = fr_dict['t_a']
        t_c  = fr_dict['t_c']

        from scipy.stats import gmean

        g_mean = gmean(np.var(firing_rate, axis=1))

        c_mean  = fr_dict['c_mean'] / g_mean
        a_mean  = fr_dict['a_mean'] / g_mean
        l1, = ax_a.plot(t_a, a_mean, c='k')
        l2, = ax_a.plot(t_c, c_mean, c='gray')


        for t_idx in range(100):
            spks = t[np.where(firing_rate[t_idx] > 0.)]
            ax.plot(spks, t_idx + np.zeros(len(spks)),
                   '|', ms=2, c='k')

        if c_idx == 0:
            ax.plot([xlim[1] - 200, xlim[1]], [-5, -5], c='k', lw=1)
            ax.text(xlim[1] - 100, -8, "200 ms", va="top", ha="center")

    fig.legend([l1, l2], [r'$A_{s}$', r'$C_{s}$'],
               frameon=False, loc=(0.65, 0.0), ncol=3)
    mark_subplots(axes_to_mark, ypos=1.2, xpos=-0.15)
    simplify_axes(fig.axes)
    simplify_axes(fig2.axes)
    fig.savefig("illustrate_firing_rates.png")
    fig.savefig("illustrate_firing_rates.pdf")

    fig2.savefig(f"pop_firing_rate_variability.png")


def return_rate_model_fr(dt):


    cutoff = 0.5
    T = 5   

    eta_bar = -10.
    Delta = 2.
    J = 15 * np.sqrt(Delta)
    tau = 100e-3  # ms


    times = np.arange(0, T + cutoff, dt)
    I = np.zeros_like(times)
    r = np.zeros_like(times)
    v = np.zeros_like(times)

    r[0] = 6
    v[0] = -1
    gamma = 4.719

    I[int(1 / dt):int(3 / dt)] = 4.
    #I = A * (gamma * np.sin(np.pi * f * times) **20 - 1)
    #I -= np.mean(I)
    print(np.sum(I))
    for i, t in enumerate(times[:-1]):
        r[i + 1] = r[i] + dt / tau**2 * (Delta / np.pi + 2 * r[i] * v[i] * tau)
        v[i + 1] = v[i] + dt / tau * (v[i] ** 2 + eta_bar + J * r[i] * tau + I[i] - np.pi ** 2 * r[i] ** 2* tau**2)

    idx0 = np.argmin(np.abs(times - cutoff))

    times = times[idx0:] - times[idx0]
    r = r[idx0:]
    I = I[idx0:]
    v = v[idx0:]

    plt.close("all")
    plt.subplot(311)
    plt.plot(times, I)
    plt.subplot(312)

    plt.plot(times, r)
    # plt.subplot(313)
    # plt.plot(times, v)

    plt.savefig("ratetesting.png")
    #sys.exit()
    return times, r


def rate_model_figure(data_folder):
    if rank != 0:
        return
    case_name = 'default'
    data_dict = np.load(join(data_folder, f"kernels_case_{case_name}.npy"),
                        allow_pickle=True)[()]
    params = data_dict['parameters']

    avrg_kernel = np.array(data_dict['kernel_trials']).mean(axis=0)

    presyn_pop_size = 10000

    dt = 2**-4
    kernel_length = avrg_kernel.shape[1]
    tvec_k = np.arange(kernel_length) * dt
    tvec_k -= tvec_k[int(kernel_length / 2)]
    eeg_kernel = np.array(data_dict['cdm_trials']).mean(axis=0)[2, :] * M_eeg

    times, firing_rate = return_rate_model_fr(dt/1000)
    #times = np.arange(int(1 / dt * 1000)) * dt / 1000
    #firing_rate = np.zeros(len(times))
    #firing_rate[int(len(firing_rate) / 2)] = 1

    num_elecs = avrg_kernel.shape[0]
    signal_length = len(times)
    phi_tilde = np.zeros((num_elecs, signal_length))
    eeg_tilde = np.zeros(signal_length)

    # The convolution needs spikes per dt instead of Hz:
    spikes_per_dt = firing_rate * dt / 1000 * presyn_pop_size

    for elec_idx in range(num_elecs):

        phi_tilde[elec_idx] = np.convolve(avrg_kernel[elec_idx, :],
                                               spikes_per_dt,
                                               mode="same")
        eeg_tilde = np.convolve(eeg_kernel, spikes_per_dt, mode="same")
        # self.phi_tilde[elec_idx, :] = self.phi_tilde[self.kernel_length:]

    elec_clr = neural_simulations.elec_clr
    eeg_clr = neural_simulations.eeg_clr
    eeg_plot_loc = 120
    plt.close("all")
    fig = plt.figure(figsize=[6, 2])
    fig.subplots_adjust(top=0.9, right=0.97, wspace=0.3, left=0.05, bottom=0.04)
    ax1 = fig.add_subplot(131, title="firing rate",
                          frameon=False, xticks=[], yticks=[]
                          )
    ax2 = fig.add_subplot(132, xlim=[-0, 20],
                          title="kernels",
                          frameon=False, xticks=[], yticks=[])
    ax3 = fig.add_subplot(133, yticklabels=[], title="signals",
                          frameon=False, xticks=[], yticks=[])

    lfp_norm = np.max(np.abs(phi_tilde))
    kernel_norm = np.max(np.abs(avrg_kernel))
    eeg_kernel_norm = np.max(np.abs(eeg_kernel))
    eeg_norm = np.max(np.abs(eeg_tilde))

    for elec_idx in range(num_elecs):
        ax2.plot(tvec_k, avrg_kernel[elec_idx, :] / kernel_norm * dz + elec_z[elec_idx],
                 c='k', lw=1)
        l1, = ax3.plot(times, phi_tilde[elec_idx] / lfp_norm * dz + elec_z[elec_idx],
                          lw=1, c='k', ls='-')

    ax2.plot(tvec_k,
                  eeg_kernel / eeg_kernel_norm * dz + eeg_plot_loc,
                  c=eeg_clr, lw=1., zorder=1)
    ax3.plot(times,
                  eeg_tilde / eeg_norm * dz + eeg_plot_loc,
                  c=eeg_clr, lw=1., zorder=1)

    ax2.plot([20.5, 20.5], [-1000, -1000 - dz], c='k', lw=1,
                  clip_on=False)
    ax2.text(21, -1000 - dz/2, f'{kernel_norm:0.0f}\nµV',
                  va='center', ha='left')

    ax3.plot([times[-1] + 0.01, times[-1] + 0.01], [-1000, -1000 - dz], c='k', lw=1,
                  clip_on=False)
    ax3.text(times[-1] + 0.02, -1000 - dz/2, f'{lfp_norm/1000:0.0f}\nmV',
                  va='center', ha='left')

    ax3.plot([times[-1] + 0.01, times[-1] + 0.01], [eeg_plot_loc, eeg_plot_loc - dz],
             lw=1, clip_on=False, c=eeg_clr)
    ax3.text(times[-1] + 0.02, eeg_plot_loc - dz/2, f'{eeg_norm/1000:0.0f}\nµV',
                  va='center', ha='left', c=eeg_clr)

    ax3.plot(elec_x, elec_z, 'o', c=elec_clr, ms=3., clip_on=False)
    ax3.text(-0.05, np.max(elec_z) + 20, "LFP", color=elec_clr, ha='right',
             va="top")
    ax2.plot(elec_x, elec_z, 'o', c=elec_clr, ms=3., clip_on=False)
    ax2.text(-1, np.max(elec_z) - 0, "LFP\nkernel", color=elec_clr, ha='right',
             va="top")

    ax2.plot([20.5, 20.5], [eeg_plot_loc, eeg_plot_loc - dz],
             lw=1, clip_on=False, c=eeg_clr)
    ax2.text(21, eeg_plot_loc - dz/2, f'{eeg_kernel_norm:0.1f}\nnV',
                  va='center', ha='left', c=eeg_clr)

    ax3.text(-0.02, eeg_plot_loc + 25, "EEG", color=eeg_clr, ha='right',
             va="bottom")
    ax3.plot(0, eeg_plot_loc, 'o', c=eeg_clr, clip_on=False)
    ax2.text(-1, eeg_plot_loc + 130, "EEG\nkernel", color=eeg_clr, ha='right',
             va="top")
    ax2.plot(0, eeg_plot_loc, 'o', c=eeg_clr, clip_on=False)

    ax2.plot([-0.8, -0.8], [-800, -800 - dz], c=elec_clr, lw=1, clip_on=False)
    ax2.text(-1.2, -800 - dz/2, f"{dz}\nµm", ha='right', va='center', c=elec_clr)

    ax1.plot([2, 3], [0.5, 0.5], lw=1, c='k')
    ax1.text(2.5, 0.55, "1 s", ha='center', va='bottom')

    ax3.plot([2, 3], [-1650, -1650], lw=1, c='k')
    ax3.text(2.5, -1700, "1 s", ha='center', va='top')

    ax2.plot([5, 15], [-1650, -1650], lw=1, c='k')
    ax2.text(10, -1700, "10 ms", ha='center', va='top')

    ax1.plot([0.05, 0.05], [5, 15], lw=1, c='k')
    ax1.text(0.01, 10, "10 s⁻¹", ha='right', va='center')

    ax1.plot(times, firing_rate, 'k', lw=1)
    simplify_axes(fig.axes)
    mark_subplots(fig.axes, ypos=1.05)

    plt.savefig("rate_model_example_remade.pdf")


def investigate_error_measure(data_folder, firing_rate_folder):
    if rank != 0:
        return
    correlation_types = [
        "mip_0.0_10_0.0",
        #"mip_0.1_1_0.0",
        # "mip_0.1_10_0.0",
        # "mip_0.0_50_0.0",
        # "mip_0.1_50_0.0",
        # "mip_0.1_12_0.0",
        # "mip_0.0_50_0.0",
        # "mip_0.1_50_0.0",
        # "brunel_AI_slow",
        # "brunel_AI_fast",
        # "brunel_SI_slow",
        # "brunel_SI_fast",
        # "brunel_SR"
        ]

    kernel_names = [
        'default',
        # 'apical',
        # 'uniform',
        # 'small_population',
        # 'large_population',
        # 'narrow_input_region',
        # 'broad_input_region',
        # 'similar_synapses',
        # 'variable_synapses',
        ]


    for correlation_type in correlation_types:
        for kernel_name in kernel_names:
            #fr_part = f'corrtype:{correlation_type}'
            if "brunel" in correlation_type:
                # jitter = None
                # correlation = None
                # rate = None
                fr_part = f'corrtype:{correlation_type}'
            else:
                fr_params = correlation_type.split('_')
                correlation = float(fr_params[1])
                rate = int(fr_params[2])
                jitter = float(fr_params[3])
                # correlation_type = fr_params[0]
                fr_part = "rate:{}corrtype:{}({:1.3f})_jitter:{}".format(rate,
                                                                         fr_params[0],
                                                                         correlation,
                                                                         jitter)
            k_part = f'{kernel_name}'

            data_dict = np.load(join(data_folder, f"kernels_case_{kernel_name}.npy"),
                                allow_pickle=True)[()]

            lfp_kernels = np.array(data_dict['kernel_trials'])

            num_kernels, num_elecs, num_tsteps = lfp_kernels.shape

            tvec_k = np.arange(num_tsteps) * neural_simulations.dt
            tvec_k -= tvec_k[int(len(tvec_k) / 2)]

            fr_dict = np.load(join(firing_rate_folder, f'corrtype:{correlation_type}.npy'),
                              allow_pickle=True)[()]
            firing_rate = fr_dict['firing_rate']
            avrg_firingrate = np.average(firing_rate, axis=0)

            # low-pass filter firing rate for visualization:
            sigma_g = 1
            gx = np.arange(-3 * sigma_g, 3 * sigma_g, neural_simulations.dt)
            gaussian = np.exp(-(gx / sigma_g) ** 2 / 2)
            # plt.plot(gx, gaussian)
            # plt.show()
            fr_lp = np.convolve(avrg_firingrate, gaussian, mode="same")

            lfp_mean_kernel = lfp_kernels.mean(axis=0)
            kernels_norm = 8#np.max(np.abs(lfp_mean_kernel))
            lfp_norm = 50

            sim_name = f"num_kernels:{num_kernels}_{k_part}_{fr_part}"

            phi = np.load(join(data_folder, f'phi_{sim_name}.npy'))
            phi_tilde = np.load(join(data_folder, f'phi_tilde_{sim_name}.npy'))
            error_theory = np.load(join(data_folder, f'error_theory_{sim_name}.npy'))
            error_observed = np.load(join(data_folder, f'error_observed_{sim_name}.npy'))


            num_elecs, num_tsteps = phi.shape

            R2 = np.zeros(num_elecs)
            for elec_idx in range(num_elecs):
                corr_matrix = np.corrcoef(phi[elec_idx], phi_tilde[elec_idx])
                corr = corr_matrix[0, 1]
                R2[elec_idx] = corr ** 2

            tvec = np.arange(num_tsteps) * neural_simulations.dt
            # print(firing_rate.shape, tvec.shape)
            tlim = [500, 1000]
            t0 = np.argmin(np.abs(tvec - tlim[0]))
            t1 = np.argmin(np.abs(tvec - tlim[1]))

            phi -= np.mean(phi[:, t0:t1], axis=1)[:, None]
            phi_tilde -= np.mean(phi_tilde[:, t0:t1], axis=1)[:, None]

            #difference = phi - phi_tilde
            plt.close('all')
            fig = plt.figure(figsize=[6, 2.5])
            fig.subplots_adjust(bottom=0.19, top=0.93, left=0.04, right=0.99)

            ax1 = fig.add_subplot(141, frameon=False, title="LFP kernel",
                                  ylim=[-1700, 10],
                                  yticks=[], xlim=[0, 15], xticks=[])

            ax2 = fig.add_subplot(142, title="spikes",
                                  frameon=False, xticks=[],
                                  xlim=tlim,
                                  yticks=[])
            ax2b = fig.add_axes([ax2.axes.get_position().x0, 0.03,
                                 ax2.axes.get_position().width, 0.1],
                                xlim=tlim,frameon=False, xticks=[], yticks=[],
                                )
            ax2b.set_title('population rate', pad=-15)
            ax3 = fig.add_subplot(143, yticklabels=[], title="LFP signal",
                                  frameon=False, xticks=[], yticks=[],
                                  ylim=[-1700, 10], xlim=tlim)

            ax5 = fig.add_subplot(144, xlabel="relative error",
                                  ylim=[-1700, 10], yticklabels=[],
                                  yticks=elec_z)

            for t_idx in range(100):
                spks = tvec[np.where(firing_rate[t_idx] > 0.)]
                ax2.plot(spks, t_idx + np.zeros(len(spks)),
                          '|', ms=2, c='gray')
            # ax2b.plot(tvec, avrg_firingrate, 'k')
            ax2b.plot(tvec, fr_lp, 'k', lw=0.5)

            for elec_idx in range(num_elecs):
                for k_idx in range(num_kernels):
                    l0, = ax1.plot(tvec_k, lfp_kernels[k_idx, elec_idx
                        ] / kernels_norm * dz + elec_z[elec_idx],
                                   c='gray', lw=0.5, zorder=-1)
                l0a, = ax1.plot(tvec_k, lfp_mean_kernel[elec_idx
                    ] / kernels_norm * dz + elec_z[elec_idx],
                               c='k', lw=0.5, zorder=-1)

                l1, = ax3.plot(tvec, phi[elec_idx] / lfp_norm * dz +
                               elec_z[elec_idx],
                                  lw=1.5, c='gray', ls='-')
                l2, = ax3.plot(tvec, phi_tilde[elec_idx] / lfp_norm * dz +
                               elec_z[elec_idx],
                                  lw=0.5, c='k', ls='-')
                l3, = ax3.plot(tvec,
                               (phi[elec_idx] - phi_tilde[elec_idx]
                                ) / lfp_norm * dz + elec_z[elec_idx],
                                  lw=0.5, c='red', ls='-')

            #ax4.plot(R2, elec_z, c='k', lw=0.5)
            #ax4.axvline(1, ls='--', lw=.5, c='gray')

            # ax5.plot(np.std(difference, axis=1)/ np.max(np.std(phi, axis=1)),
            #          elec_z, c='k', lw=1)
            lt, = ax5.plot(np.sqrt(error_theory) / np.max(np.std(phi, axis=1)),
                     elec_z, c='k', lw=1., ls=':')
            lo, = ax5.plot(np.sqrt(error_observed) / np.max(np.std(phi, axis=1)),
                     elec_z, c='k', lw=0.5, ls='-')

            ax1.legend([l0, l0a], ["single-cell", "population kernel"],
                       frameon=False, loc=(-0.25,-0.22), ncol=1)
            ax3.legend([l1, l2, l3], ["ground truth", "population kernel",
                                      "difference"],
                       frameon=False, loc=(-0.1,-0.28), ncol=1)
            ax5.legend([lo, lt], ['simulated', 'theory'], frameon=False)

            ax1.plot([15.5, 15.5], [-1000, -1000 - dz], c='k', lw=1,
                     clip_on=False)
            ax1.text(16, -1000 - dz / 2, f'{kernels_norm:0.0f}\nµV',
                     va='center', ha='left')
            ax1.plot([5, 15], [-1650, -1650], c='k', lw=1, clip_on=False)
            ax1.text(10, -1690, f'10 ms', va='top', ha='center')

            ax2.plot([tlim[1] - 200, tlim[1]], [-2, -2], c='k', lw=1,
                     clip_on=False)
            ax2.text(tlim[1] - 50, -4, f'200 ms', va='top', ha='center')

            ax3.plot([tlim[1] + 5, tlim[1] + 5], [-1000, -1000 - dz],
                     c='k', lw=1, clip_on=False)
            ax3.text(tlim[1] + 10, -1000 - dz / 2, f'{lfp_norm:0.0f}\nµV',
                     va='center', ha='left')

            ax3.plot([tlim[1] - 200, tlim[1]], [-1650, -1650], c='k', lw=1,
                     clip_on=False)
            ax3.text(tlim[1] - 100, -1690, f'200 ms', va='top', ha='center')

            simplify_axes(fig.axes)
            mark_subplots([ax1, ax2, ax3, ax5], ypos=1.02)
            os.makedirs(join("error_investigations"), exist_ok=True)
            fig.savefig(join("error_investigations",
                             f"error_measure_investigation_{kernel_name}_{correlation_type}_delta.png"))
            fig.savefig(join("error_investigations",
                             f"error_measure_investigation_{kernel_name}_{correlation_type}_delta.pdf"))


def gather_pop_kernels(data_folder):
    if rank != 0:
        return
    import json
    sim_names = ['default',
                 'small_population', 'large_population',
                 'small_radius', 'large_radius',
                 'uniform', 'apical',
                 'similar_synapses', 'variable_synapses',
                 'broad_input_region', 'narrow_input_region'
                 ]

    for case_name in sim_names:

        data_dict = np.load(join(data_folder, f"kernels_case_{case_name}.npy"),
                            allow_pickle=True)[()]
        params = data_dict['parameters']
        lfp_kernel = np.array(data_dict['kernel_trials']).mean(axis=0)
        cdm_kernel = np.array(data_dict['cdm_trials']).mean(axis=0)[2, :]

        pop_dict = {'name': case_name,
                    'params': params,
                    'lfp_kernel': lfp_kernel.tolist(),
                    'cdm_kernel': cdm_kernel.tolist()}

        with open(join(data_folder, f"pop_kernel_{case_name}.json"), "w") as file:
            json.dump(pop_dict, file)


def run_parameter_scan_biophysical_model(figure_folder, data_folder, firing_rate_folder):
    correlation_types = [
        "brunel_AI_slow",
        "brunel_SI_slow",
        "mip_0.0_10_0.0",
        "mip_0.1_10_0.0",
        "mip_0.0_50_0.0",
        "mip_0.1_50_0.0",
    ]

    sim_names = ['default',
                 'small_population', 'large_population',
                 'small_radius', 'large_radius',
                 'uniform', 'apical',
                 'similar_synapses', 'variable_synapses',
                 'broad_input_region', 'narrow_input_region'
                 ]

    # Scan through combinations
    task_count = 0
    for k_idx, kernel_name in enumerate(sim_names):
        for c_idx, c_type in enumerate(correlation_types):
            if task_count % size == rank:
                np.random.seed(2514)
                print(f"Rank {rank} on task {task_count}")
                run_simpop_example(figure_folder, data_folder, kernel_name,
                               c_type, firing_rate_folder)
            task_count += 1
        # Additional scan for MIPs and default kernels:
    task_count = 0
    for fr in [5, 10, 50, 100]:
        for corr_value in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        # for corr_value in [1.0]:
            if task_count % size == rank:
                c_type = f"mip_{corr_value}_{fr}_0.0"
                np.random.seed(2514)
                run_simpop_example(figure_folder, data_folder, "default",
                                   c_type, firing_rate_folder)
            task_count += 1


def simplest_toy_illustration(figure_folder):
    num_kernels = 2
    tau_1 = 0.2
    tau_2 = 1.0
    dt = 0.1
    kernel_length = 200
    signal_length = 1000

    kernel_amps = np.array([0.75, 1.25])
    spike_times = np.array([[35, 50, 70],
                            [25, 50, 90]])

    t_kernel = np.arange(kernel_length) * dt
    t_signal = np.arange(signal_length) * dt

    kernels = np.zeros((num_kernels, kernel_length))
    firing_rates = np.zeros((num_kernels, signal_length))
    phi = np.zeros(signal_length)

    # Because of causality, kernels should be zero before mid-point
    tmid_idx = int(kernel_length / 2)
    t_ = t_kernel[:tmid_idx]
    for k_idx in range(num_kernels):
        kernels[k_idx, tmid_idx:] = (-np.exp(-t_ / tau_1) + np.exp(-t_ / tau_2)) * kernel_amps[k_idx]
    for s_idx in range(num_kernels):
        t_idxs = np.array([np.argmin(np.abs(t_signal - spiketime)) for spiketime in spike_times[s_idx]])
        firing_rates[s_idx, t_idxs] = 1
    firing_rates /= dt*1e-3
    for k_idx in range(num_kernels):
        phi_i = np.convolve(kernels[k_idx,], firing_rates[k_idx], mode="same") * (dt * 1e-3)
        phi += phi_i
    avrg_kernel = np.average(kernels, axis=0)
    pop_firing_rate = np.average(firing_rates, axis=0)
    phi_tilde = np.convolve(avrg_kernel, pop_firing_rate, mode="same") * (dt * 1e-3) * num_kernels

    difference = phi - phi_tilde

    xlim = [t_kernel[-1], t_signal[-1]]
    plt.close("all")
    fig = plt.figure(figsize=[4, 4])
    fig.subplots_adjust(hspace=0.8, top=0.95, left=0.15, right=0.97, bottom=0.13)

    ax_k = fig.add_subplot(421, title="kernels",
                           xlabel="time (ms)", ylabel=r"$V$ (µV)")
    ax_sp = fig.add_subplot(412, ylabel="neuron id",
                            ylim=[0.5, num_kernels + 0.5], yticks=[1, 2],
                            xlim=xlim,
                            xlabel="time (ms)")
    ax_fr = fig.add_subplot(413, ylabel="population rate\n(spikes / $\Delta t$)", xlim=xlim,
                            xlabel="time (ms)")

    ax_sig = fig.add_subplot(414, xlim=xlim, xlabel="time (ms)",
                             ylabel=r"$V$ (µV)")
    kls = []
    kl_names = []
    for k_idx in range(num_kernels):
        kl, = ax_k.plot(t_kernel - t_kernel[tmid_idx], kernels[k_idx,], lw=1)
        spike_time_idxs = np.where(firing_rates[k_idx] > 0.1)[0]
        ax_sp.plot(t_signal[spike_time_idxs],
                   np.ones(len(spike_time_idxs)) + k_idx, '|', ms=7)
        kls.append(kl)
        kl_names.append(f"$k_{k_idx + 1}$")

    l, = ax_k.plot(t_kernel - t_kernel[tmid_idx], avrg_kernel, lw=1.5, c='k')
    ax_fr.plot(t_signal, pop_firing_rate * dt * 1e-3, lw=2, c='k')

    kls.append(l)
    kl_names.append(r"$\bar{k}$")

    l1, = ax_sig.plot(t_signal, phi_tilde, lw=2, c='k', ls='-')
    l2, = ax_sig.plot(t_signal, phi, c='gray')
    l3, = ax_sig.plot(t_signal, difference, c='r')

    ax_k.legend(kls, kl_names, frameon=False, loc=(1.1, 0.1))

    fig.legend([l1, l2, l3], [r"population kernel",
                              r"ground truth",
                              r"difference"],
               frameon=False, ncol=3, loc=(0.1, 0.0))
    simplify_axes(fig.axes)
    mark_subplots(fig.axes, xpos=-0.02)

    plt.savefig(join(figure_folder, "toy_kernel_illustration.pdf"))

if __name__ == '__main__':
    toy_results_folder = join(os.path.dirname(os.path.abspath(__file__)),
                         "toy_kernel_results")
    simpop_results_folder = join(os.path.dirname(os.path.abspath(__file__)),
                         "simpop_kernel_lognorm_results")
    firing_rate_folder = join(os.path.dirname(os.path.abspath(__file__)),
                         "firing_rate_results")
    data_folder = join(os.path.dirname(os.path.abspath(__file__)),
                         "simulated_pop_kernels")

    # Figure 3:
    simplest_toy_illustration(toy_results_folder)

    # Figure 4:
    run_illustrative_examples(toy_results_folder)

    # Figure 5
    run_parameter_scan(toy_results_folder)
    plot_parameter_scan_results(toy_results_folder)

    run_parameter_scan_biophysical_model(simpop_results_folder, data_folder, firing_rate_folder)
    # Figure 7:
    investigate_error_measure(data_folder, firing_rate_folder)

    # Figure 8:
    compare_errors("mip_0.0_10_0.0")

    #Figure 9:
    illustrate_firing_rates(firing_rate_folder)

    # Figure 10:
    error_with_correlation_type(data_folder)

    # Figure 11:
    error_summary_figure(data_folder)

    # Figure 12:
    rate_model_figure(data_folder)

    # Appendix Figure
    plot_all_signals_and_kernels(data_folder, firing_rate_folder)
    gather_pop_kernels(data_folder)
