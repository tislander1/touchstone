# touchstone
Touchstone manipulations
- def read_touchstone(self, filename):
- - #returns freq, S
- def downsample_sparam(self, freq_list, sparam, downsample_ratio=2)
- def construct_sparam(self, list_of_sparam_vectors):
        #make a single list of N-port s-parameters vectors
        #example: sk = spt.construct_sparam([s11, s21, s21, s11])
- def get_vector_sparam(self, sparam, port_pair):
      #get the required sparameter in sparam
      #port_pair_list = [(1,3), (2,4)] would get s13 and s24
- def vector_to_dB(self, vector):
- def vector_to_mag(self, vector):
- def vector_to_deg(self, vector):
- def vector_to_rad(self, vector):
- def plot_sparam_rectangular(self, freq_list, sparam_list, port_pair_list, style='dB'):
        #styles: 'mag', 'dB', 'phase'
- def save_touchstone(self, freq, sparam, filename):
        #outputs 50 ohm s-parameters in mag-angle format
- def linear_interp_sparam(self, sparam, current_freq_vector, desired_freq_step, desired_init_freq = -1, desired_end_freq = -1):
        #resamples s-parameters to the desired spacing.
        #apply with caution -- this just does linear interpolation, which can amplify noise.
- def renumber_ports_with_list(self, sparam, left_ports_list, right_ports_list):
        #new renumbered order is left_port[0], right_port_[0], left_port_[1], right_port_[1], left_port[2], right_port_[2] ...
        #the number of ports can be equal to or less than the number in the original s-parameter set
- def _two_port_S2T(self, s):
        #convert a two-port s-parameter to a t-parameter
- def _two_port_T2S(self, t):
      #convert a two-port t-parameter to an s-parameter
- def _two_port_S2ABCD(self, s, Z0=50):
      #convert a two-port t-parameter to a z-parameter
      # Ludwig & Bretchko, Appendix D
- def _two_port_ABCD2S(self, A, Z0=50):
      #convert a two-port t-parameter to a z-parameter
      # Ludwig & Bretchko, Appendix D
- def _four_port_single2diff(self, s):
        #Note, not properly tested!
        #input should be in the format
        # port 1 ------ port2
        # port 3 ------ port4
        #output will be in the format
        #diff pair 1 ========= diff pair 2
        #Example call:      sdd, scd, sdc, scc = _four_port_single2diff(sparam4)
        #reference https://blog.lamsimenterprises.com/2020/07/24/single-ended-to-mixed-mode-conversions/
- def single_to_balanced_sparams(self, sparam):
        #Note, not fully tested
        #Convert single ended sparams to balanced mode.
        #Example call:       sdd, sdc, scd, scc = single_to_balanced_sparams(sparam)
        #The port assignment for the input should be:
        #       Port 1 ------- Port 2
        #       Port 3 ------- Port 4
        #       Port 5 ------- Port 6
        #       Port 7 ------- Port 8
        #       Etc.
        #The port assignment for the output will be:
        #       Diff port 1 ====== Diff port 2
        #       Diff port 3 ====== Diff port 4
        #       Etc.
        #Each of the four output matrices (sdd, sdc, scd, scc) will have half the number of ports as the input.
- def _convert_nport_from_convert_twoport(self, param, func):
- def convert_S_to_T(self, sparam):
- def convert_T_to_S(self, tparam):
- def convert_S_to_ABCD(self, sparam):
- def convert_ABCD_to_S(self, abcd_param):
- def multiply_params(self, param1, param2):
        #require parameters to be the same length
- def concatenate_tparams(self, tparam1, tparam2):
- def concatenate_sparams(self, sparam1, sparam2):
- def get_connection_list(self, sparam, cutoff, index1 = 10):
        #outputs a sorted list of all port pairs for which the sparameter magnitude is greater than cutoff at index1.
- def get_passivity_vector(self, sparam):

  
