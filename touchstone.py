#David Kopp, October 17, 2021.  Version 0.12

import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime

class SParamTools(object):
    def __init__(self):
        self.info_string = 'SParamTools, by David Kopp.  Last updated to version 0.12 on October 17, 2021.'
    def _is_square(self, apositiveint):
        x = apositiveint // 2
        seen = set([x])
        while x * x != apositiveint:
            x = (x + (apositiveint // x)) // 2
            if x in seen:
                return False
            seen.add(x)
        return True
    def read_touchstone(self, filename):
        #returns freq, S
        with open(filename) as f:
            lines = f.read().splitlines()
            
        p = re.compile(r'\.s(\d+)p')    #get number of ports from filename.  Match the item in parentheses.
        port_count = int(p.findall(filename)[0])

        freq =[]
        data =[]
        data_at_this_freq = []

        freq_multiplier = 1
        ix = -1
        for line in lines:
            ix = ix + 1
            if '#' in line:
                #this character indicates where the frequency scaling (hz/khz/MHz/GHz) and data format (RI/MA/dB)are declared
                print(line)
                p = re.compile(r'\s*#\s(\w+)\s*\w+\s*(\w+)') #matches frequency-scaling and data-format
                touchstone_mode = list(p.findall(line)[0])  #list with frequency-scaling and data-format
                
                for ix in range(0, len(touchstone_mode)): #convert to lowercase
                    touchstone_mode[ix] = str(touchstone_mode[ix]).lower()
                print(touchstone_mode)

                if 'hz' in touchstone_mode[0]:
                    freq_multiplier = 1
                if 'khz' in touchstone_mode[0]:
                    freq_multiplier = 1e3
                if 'mhz' in touchstone_mode[0]:
                    freq_multiplier = 1e6                
                if 'ghz' in touchstone_mode[0]:
                    freq_multiplier = 1e9
                continue
            if '!' in line: #match comment lines
                continue
            if not (line and line.strip()): #match blank lines
                continue
            else:
                p = re.compile(r'^\d*\.?\d*')   #the frequency will be a decimal at the beginning of the line (marked by a caret).  Match that
                f = p.findall(line)
                if f and not(f == ['']):        #if there are nontrivial matches
                    frequency = freq_multiplier * float(f[0])
                    freq.append(frequency)  #add the frequency to the list
                    data.append(data_at_this_freq)      #add data to the list
                    data_at_this_freq = line.split()    #splits data on whitespace
                else:
                    data_at_this_freq = data_at_this_freq + line.split()
        data.append(data_at_this_freq)  #covers last datapoint

        #at this point, all of the data should be in a list with the first element being the frequency, and then the next 2 data are complex


        if 'ma' in touchstone_mode[1]:  #mag-angle format
            S =[]
            for row in data:
                if not row: #exception handling
                    continue
                srow = []
                for ix in range(1,len(row),2):
                    mag = float(row[ix])
                    angle = (np.pi/180)* float(row[ix+1])
                    c = mag*np.exp(1j*angle)
                    srow.append(c)
                mat1 = np.array(srow).reshape(port_count,port_count)
                #print('len:',len(srow))  
                
                S.append(mat1)

        if 'ri' in touchstone_mode[1]:  #real-imag format --- not tested yet!
            S =[]
            for row in data:
                if not row: #exception handling
                    continue
                srow = []
                for ix in range(1,len(row),2):
                    c = float(row[ix]) + 1j*float(row[ix+1])
                    srow.append(c)
                mat1 = np.array(srow).reshape(12,12)
                #print('len:',len(srow))  
                
                S.append(mat1)
        return freq, S

    def downsample_sparam(self, freq_list, sparam, downsample_ratio=2):
        f_out = []
        s_out = []
        for ix in range(0,len(freq_list)):
            if ix % downsample_ratio == 0:
                f_out.append(freq_list[ix])
                s_out.append(sparam[ix])
        return f_out, s_out

    def construct_sparam(self, list_of_sparam_vectors):
        #make a single list of N-port s-parameters vectors
        #example: sk = spt.construct_sparam([s11, s21, s21, s11])
        if not self._is_square(len(list_of_sparam_vectors)):
            print('The number of s-parameter vectors should be square.  Please try again.')
            return 0, 0
        num_ports = int(np.sqrt(len(list_of_sparam_vectors)))
        print('Constructing scattering parameters with '+str(num_ports)+' port(s).')
        s_out = []
        for ix in range(0,len(list_of_sparam_vectors[0])):
            s_current = np.zeros((num_ports, num_ports), dtype=np.cdouble)
            #print(s_current)
            for rownum in range(0, num_ports):
                for colnum in range(0, num_ports):
                    s_current[rownum][colnum] = list_of_sparam_vectors[rownum*num_ports + colnum][ix]
            s_out.append(s_current)
        return s_out

    def get_vector_sparam(self, sparam, port_pair):
        #get the required sparameter in sparam
        #port_pair_list = [(1,3), (2,4)] would plot s13 and s24
        s_vector = []
        for s in sparam:
            s_vector.append(s[port_pair[0]][port_pair[1]])
        return s_vector

    def vector_to_dB(self, vector):
        vector_out =[]
        for s in vector:
            vector_out.append(20*np.log10(np.absolute(s)))
        return vector_out

    def vector_to_mag(self, vector):
        vector_out =[]
        for s in vector:
            vector_out.append(np.absolute(s))
        return vector_out

    def vector_to_deg(self, vector):
        vector_out =[]
        scale_factor = 180/np.pi
        for s in vector:
            vector_out.append(scale_factor*np.angle(s))
        return vector_out

    def vector_to_rad(self, vector):
        vector_out =[]
        scale_factor = 1
        for s in vector:
            vector_out.append(scale_factor*np.angle(s))
        return vector_out

    def plot_sparam_rectangular(self, freq_list, sparam_list, port_pair_list, style='dB'):
        #styles: 'mag', 'dB', 'phase'
        fig1, ax1 = plt.subplots()
        ix=-1
        for sparam in sparam_list:
            ix = ix + 1
            for port_pair in port_pair_list:
                if style == 'dB':
                    s = self.vector_to_dB(self.get_vector_sparam(sparam, port_pair))
                    ax1.plot(freq_list[ix], s)
                    ax1.set_ylabel('Sparameter (dB)')
                if style == 'mag':
                    s = self.vector_to_mag(self.get_vector_sparam(sparam, port_pair))
                    ax1.plot(freq_list[ix], s)
                    ax1.set_ylabel('Sparameter (mag)')
                if style == 'deg':
                    s = self.vector_to_deg(self.get_vector_sparam(sparam, port_pair))
                    ax1.plot(freq_list[ix], s)
                    ax1.set_ylabel('Sparameter (deg)')
                if style == 'rad':
                    s = self.vector_to_rad(self.get_vector_sparam(sparam, port_pair))
                    ax1.plot(freq_list[ix], s)
                    ax1.set_ylabel('Sparameter (rad)')
                #print('hello', sparam[port_pair[0]][port_pair[1]])
        ax1.set_xlabel('Frequency (Hz)')
        plt.show()

    def save_touchstone(self, freq, sparam, filename):
        #outputs 50 ohm s-parameters in mag-angle format
        num_ports = np.shape(sparam)[1]
        if np.shape(sparam)[1] != np.shape(sparam)[2]:
            print('Error. The s-parameters are not square and cannot be processed.')
            return False
        if not len(freq) == len(sparam):
            print('Error.  The length of freq and sparam vectors should be the same.')
            return False
        filetype = '.s'+str(int(num_ports))+'p'
        with open(filename+filetype, 'w') as file:
            rad2deg = 180/np.pi
            file.write('! Touchstone file exported from '+str(self.info_string)+'\n')
            file.write('! Generated: '+str(datetime.now())+'\n')
            file.write('!\n')
            file.write('# GHz S MA R 50.000000\n')
            file.write('!\n')
            print('lw', len(sparam[0]))
            for ix in range(0,len(freq)):
                file.write('\n'+str(freq[ix]/1e9)+'             ')

                counter = 0
                firstloop = True
                for p in sparam[ix]:
                    for sparampoint in p:
                        counter = counter + 1
                        if counter % 4 == 1 and not firstloop:
                            file.write('\n                     ')
                        firstloop = False
                        mag = str(np.absolute(sparampoint))
                        angle = str(rad2deg*np.angle(sparampoint))
                        file.write(' '+mag+' '+angle)
                        
    def linear_interp_sparam(self, sparam, current_freq_vector, desired_freq_step, desired_init_freq = -1, desired_end_freq = -1):
        #resamples s-parameters to the desired spacing.
        #apply with caution -- this just does linear interpolation, which can amplify noise.
        if desired_end_freq == -1:
            desired_end_freq = np.max(current_freq_vector)
        if desired_init_freq == -1:
            desired_init_freq = np.min(current_freq_vector)
        num_ports = np.shape(sparam)[1]
        print('num_ports', num_ports)
        f = np.arange(desired_init_freq, desired_end_freq, desired_freq_step)
        print(f)
        sparam_list = []
        for ix in range(0, num_ports):
            for iy in range(0, num_ports):
                sp = self.get_vector_sparam(sparam, (ix, iy))        #get a single s-param
                sparam_list.append(np.interp(f, current_freq_vector, sp))    #linear interpolation to desired frequency spacing, then append to the list of all sparams
        print("sp100: ",sparam_list[100])
        return f, self.construct_sparam(sparam_list) #make an s-parameter structure

    def renumber_ports_with_list(self, sparam, left_ports_list, right_ports_list):
        #new renumbered order is left_port[0], right_port_[0], left_port_[1], right_port_[1], left_port[2], right_port_[2] ...
        #the number of ports can be equal to or less than the number in the original s-parameter set

        new_ports_list = []
        for ix in range(0, len(left_ports_list)):
            new_ports_list.append(left_ports_list[ix])
            new_ports_list.append(right_ports_list[ix])
        sparam_out = []
        old_num_ports = np.shape(sparam)[1]
        new_num_ports = len(new_ports_list)
        if new_num_ports != old_num_ports:
            print(str(old_num_ports)+' port s-parameters will now be modified to '+str(new_num_ports)+' ports')
        else:
            print('Renumbering the '+str(new_num_ports)+ ' port parameter.')
            
        for s in sparam:
            sparam_out_current = np.zeros((new_num_ports, new_num_ports), dtype=np.cdouble)
            for ix in range(0, new_num_ports):
                for iy in range(0, new_num_ports):
                    sparam_out_current[ix][iy] = s[new_ports_list[ix]][new_ports_list[iy]]
            sparam_out.append(sparam_out_current)
        return sparam_out
         

    def _two_port_S2T(self, s):
        #convert a two-port s-parameter to a t-parameter
        s11 = s[0][0];  s12=s[0][1]; s21=s[1][0]; s22=s[1][1];
        t11 = -(s11*s22-s12*s21)/s21
        t12 = s11/s21 
        t21 = -s22/s21 
        t22 = 1/s21 
        return np.array([[t11, t12],[t21,t22]])  #This works fine for t-params in addition to s-params

    def _two_port_T2S(self, t):
        #convert a two-port t-parameter to an s-parameter
        t11 = t[0][0];  t12=t[0][1]; t21=t[1][0]; t22=t[1][1];
        s11 = t12/t22 
        s12 = (t11*t22-t12*t21)/t22 
        s21 =  1/t22 
        s22 = -t21/t22 
        return np.array([[s11, s12],[s21,s22]]) 

    def _two_port_S2ABCD(self, s, Z0=50):
        #convert a two-port t-parameter to a z-parameter
        # Ludwig & Bretchko, Appendix D
        s11 = s[0][0];  s12=s[0][1]; s21=s[1][0]; s22=s[1][1];

        psiAinv = (1/(2*s21))
        a = psiAinv*((1+s11)*(1-s22)+s12*s21)
        b = psiAinv*((1+s11)*(1+s22)-s12*s21)*Z0
        c = psiAinv*((1-s11)*(1-s22)-s12*s21)/Z0
        d = psiAinv*((1-s11)*(1+s22)+s12*s21)

        return np.array([[a, b],[c,d]])

    def _two_port_ABCD2S(self, A, Z0=50):
        #convert a two-port t-parameter to a z-parameter
        # Ludwig & Bretchko, Appendix D
        a = A[0][0];  b=A[0][1]; c=A[1][0]; d=A[1][1];
       
        psi7 = a + b/Z0 + c*Z0 + d
        s11 = (a + b/Z0 -c*Z0 -d)/psi7
        s12 = 2*(a*d-b*c)/psi7
        s21 = 2/psi7
        s22 = (-a + b/Z0 - c*Z0 + d)/psi7
        
        return np.array([[s11, s12],[s21,s22]])

    def _four_port_single2diff(self, s):
        #Note, not properly tested!
        #
        #input should be in the format
        # port 1 ------ port2
        # port 3 ------ port4

        #output will be in the format
        #diff pair 1 ========= diff pair 2

        #Example call:      sdd, scd, sdc, scc = _four_port_single2diff(sparam4)

        #reference https://blog.lamsimenterprises.com/2020/07/24/single-ended-to-mixed-mode-conversions/

        s11 = s[0][0]; s12=s[0][1]; s13=s[0][2]; s14=s[0][3];
        s21 = s[1][0]; s22=s[1][1]; s23=s[1][2]; s24=s[1][3];
        s31 = s[2][0]; s32=s[2][1]; s33=s[2][2]; s34=s[2][3];
        s41 = s[3][0]; s42=s[3][1]; s43=s[3][2]; s44=s[3][3];
        
        #differential to differential
        #Note, sdd21 means differential out of port 2, and differential into port 1
        sdd11 = 0.5*(s11-s13-s31+s33)
        sdd12 = 0.5*(s12-s14-s32+s34)
        sdd21 = 0.5*(s21-s23-s41+s43)
        sdd22 = 0.5*(s22-s24-s42+s44)

        #common to common
        #Note, sdd21 means common out of port 2, and common into port 1
        scc11 = 0.5*(s11+s13+s31+s33)
        scc12 = 0.5*(s12+s14+s32+s34)
        scc21 = 0.5*(s21+s23+s41+s43)
        scc22 = 0.5*(s22+s24+s42+s44)

        #differential to common
        #Note, scd21 means common out of port 2, and differential into port 1
        scd11 = 0.5*(s11-s13+s31-s33)
        scd12 = 0.5*(s12-s14+s32-s34)
        scd21 = 0.5*(s21-s23+s41-s43)
        scd22 = 0.5*(s22-s24+s42-s44)

        #common to differential
        #Note, sdc21 means differential out of port 2, and common into port 1
        sdc11 = 0.5*(s11+s13-s31-s33)
        sdc12 = 0.5*(s12+s14-s32-s34)
        sdc21 = 0.5*(s21+s23-s41-s43)
        sdc22 = 0.5*(s22+s24-s42-s44)

        sdd = np.array([[sdd11,sdd12],[sdd21,sdd22]])
        scc = np.array([[scc11,scc12],[scc21,scc22]])
        scd = np.array([[scd11,scd12],[scd21,scd22]])
        sdc = np.array([[sdc11,sdc12],[sdc21,sdc22]])

        return sdd, scd, sdc, scc
        

    def single_to_balanced_sparams(self, sparam):
        #Note, not fully tested
        #
        #Convert single ended sparams to balanced mode.
        #Example call:       sdd, sdc, scd, scc = single_to_balanced_sparams(sparam)
        #The port assignment for the input should be:
        #       Port 1 ------- Port 2
        #       Port 3 ------- Port 4
        #       Port 5 ------- Port 6
        #       Port 7 ------- Port 8
        #       Etc.
        #
        #The port assignment for the output will be:
        #       Diff port 1 ====== Diff port 2
        #       Diff port 3 ====== Diff port 4
        #       Etc.
        #Each of the four output matrices (sdd, sdc, scd, scc) will have half the number of ports as the input.
        
        num_SE_ports = np.shape(sparam)[1]
        if (num_SE_ports == 0) or (num_SE_ports % 4 != 0):
            print('Error, balanced s-parameters require a multiple of 4 single ended ports.')
            return -1

        halfNumPorts = int(num_SE_ports/2)
        
        sdd_output = []
        scd_output = []
        sdc_output = []
        scc_output = []

        for s in sparam:
            sparam_dd_current = np.zeros((halfNumPorts, halfNumPorts), dtype=np.cdouble)
            sparam_cd_current = np.zeros((halfNumPorts, halfNumPorts), dtype=np.cdouble)
            sparam_dc_current = np.zeros((halfNumPorts, halfNumPorts), dtype=np.cdouble)
            sparam_cc_current = np.zeros((halfNumPorts, halfNumPorts), dtype=np.cdouble)
            for ix in range(0, num_SE_ports, 4):
                for iy in range(0, num_SE_ports, 4):
                    four_port_input_param = s[ix:ix+4, iy:iy+4]
                    sdd, scd, sdc, scc = self._four_port_single2diff(four_port_input_param)
                    h_ix = int(ix/2);  h_iy = int(iy/2);    #output is only a 2x2, for each 4x4.  h_ix needs to be half of ix to properly index the output
                    sparam_dd_current[h_ix:h_ix+2, h_iy:h_iy+2] = sdd
                    sparam_cd_current[h_ix:h_ix+2, h_iy:h_iy+2] = scd
                    sparam_dc_current[h_ix:h_ix+2, h_iy:h_iy+2] = sdc
                    sparam_cc_current[h_ix:h_ix+2, h_iy:h_iy+2] = scc
            sdd_output.append(sparam_dd_current)
            scd_output.append(sparam_cd_current)
            sdc_output.append(sparam_dc_current)   
            scc_output.append(sparam_cc_current)

        return sdd_output, scd_output, sdc_output, scc_output 


    def _convert_nport_from_convert_twoport(self, param, func):
        
        num_ports = np.shape(param)[1]
        if num_ports % 2 != 0:
            print('Warning, the input does not have an even number of ports.  Results may be undefined.')
        nport_out = []
        for p in param:
            param_current = np.zeros((num_ports, num_ports), dtype=np.cdouble)
            for ix in range(0, num_ports, 2):
                for iy in range(0, num_ports, 2):
                    two_port_input_param = p[ix:ix+2, iy:iy+2]
                    two_port_output_param = func(two_port_input_param)
                    param_current[ix:ix+2, iy:iy+2] = two_port_output_param
            nport_out.append(param_current)
        return nport_out
        
    def convert_S_to_T(self, sparam):
        return self._convert_nport_from_convert_twoport(sparam, self._two_port_S2T)

    def convert_T_to_S(self, tparam):
        return self._convert_nport_from_convert_twoport(tparam, self._two_port_T2S)

    def convert_S_to_ABCD(self, sparam):
        return self._convert_nport_from_convert_twoport(sparam, self._two_port_S2ABCD)

    def convert_ABCD_to_S(self, abcd_param):
        return self._convert_nport_from_convert_twoport(abcd_param, self._two_port_ABCD2S)

    def multiply_params(self, param1, param2):
        #require parameters to be the same length
        if len(param1) != len(param2):
            print( 'Error in multiply, parameters are not the same length!' )
            return -1
        product = []
        for ix in range(0, len(param1)):
            product.append(np.matmul(param1[ix], param2[ix]))
        return product

    def concatenate_tparams(self, tparam1, tparam2):
        return self.multiply_params(tparam1, tparam2)

    def concatenate_sparams(self, sparam1, sparam2):
        concat_tparam = self.concatenate_tparams( self.convert_S_to_T(sparam1), self.convert_S_to_T(sparam2)) #convert s-params to t-parameters, then concatenate the t-parameters
        return self.convert_T_to_S(concat_tparam)

    def get_connection_list(self, sparam, cutoff, index1 = 10):
        #outputs a sorted list of all port pairs for which the sparameter magnitude is greater than cutoff at index1.
        num_ports = np.shape(sparam)[1]
        port_list = []
        out_list = []
        
        for ix in range(0, num_ports):
            for iy in range(0, num_ports):
                mag = np.absolute(sparam[index1][ix][iy])
                if mag>=cutoff and (iy, ix) not in port_list :
                    if ix <= iy:
                        port_list.append((ix,iy))
                    else:
                        port_list.append((iy, ix))
                out_list.append(((ix, iy), mag))
        out_list.sort(key=lambda y: y[1], reverse=True)
        port_list.sort()
        separated_port_list = [[],[]]
        for item in port_list:
            separated_port_list[0].append(item[0])
            separated_port_list[1].append(item[1])
        return port_list, separated_port_list, out_list

    def get_passivity_vector(self, sparam):
        passivity_vector = []
        for item in sparam:
            passivity_vector.append(max(np.linalg.svd(item, compute_uv=False)))
        return passivity_vector        
                    

                
            
#------------------------------------------------------------------------------------------------------------------------------------
#
# Begin testing the class
#
#------------------------------------------------------------------------------------------------------------------------------------

spt = SParamTools()
f, s  = spt.read_touchstone('ViaOnly_Main.s12p')

f1, s1 = spt.downsample_sparam(f, s, 50)

print('freq:', f[100])
print('s22:', s[100][2-1][2-1])



s11=list(np.arange(0,10)+1j)
s21=list(np.arange(10,0,-1)+1j)


sk = spt.construct_sparam([s11, s21, s21, s11])

s11b = [11.7, 11.8, 11.9]
s21b = [21.7, 21.8, 21.9]
s31b = [31.7, 31.8, 31.9]
s41b = [41.7, 41.8, 41.9]

s12b = [12.7, 12.8, 12.9]
s22b = [22.7, 22.8, 22.9]
s32b = [32.7, 32.8, 32.9]
s42b = [42.7, 42.8, 42.9]

s13b = [13.7, 13.8, 13.9]
s23b = [23.7, 23.8, 23.9]
s33b = [33.7, 33.8, 33.9]
s43b = [43.7, 43.8, 43.9]

s14b = [14.7, 14.8, 14.9]
s24b = [24.7, 24.8, 24.9]
s34b = [34.7, 34.8, 34.9]
s44b = [44.7, 44.8, 44.9]

ftest = [1e9, 2e9, 3e9]
stest = spt.construct_sparam([s11b, s12b, s13b, s14b, s21b, s22b, s23b, s24b, s31b, s32b, s33b, s34b, s41b, s42b, s43b, s44b])

fk = range(0,1000000000*len(sk),1000000000)
print(sk)

sv = spt.get_vector_sparam(sk, (2-1,1-1))

spt.plot_sparam_rectangular([fk,f],[sk,s],[(1,1),(0,1)], 'dB')

spt.save_touchstone(f1, s1, 'myfile.txt')

print(spt.info_string)

f2, s2 = spt.linear_interp_sparam(sparam=s1, current_freq_vector=f1, desired_freq_step=20e6)

spt.save_touchstone(f2, s2, 'resaved')

t3 = spt.convert_S_to_T(s1)
s3 = spt.convert_T_to_S(t3)

abcd4=spt.convert_S_to_ABCD(s1)
s4=spt.convert_ABCD_to_S(abcd4)

s5 = spt.renumber_ports_with_list(stest, left_ports_list=[0, 2], right_ports_list=[1, 3])

s6 = spt.concatenate_sparams(s3, s3)

connected_port_list, separated_port_list, mag_list = spt.get_connection_list(s1, cutoff=0.5, index1=10)
print('Connections: ', connected_port_list)

s7 = spt.renumber_ports_with_list(s1, left_ports_list=separated_port_list[0], right_ports_list=separated_port_list[1])

connected_port_list2, separated_port_list2, mag_list2 = spt.get_connection_list(s7, cutoff=0.5, index1=10)
print('Connections: ', connected_port_list2)

#sdd7, scd7, sdc7, scc7 = spt._four_port_single2diff(s7[10])

sdd7, scd7, sdc7, scc7 = spt.single_to_balanced_sparams(s7)
sdd8, scd8, sdc8, scc8 = spt.single_to_balanced_sparams(stest)
print('sdd8',sdd8)
print('scc8',scc8)

passivity1 = spt.get_passivity_vector(s1)
plt.plot(f1, passivity1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Passivity (max eigenvalue)")
plt.show()

