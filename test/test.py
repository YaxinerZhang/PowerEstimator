import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from power_spectrum_estimator_py import Power_spectrum_estimator


def test():

    print("Test power spectrum estimator.")
    pos_file = 'test/fakedata_1.0000/1/Position/000000'
    pos = np.fromfile(pos_file, dtype=np.float32, count=-1)*0.6732117
    pos = np.reshape(pos, (32*32*32, 3))


    output_dir = Path("./.temp")
    output_dir.mkdir(exist_ok=True)
    estimator1 = Power_spectrum_estimator(32, 128, 1000)

    k_vals_dft, Pk_dft, modes_dft = estimator1.get_pk_dft(pos)

    schemes = ['NGP', 'CIC', 'TSC']

    for scheme in schemes:

        k_vals1, Pk1, modes1  = estimator1.get_pk_fft(pos, assigner= scheme) #without deconvolving window function, without interlacing
        k_vals2, Pk2, modes2  = estimator1.get_pk_fft(pos, assigner= scheme, compensation = "deconvolving")  #window function deconvolved, without interlacing
        k_vals3, Pk3, modes3  = estimator1.get_pk_fft(pos, assigner= scheme, anti_aliasing = "interlacing", compensation = "deconvolving")  #window function deconvolved, leading-order interlacing
        
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"Power Spectrum with {scheme} Assignment", fontsize=14)

        
        ax1 = plt.subplot(121)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel(r'$k\ [h\ \mathrm{Mpc}^{-1}]$', fontsize=12)
        ax1.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$', fontsize=12)
        ax1.set_xlim(np.min(k_vals1), 0.2)
        ax1.set_ylim(4e2, 1e4)
        
        ax2 = plt.subplot(122)
        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.set_xlabel(r'$k\ [h\ \mathrm{Mpc}^{-1}]$', fontsize=12)
        ax2.set_ylabel(r'ratio', fontsize=12)
        ax2.set_ylim(0.92, 1.10)
        ax2.set_xlim(np.min(k_vals1), 0.2)
        

        ax1.plot(k_vals_dft, Pk_dft, label = r'DFT', color = 'r')
        ax1.plot(k_vals1, Pk1, label = f'{scheme}128, not interlaced, not deconvolved', color = 'blue')
        ax1.plot(k_vals2, Pk2, label = f'{scheme}128, not interlaced, deconvolved', color = 'orange')
        ax1.plot(k_vals3, Pk3, label = f'{scheme}128, interlaced, deconvolved', color = 'green')
        ax1.legend()

        ax2.plot(k_vals_dft, np.size(k_vals_dft)*[1.0], label = r'DFT', color = 'r', linewidth = 3)
        ax2.plot(k_vals1, Pk1/Pk_dft, label = f'{scheme}128, not interlaced, not deconvolved',color = 'blue')
        ax2.plot(k_vals2, Pk2/Pk_dft, label = f'{scheme}128, not interlaced, deconvolved', color = 'orange')
        ax2.plot(k_vals3, Pk3/Pk_dft, label = f'{scheme}128, interlaced, deconvolved', color = 'green')
        error_percent = 2.5
        ax2.fill_between(k_vals1, 
                    1 - error_percent/100, 
                    1 + error_percent/100,
                    color='gray', alpha=0.3)
        ax2.legend()


        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=15, direction='in', width=1.2, length = 10.0)
            ax.tick_params(axis='y', which='minor', direction='in', length = 5.0)
            ax.tick_params(axis='x', top='on', which='minor', direction='in', length = 5.0)
            ax.tick_params(axis='x', top='on', direction='in', length = 10.0)
            ax.tick_params(axis='x', which='minor', direction='in', length = 5.0)
            ax.tick_params(axis='y', right='on', which='minor', direction='in', length = 5.0)
            ax.tick_params(axis='y', right='on', direction='in', length = 10.0)

        
        plt.tight_layout()
        
        fig.savefig(output_dir / f"pk_{scheme}.png")
        
        data = np.column_stack((k_vals3, Pk3, modes3))
        header = (
            f"Power Spectrum ({scheme} assignment)\n"
            "Columns: k [h/Mpc], P(k) [(Mpc/h)^3], modes\n"
            f"BoxSize={estimator1.L} Mpc/h, Ngrid={estimator1.Ngrid}"
        )
        np.savetxt(
            output_dir / f"pk_data_{scheme}.txt",
            data,
            fmt=['%.5e', '%.5e', '%d'],
            delimiter='    ',
            header=header
        )

    print("All tests completed!")
