from QI import *

Nth = 0.1     # Average thermal photon numbers
N = 10        # Truncated photon numbers, i.e., 0 ~ N-1
eta = 0.01    # Transmissivity


def find_qcb_opt(num_divide, Ns, save_filename):
#    lmd = np.sqrt(Ns/(1 + Ns))
    s = np.arcsinh(np.sqrt(Ns))    # Squeezed parameter

    # meshgrid
    RA = np.linspace(0.0, 1.0, num_divide)
    RB = np.linspace(0.0, 1.0, num_divide)
    RA, RB = np.meshgrid(RA, RB)
    qcb = []

    cts = 0
    total = num_divide**2

    for ra, rb in zip(np.ravel(RA), np.ravel(RB)):
        rt_list = (np.sqrt(1-ra**2), ra, np.sqrt(1-rb**2), rb)
        rho_0 = RHO_0(PCS, N, s, Nth, rt_list)
        rho_1 = RHO_1(PCS, N, s, Nth, eta, rt_list)
        tr_sqrt = QCB(rho_0, rho_1, approx=True)
        qcb_pcs.append(tr_sqrt)

        cts += 1
        print("\% %.2f" % cts/total)

    qcb_pcs = np.array(qcb_pcs)
    qcb_pcs = qcb_pcs.reshape(RA.shape)
    np.savez('pcs_opt_1e-1', RA=RA, TB=TB, qcb_pcs=qcb_pcs)


