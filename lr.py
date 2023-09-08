from opt.util.util import get_expon_lr_func
import numpy as np


if __name__ == '__main__':
    # use default values from opt/opt.py
    lr_sigma_func = get_expon_lr_func(3e1, 5e-2, 15000, 1e-2, 250000)
    lr_sh_func = get_expon_lr_func(1e-2, 5e-6, 0, 1e-2, 250000)
    lr_basis_func = get_expon_lr_func(1e-6, 1e-6, 0, 1e-2, 250000)
    lr_sigma_bg_func = get_expon_lr_func(3e0, 3e-3, 0, 1e-2, 250000)
    lr_color_bg_func = get_expon_lr_func(1e-1, 5e-6, 0, 1e-2, 250000)

    iters = 128000
    lr_sigmas = np.zeros(iters)
    lr_shs = np.zeros(iters)
    lr_bases = np.zeros(iters)
    lr_sigma_bgs = np.zeros(iters)
    lr_color_bgs = np.zeros(iters)
    xs = np.arange(iters)

    for i in range(iters):
        lr_sigmas[i] = lr_sigma_func(i)
        lr_shs[i] = lr_sh_func(i)
        lr_bases[i] = lr_basis_func(i)
        lr_sigma_bgs[i] = lr_sigma_bg_func(i)
        lr_color_bgs[i] = lr_color_bg_func(i)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, lr_sigmas, label='sigma')
    ax.plot(xs, lr_shs, label='sh')
    ax.plot(xs, lr_bases, label='basis')
    ax.plot(xs, lr_sigma_bgs, label='sigma_bg')
    ax.plot(xs, lr_color_bgs, label='color_bg')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    ax.set_yscale('log')
    ax.set_title('Learning rate functions for plenoxels')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning rate value')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig('lr.png', dpi=300)
