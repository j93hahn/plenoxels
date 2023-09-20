import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_expon_lr_func(
    lr_delay_steps, lr_init, lr_final, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper


mapping = {
    '0.3': '3e-1',
    '0.003': '3e-3',
    '30.0': '3e1',
    '0.0005': '5e-4',
    '0.05': '5e-2',
    '5e-06': '5e-6'
}


color = {
    '30.0': 'red',
    '0.3': 'blue',
    '0.003': 'green',
    '0': '-',
    '15000': '-.'
}


plt.rcParams["font.family"] = "STIXGeneral"


def compare_all_lr_schedulers():
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
    plt.savefig('lr_all.png', dpi=300)


_ALL_SIGMA_LR_SCHEDULERS = [
    # lr_decay_steps, lr_init, lr_final
    [0, 3e1, 5e-2],
    [0, 3e-1, 5e-4],
    [0, 3e-3, 5e-6],
    [15000, 3e1, 5e-2],
    [15000, 3e-1, 5e-4],
    [15000, 3e-3, 5e-6]
]


def compare_sigma_lr_schedulers():
    fig, ax = plt.subplots()
    iters = 128000
    for params in _ALL_SIGMA_LR_SCHEDULERS:
        func = get_expon_lr_func(*params, 1e-2, 250000)
        sigmas = np.zeros(iters)
        xs = np.arange(iters)
        for i in range(iters):
            sigmas[i] = func(i)
        ax.plot(xs, sigmas, label=f'params: [{str(params[0])}, {mapping[str(params[1])]}, {mapping[str(params[2])]}]', linewidth=1.0, linestyle=color[str(params[0])], color=color[str(params[1])])

    ax.tick_params(direction='in', axis='both', width=0.5, color='black', length=4, labelsize=10.0)
    ax.yaxis.grid(True, which='both', linewidth=0.5, color='lightgray')

    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=2, prop={'size': 6})

    ax.text(3000, 3, "plenoxels default", fontsize=7.0)

    ax.set_yscale('log')
    ax.minorticks_off()
    ax.set_xlabel('iteration')
    ax.set_ylabel('σ learning rate value (log scale)')
    plt.savefig('lr_sigma.png', dpi=300)


if __name__ == '__main__':
    # compare_all_lr_schedulers()
    compare_sigma_lr_schedulers()
