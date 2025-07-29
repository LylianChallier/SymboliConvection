"""
Vizualisation utilities.

This module contains functions to:
- 
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

def plot_metrics(iters, losses_total, sindy_iters_complet, steps_list, num_pb, folder='FIGURES'):
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(sindy_iters_complet, np.sqrt(losses_total['mse_loss_train']), label='train')
    ax0.plot(iters, np.sqrt(losses_total['mse_loss_test']), label='test')
    ax0.legend()
    ax0.set_title(r'RMSE')
    ax0.set_xlabel('iterations')
    ax0.set_xticks(np.cumsum(steps_list))

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(sindy_iters_complet, losses_total['l1_loss_train'], label='train')
    ax1.plot(iters, losses_total['l1_loss_test'], label='test')
    ax1.legend()
    ax1.set_title(r'MAE')
    ax1.set_xlabel('iterations')
    ax1.set_xticks(np.cumsum(steps_list))

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(sindy_iters_complet, losses_total['r2_score_train'], label='train')
    ax2.plot(iters, losses_total['r2_score_test'], label='test')
    ax2.set_title(r'$R^2$')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_xlabel('iterations')
    ax2.set_xticks(np.cumsum(steps_list))

    ax3 = fig.add_subplot(gs[3])
    ax3.plot(np.cumsum(steps_list), kan_compl, label='complexity', marker='o')
    ax3.set_xticks(np.cumsum(steps_list))
    ax3.set_title(r'Complexité')
    ax3.set_xlabel('iterations')

    fig.tight_layout()
    plt.savefig(f'{folder}/pb{num_pb}_metrics_kan.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{folder}/pb{num_pb}_metrics_kan.pdf', bbox_inches='tight', transparent=True, backend='pdf')
    plt.show()

def plot_results(t, z, y, pred_sindy, train_id, num_pb, folder='FIGURES'):
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)

    res_norm = mcolors.Normalize(vmin=0.0, vmax=0.2)
    norm = mcolors.TwoSlopeNorm(vmin=-20.0, vcenter=0.0, vmax=20.0)

    im1 = ax1.pcolor(t, z, y[:, :, 0].T, cmap='seismic', norm=norm)
    ax1.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(r"$\partial_z u$ réel")
    ax1.set_ylabel('$z$')
    ax1.set_xlabel('$t$')

    im2 = ax2.pcolor(t, z, pred_sindy[:, :, 0].T, cmap='seismic', norm=norm)
    ax2.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(r"$\partial_z u$ prédit")
    ax2.set_ylabel('$z$')
    ax2.set_xlabel('$t$')

    im3 = ax3.pcolor(t, z, (np.sqrt((y[:, :, 0] - pred_sindy[:, :, 0]) ** 2) / (
                y[:, :, 0].max() - y[:, :, 0].min())).T, cmap='Reds', norm=res_norm)
    ax3.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im3, ax=ax3)
    ax3.set_title(r"NRMSE")
    ax3.set_ylabel('$z$')
    ax3.set_xlabel('$t$')

    im4 = ax4.pcolor(t, z, y[:, :, 1].T, cmap='seismic', norm=norm)
    ax4.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im4, ax=ax4)
    ax4.set_title(r"$\partial_z \theta$ réel")
    ax4.set_ylabel('$z$')
    ax4.set_xlabel('$t$')

    im5 = ax5.pcolor(t, z, pred_sindy[:, :, 1].T, cmap='seismic', norm=norm)
    ax5.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im5, ax=ax5)
    ax5.set_title(r"$\partial_z \theta$ prédit")
    ax5.set_ylabel('$z$')
    ax5.set_xlabel('$t$')

    im6 = ax6.pcolor(t, z, (np.sqrt((y[:, :, 1] - pred_sindy[:, :, 1]) ** 2) / (
                y[:, :, 1].max() - y[:, :, 1].min())).T, cmap='Reds', norm=res_norm)
    ax6.vlines(t[train_id[-1]], z[1], z[-2], ls='dashed', colors='k', linewidth=2)
    plt.colorbar(im6, ax=ax6)
    ax6.set_title(r"NRMSE")
    ax6.set_ylabel('$z$')
    ax6.set_xlabel('$t$')

    fig.tight_layout(pad=0.5)
    plt.savefig(f'{folder}/pb{num_pb}_res_kan.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{folder}/pb{num_pb}_res_kan.pdf', bbox_inches='tight', transparent=True, backend='pdf')
    plt.show()

def plot_nrmse_distribution(NRMSE, num_pb, folder='FIGURES'):
    plt.hist(NRMSE.flatten(), bins=100)
    plt.title("Distribution des NRMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{folder}/pb{num_pb}_nrmse_kan.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{folder}/pb{num_pb}_nrmse_kan.pdf', bbox_inches='tight', transparent=True, backend='pdf')
    plt.show()

def plot_feature_importance(scores, features, num_pb, folder='FIGURES'):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(features, scores, color='tab:blue')
    plt.yscale('log')
    plt.xlabel('Variables d\'entrée', fontsize=15)
    plt.ylabel('Score d\'attribution', fontsize=15)
    plt.title("Importance des variables d'entrée", fontsize=18)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height * 1.0, f'{height:.2e}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f'{folder}/pb{num_pb}_varimp_kan.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{folder}/pb{num_pb}_varimp_kan.pdf', bbox_inches='tight', transparent=True, backend='pdf')
    plt.show()