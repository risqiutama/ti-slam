import numpy as np
# import matplotlib.pylab as plt # python 2
# import matplotlib
import matplotlib.pyplot as plt # python 3
plt.switch_backend('agg')
plt.style.use('ggplot')

lw = 6
linestyle_ls = ['-', ':',  '--',  '-.', '-', '-']

def plot2d(output_pred, output_gt, fig_path):
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18

    pred_x, pred_y = output_pred[:, 3], output_pred[:, 7]
    plt.plot(pred_x, pred_y, linestyle=linestyle_ls[0], linewidth=lw, label='Prediction')
    gt_x, gt_y = output_gt[:, 3], output_gt[:, 7]
    plt.plot(gt_x, gt_y, linestyle=linestyle_ls[1], linewidth=lw, label='Ground_truth')

    # plt.gca().set_aspect("equal")
    plt.legend(loc='upper right', ncol=1)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.show()
    plt.savefig(fig_path, bbox_inches='tight')

def plot2d_from2d_pose(output_pred, output_gt, fig_path):
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18

    pred_x, pred_y = output_pred[:, 2], output_pred[:, 5]
    plt.plot(pred_x, pred_y, linestyle=linestyle_ls[0], linewidth=lw, label='Prediction')
    gt_x, gt_y = output_gt[:, 3], output_gt[:, 7]
    plt.plot(gt_x, gt_y, linestyle=linestyle_ls[1], linewidth=lw, label='Ground_truth')

    # plt.gca().set_aspect("equal")
    plt.legend(loc='pper right', ncol=1)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.show()
    plt.savefig(fig_path, bbox_inches='tight')
