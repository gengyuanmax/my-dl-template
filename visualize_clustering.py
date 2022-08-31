import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

items = json.load(open('/nfs/data3/zhang/Anet-encode/split/train.json', "r"))

def fancy_dendrogram(*args, **kwargs):
    gt_n = kwargs.pop('gt_n', None)
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram {}'.format(gt_n))
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

for vid, v in items.items():
    duration = v['duration']
    n = len(v['timestamps'])

    vis_feat = np.load('/nfs/data3/zhang/Anet-encode/videos/' + vid + '.npy')
    # vis_feat = scaler.fit_transform(vis_feat)
    # eva, eve = np.linalg.eig(np.cov(vis_feat))

    # nn = 0
    # # for i in range(len(eva)):
    # #     if np.sum(eva[:i+1]) / np.sum(eva) >= 0.7:
    # #         nn = i+1
    # #         break
    # print(vid, np.sum(eva > 0.02*eva.sum()), n)

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    Z = linkage(vis_feat, 'single', 'cosine', False)


    plt.figure(figsize=(50, 10))
    plt.title('Hierarchical Dendrogram {}'.format(n))
    plt.xlabel('sample index')
    plt.ylabel('distance')
    # dendrogram(Z, leaf_rotation=90., leaf_font_size=8.
    # )
    fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=0.04, gt_n=n)

    plt.savefig('/nfs/data3/zhang/Anet-encode/dendrogram_fancy/' + vid + '.png')
    plt.close()

    print(vid)