import torch
import pickle
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cellpose import plot as cplt

import warnings
from torch.linalg import eigvals


def is_valid(pnm, gnm):
    for p in pnm:
        if p not in gnm:
            return False
    return True


def dt_region(hst, wst, hnm, wnm, 
              size=256, is_gt=True):
    pad = size // 2
    dt_lst = []
    for ph in range(hnm):
        hsz = ph * size
        for pw in range(wnm):
            wsz = pw * size
            pnm = [hst + hsz, hst + size + hsz, 
                   wst + wsz, wst + size + wsz]
            if is_gt:
                pnm += [hst - pad + hsz, hst + size + pad + hsz,
                        wst - pad + wsz, wst + size + pad + wsz]
            pnm = '_'.join([str(p) for p in pnm])
            dt_lst.append(pnm)
    return dt_lst


def dt_sublst(gnm, hst=256, wst=256, hnm=16, wnm=16,
              size=256, step=1, is_gt=True):
    # for 1024 x 1024, hst=wst=512, 
    # hnm=412, wnm=284, step = 4
    dt_lst= []
    # Here, we assume no boundary issue
    for pw in range(0, wnm, step):
        wsz = pw * size
        for ph in range(0, hnm, step):
            hsz = ph * size
            dt_reg = dt_region(hst+hsz, wst+wsz,
                               step, step, is_gt=is_gt)
            if is_valid(dt_reg, gnm):
                dt_lst.append(dt_reg)
    return dt_lst


def _d_novel(sigma1, sigma2):
    r"""
    The core and more efficient impl of d_FID

    Args:
        sigma1: Covariance of one image collection
        sigma2: Covariance of compared image collection
    """

    eigval = eigvals(sigma1 @ sigma2)
    eigval = eigval.real
    eigval[eigval < 0] = 0
    return 2 * eigval.sqrt().sum(-1)


def calc_d_fid(mu1, mu2, sigma1, sigma2):
    r"""
    The Function of d_FID calc

    Args:
        mu1: Mean of one image feat collection
        mu2: Mean of compared image feat collection
        sigma1: Covariance of one image feat collection
        sigma2: Covariance of compared image feat collection
    """

    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    fid_easy = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2)
    fid_hard = _d_novel(sigma1, sigma2)
    fid = fid_easy - fid_hard
    return fid


def calc_d_fid3(mu1, mu2, sigma1, sigma2):
    mu1 = torch.atleast_2d(mu1)
    mu2 = torch.atleast_2d(mu2)

    sigma1 = torch.atleast_3d(sigma1)
    sigma2 = torch.atleast_3d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    dif = mu1 - mu2
    fid_easy = (dif ** 2).sum(-1) + torch.vmap(torch.trace)(sigma1) + torch.vmap(torch.trace)(sigma2)
    fid_hard = _d_novel(sigma1, sigma2)
    fid = fid_easy - fid_hard
    return fid

def calc_mean_var(mu, scm, tot):
    mu = mu / tot[..., None]
    scm = scm / tot[..., None, None]
    sigma = scm - mu.unsqueeze(-1) @ mu.unsqueeze(-2)
    return mu, sigma

def calc_slc_all_fid(m_r, s_r, tot_r,
                     m_g, s_g, tot_g, is_str=False):
    m_r0, s_r0 = calc_mean_var(m_r, s_r, tot_r)
    m_g0, s_g0 = calc_mean_var(m_g, s_g, tot_g)
    fid0 = calc_d_fid3(m_r0, m_g0, s_r0, s_g0)
    m_r1, s_r1 = calc_mean_var(m_r.sum(0)[None], s_r.sum(0)[None], tot_r.sum(0)[None])
    m_g1, s_g1 = calc_mean_var(m_g.sum(0)[None], s_g.sum(0)[None], tot_g.sum(0)[None])
    fid1 = calc_d_fid3(m_r1, m_g1, s_r1, s_g1)
    fid = torch.cat((fid0, fid1)).cpu().numpy()
    if is_str:
        return [f'{fd:.2f}' for fd in fid]
    return fid

def calc_mean_std_msk(stat, dims):
    msk = stat!=0
    avg = (stat*msk).sum(dims)/msk.sum(dims)
    # Here, we always assume (batch, z) for stat
    var = (((stat - avg)**2)*msk).sum(dims)/msk.sum(dims)
    return avg, var.sqrt()

def calc_slc_all_1d(stat, is_str=False):
    avg0, std0 = calc_mean_std_msk(stat, 0)
    avg1, std1 = calc_mean_std_msk(stat, (0, 1))
    avg = torch.cat((avg0, avg1[None]))
    std = torch.cat((std0, std1[None])) 
    if is_str:
        return [f'{a:.2f} {s:.2f}' for a, s in zip(avg, std)]
    return avg, std

def calc_cellpose(img, mod, met, 
                  pth, roi, debug):
    im = img.astype('float') / 255.
    im_lst = np.split(im[:, 0], im.shape[0], 0)
    im_lst = [(v, i[0]) for v, i in enumerate(im_lst) if (img[v, 0] != 0).any()]
    if not im_lst:
        print(f'{roi} all black, ignore.')
        return

    vl_lst, mk_lst = zip(*im_lst)
    masks, flows, styles, diams = mod.eval(
        list(mk_lst), diameter=None, 
        normalize=False, channels=[0, 0])
    
    expr = im.mean((-1, -2))
    out = [torch.zeros(3, im.shape[0]),
           # Estimate the maximum num of cells
           torch.zeros(512, im.shape[0]), 
           vl_lst]
    for mid, msk in zip(vl_lst, masks): 
        cnt = np.unique(msk, return_counts=True)[1]
        cnt = cnt[1:]
        out[0][0, mid] = len(cnt)
        out[0][1:, mid] = torch.FloatTensor(expr[mid])
        out[1][:len(cnt), mid] = torch.FloatTensor(cnt)
    mlen = int(out[0][0].max())
    assert mlen <= out[1].shape[0]
    out[1] = out[1][:mlen]

    met['nstat'].append(out[0])
    met['narea'].append(out[1])
    met['valid'].append(out[2])

    if debug:
        print(roi, vl_lst, expr.shape, out[1].shape)
        for mid in range(im.shape[0]):
            if mid not in vl_lst:
                if (expr[mid] != 0).any():
                    print(mid, expr[mid])
        _s = random.choice(range(len(vl_lst)))
        fig = plt.figure(figsize=(12, 5))
        cplt.show_segmentation(fig, img[vl_lst[_s], 0], masks[_s],
                               flows[_s][0], channels=[0, 0])
        plt.tight_layout()
        plt.savefig(str(pth / f'{roi}_{vl_lst[_s]}.png'),
                    bbox_inches='tight', dpi=200)
        plt.close()

class PSNR(torch.nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, mval=255.):
        super(PSNR, self).__init__()
        self.mval = mval

    def forward(self, img1, img2):
        if len(img1.shape) == 3:
            dim = [1, 2]
        elif len(img1.shape) == 4:
            dim = [1, 2, 3]
        mse = torch.mean((img1 - img2) ** 2, dim=dim)
        return 20 * torch.log10(self.mval / torch.sqrt(mse))


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
        unsqueeze=False
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.unsqueeze = unsqueeze

    def forward(self, X, Y):
        if self.unsqueeze:
            # this suggests that the input 
            # has only 3 channels [N, H, W]
            # then unsqueeze the C channel
            assert len(X.shape) == len(Y.shape) == 3
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
        unsqueeze=False
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.win_gray = _fspecial_gauss_1d(win_size, win_sigma).repeat([1, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.unsqueeze = unsqueeze

    def forward(self, X, Y):
        is_gray = False
        if len(X.shape) == 3:
            X, Y = X[:, None], Y[:, None] 
            is_gray = True
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win_gray if is_gray else self.win,
            weights=self.weights,
            K=self.K,
        )
    

if __name__ == '__main__':
    import torch
    # mu = torch.rand(3, 50)
    # scm = torch.rand(3, 50, 50)
    # tot = torch.rand(3)
    # a = mu.clone()
    # m0, s0 = calc_mean_var(mu, scm, tot)
    # print((a == mu).all())

    # ssim_ = SSIM(channel=2, size_average=False)
    # a = torch.rand(23, 2, 125,125)
    # b = torch.rand(23, 2, 125,125)
    # c = ssim_(a, b)
    # print(c.shape)