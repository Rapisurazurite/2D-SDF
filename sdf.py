import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import torch
from PIL import Image


MAX_DIST = 2147483647
null = ti.Vector([-1, -1, MAX_DIST])
vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])
eps = 1e-5


def min_max_rescale(tensor: torch.tensor or np.ndarray):
    _min = tensor.min()
    _max = tensor.max()
    return (tensor - _min) / (_max - _min + 1e-8)


def plt_show(tensor, norm=True, title=None, show=True, save=None, **kwargs):
    """
    Show a tensor as an image
    Support fomat: (C, H, W), (H, W, C), (H, W), (1, C, H, W). C must be 1 or 3
    Args:
        tensor: tensor or PIL image or numpy array
        norm: normalize the tensor to [0, 1]
        title: title of the image
        show: whether to show the image
        save: path to save the image
        kwargs: other arguments for plt.imshow
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(tensor, Image.Image):
        tensor = np.array(tensor)
    assert isinstance(tensor, np.ndarray), "Unsupported type: {}".format(type(tensor))
    try:
        tensor = min_max_rescale(tensor) if norm else tensor
        # squeeze batch dimension
        while tensor.shape[0] == 1:
            tensor = tensor[0]
        # if CHW, convert to HWC
        if tensor.shape[0] == 3:
            tensor = tensor.transpose(1, 2, 0)
        plt.imshow(tensor, **kwargs)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
    except Exception as e:
        print("Failed to show image: {}".format(e))


@ti.data_oriented
class SDF2D:
    def __init__(self, im):
        self.num = 0  # index of bit_pic

        self.im = im
        self.width, self.height = self.im.shape[0], self.im.shape[1]
        self.pic = ti.Vector.field(1, dtype=ti.f32, shape=(self.width, self.height))
        self.bit_pic_white = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.bit_pic_black = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.output_pic = ti.Vector.field(1, dtype=ti.i32, shape=(self.width, self.height))
        self.output_linear = ti.Vector.field(1, dtype=ti.f32, shape=(self.width, self.height))
        self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

    def reset(self, im):
        self.im = im

    def output_filename(self, ins):
        path = pathlib.Path(self.filename)
        out_dir = path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (path.stem + ins + path.suffix))

    @ti.kernel
    def pre_process(self, bit_pic: ti.template(), keep_white: ti.i32):  # keep_white, 1 == True, -1 == False
        for i, j in self.pic:
            if (self.pic[i, j][0] - 0.5) * keep_white > 0:
                bit_pic[0, i, j] = ti.Vector([i, j, 0])
                bit_pic[1, i, j] = ti.Vector([i, j, 0])
            else:
                bit_pic[0, i, j] = null
                bit_pic[1, i, j] = null

    @ti.func
    def cal_dist_sqr(self, p1_x, p1_y, p2_x, p2_y):
        return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2

    @ti.kernel
    def jump_flooding(self, bit_pic: ti.template(), stride: ti.i32, n: ti.i32):
        # print('n =', n, '\n')
        for i, j in ti.ndrange(self.width, self.height):
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                i_off = i + stride * di
                j_off = j + stride * dj
                if 0 <= i_off < self.width and 0 <= j_off < self.height:
                    dist_sqr = self.cal_dist_sqr(i, j, bit_pic[n, i_off, j_off][0],
                                                 bit_pic[n, i_off, j_off][1])
                    # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr,', ', i_off, j_off)
                    if not bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < bit_pic[1 - n, i, j][2]:
                        bit_pic[1 - n, i, j][0] = bit_pic[n, i_off, j_off][0]
                        bit_pic[1 - n, i, j][1] = bit_pic[n, i_off, j_off][1]
                        bit_pic[1 - n, i, j][2] = dist_sqr
                        # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr, ', ', i_off, j_off)

    @ti.kernel
    def copy(self, bit_pic: ti.template()):
        for i, j in ti.ndrange(self.width, self.height):
            self.max_reduction[i * self.width + j] = bit_pic[self.num, i, j][2]

    @ti.kernel
    def max_reduction_kernel(self, r_stride: ti.i32):
        for i in range(r_stride):
            self.max_reduction[i] = max(self.max_reduction[i], self.max_reduction[i + r_stride])

    @ti.kernel
    def post_process_udf(self, bit_pic: ti.template(), n: ti.i32, coff: ti.f32, offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[n, i, j][2]) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32, coff: ti.f32,
                         offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(
                ti.cast((ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf_linear_1channel(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in self.output_pic:
            self.output_linear[i, j][0] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    def gen_udf(self, dist_buffer, keep_white=True):

        keep_white_para = 1 if keep_white else -1
        self.pre_process(dist_buffer, keep_white_para)
        self.num = 0
        stride = self.width >> 1
        while stride > 0:
            self.jump_flooding(dist_buffer, stride, self.num)
            stride >>= 1
            self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 2, self.num)
        self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 1, self.num)
        self.num = 1 - self.num

    def find_max(self, dist_buffer):
        self.copy(dist_buffer)

        r_stride = self.width * self.height >> 1
        while r_stride > 0:
            self.max_reduction_kernel(r_stride)
            r_stride >>= 1

        return self.max_reduction[0]

    def mask2udf(self, normalized=(0, 1), to_rgb=True, output=True):  # unsigned distance
        self.pic.from_numpy(self.im)
        self.gen_udf(self.bit_pic_white)

        max_dist = ti.sqrt(self.find_max(self.bit_pic_white))

        if to_rgb:  # scale sdf proportionally to [0, 1]
            coefficient = 255.0 / max_dist
            offset = 0.0
        else:
            coefficient = (normalized[1] - normalized[0]) / max_dist
            offset = normalized[0]

        self.post_process_udf(self.bit_pic_white, self.num, coefficient, offset)
        if output:
            if to_rgb:
                cv2.imwrite(self.output_filename('_udf'), self.output_pic.to_numpy())

    def gen_udf_w_h(self):
        self.pic.from_torch(self.im)
        self.gen_udf(self.bit_pic_white, keep_white=True)
        self.gen_udf(self.bit_pic_black, keep_white=False)

    def mask2sdf(self, to_rgb=True, output=True):
        self.gen_udf_w_h()

        if to_rgb:  # grey value == 0.5 means sdf == 0, scale sdf proportionally
            max_positive_dist = ti.sqrt(self.find_max(self.bit_pic_white))
            min_negative_dist = ti.sqrt(self.find_max(self.bit_pic_black))  # this value is positive
            coefficient = 127.5 / max(max_positive_dist, min_negative_dist)
            offset = 127.5
            self.post_process_sdf(self.bit_pic_white, self.bit_pic_black, self.num, coefficient, offset)
            if output:
                cv2.imwrite(self.output_filename('_sdf'), self.output_pic.to_numpy())
        else:  # no normalization
            if output:
                pass
            else:
                self.post_process_sdf_linear_1channel(self.bit_pic_white, self.bit_pic_black, self.num)
        return self


class SdfInterpolator:
    def __init__(self, width, height):
        self.sdf = SDF2D(np.zeros((width, height, 1), dtype=np.uint8))

    def gen_sdf(self, img) -> np.ndarray:
        self.sdf.reset(img)
        return self.sdf.mask2sdf(to_rgb=False, output=False).output_linear.to_torch()[..., 0][None]

    def interpolate(self, img1, img2, alpha) -> np.ndarray:
        img1_sdf = self.gen_sdf(img1)
        img2_sdf = self.gen_sdf(img2)
        return img1_sdf * (1 - alpha) + img2_sdf * alpha


if __name__ == '__main__':
    ti.init(arch=ti.gpu, device_memory_GB=1.0, kernel_profiler=True, debug=True, print_ir=False)

    img_paths = ['./data/419.png', './data/5028.png']
    assert all(os.path.exists(path) for path in img_paths)
    img_0, img_1 = map(lambda path: torch.tensor(
        cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (512, 512), interpolation=cv2.INTER_CUBIC)), img_paths)
    print(img_0.shape, img_1.shape)

    plt.subplot(1, 2, 1)
    plt_show(img_0, show=False, cmap='gray')
    plt.subplot(1, 2, 2)
    plt_show(img_1, show=False, cmap='gray')
    plt.show()

    interpolator = SdfInterpolator(width=512, height=512)

    img_0, img_1 = [img.unsqueeze(-1) for img in [img_0, img_1]]
    print(img_0.shape, img_1.shape)

    for i in range(20 + 1):
        interpolate_im = interpolator.interpolate(img_0, img_1, i / 20)
        mask = torch.where(interpolate_im <= 0, 1, 0)
        plt_show(mask, "interpolate_im", cmap='gray')
