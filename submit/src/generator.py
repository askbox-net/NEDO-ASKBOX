from PIL import Image, ImageDraw
from PIL import ImageOps, ImageChops, ImageFont
from multiprocess import Process
import cv2
import os
import json
import pandas as pd
import numpy as np


class KanjiImage(object):
    @classmethod
    def get_params(cls, params_path, seed=42):
        with open(os.path.join(params_path, 'fonts.json')) as f:
            cls.font_details = json.load(f)
        with open(os.path.join(params_path, 'params.json')) as f:
            cls.params = json.load(f)
        cls.fonts = []
        cls.font_ratio = []
        # マルチスレッドの各スレッドでnumpyの乱数固定
        cls.rndstate = list(map(lambda x: np.random.RandomState(seed+x), range(cls.params['num_class'])))
        cls.font_path = os.path.join(params_path, 'fonts')
        for font, ratio in cls.font_details:
            cls.fonts.append(font)
            cls.font_ratio.append(ratio)
        cls.font_size = len(cls.font_details)

    @classmethod
    def gen_char_gray(cls, char, thread_id=0, size=(256,256)):
        img = Image.new("L", size, color=0)
        draw = ImageDraw.Draw(img)
        font_idx = cls.rndstate[thread_id].randint(0, cls.font_size)
        rnd = cls.rndstate[thread_id].random()
        draw.font = ImageFont.truetype(
                os.path.join(cls.font_path, cls.fonts[font_idx]),
                int(64*2.2*(rnd+1.))
                )

        left, top, right, bottom = draw.font.getbbox(char)
        w, h = right - left, bottom - top
        W, H = size

        m = 2
        pos = ((W - w)/2.0, (H - h*cls.font_ratio[font_idx])/2.0)
        rnd = cls.rndstate[thread_id].randint(8*m, 256)
        draw.text(pos, char, rnd, stroke_fill=True)
        img = img.rotate(cls.rndstate[thread_id].randint(0, 74)*5)

        img.crop(img.getbbox())

        if cls.rndstate[thread_id].randint(0, 2):
            img = ImageOps.flip(img)
        if cls.rndstate[thread_id].randint(0, 2):
            img = ImageOps.mirror(img)

        x = int(cls.rndstate[thread_id].normal() * W // 4)
        y = int(cls.rndstate[thread_id].normal() * W // 4)
        img = ImageChops.offset(img, x, y)
        img = np.array(img)
        kernel_size = cls.rndstate[thread_id].randint(-4, 4)
        if kernel_size > 2:
            img = cv2.blur(img, (kernel_size, kernel_size))
        return img

    @classmethod
    def chanel(cls, chars, thread_id=0):
        xs = []
        for i in range(len(chars)):
            img = cls.gen_char_gray(chars[i], thread_id=thread_id, size=(128*2, 128*2))
            xs.append(img)

        w = cls.rndstate[thread_id].rand(1, len(chars))
        w = w / w.sum()
        w = w.reshape(-1)
        xs = np.array(xs).transpose(1,2,0)

        img = xs @ w
        imax = max(1,int(img.max()))
        mul = cls.rndstate[thread_id].randint(imax, 256) / imax
        img = np.clip(img*mul, 0, 255).astype(np.uint8)

        return img

    @classmethod
    def gen_rgb(cls, chars, output_filename, thread_id=0, bitwise_not=False):
        cs1 = []
        for _ in range(3):
            cs1.append(cls.chanel(chars, thread_id))

        cs2 = []
        for _ in range(3):
            cs2.append(cls.chanel(chars, thread_id))

        rgb1 = np.stack(cs1).T
        rgb2 = np.stack(cs2).T
        #alpha = random.random()
        alpha = cls.rndstate[thread_id].random()
        beta = 1.0 - alpha
        rgb = cv2.addWeighted(rgb1, alpha, rgb2, beta, 0)
        if bitwise_not:
            rgb = cv2.bitwise_not(rgb)
        cv2.imwrite(output_filename, rgb)

    @classmethod
    def gen_rgb_label(cls, char, output_pattern, num_images, thread_id=0):
        bitwise_not = False
        if cls.rndstate[thread_id].randint(0, 4) == 0:
            bitwise_not = False
        for no in range(0, num_images, 1):
            cls.gen_rgb(char, output_pattern % no, thread_id=thread_id)

class Generator():
    @classmethod
    def get_params(cls, params_path, debug=True):
        cls.debug = debug
        cls.font_path = params_path
        with open(os.path.join(params_path, 'params.json')) as f:
            cls.params = json.load(f)
        cls.chars = pd.read_csv(os.path.join(params_path, 'kanji_label.csv'))
        KanjiImage.get_params(params_path)
        return True

    @classmethod
    def generate(cls, out_path):
        if cls.debug:
            print(out_path)

        img_size = cls.params['img_size']
        num_images = cls.params['num_images']
        img_type = cls.params['img_type']
        numof_thread = cls.params['numof_thread']

        chars = cls.chars.values
        for i in range(0, chars.shape[0], numof_thread):
            print('i = ', i)
            procs = []
            for thread_id in range(numof_thread):
                cat_path = os.path.join(out_path, 'cat_%03d' % (i+thread_id))
                if not os.path.exists(cat_path):
                    os.makedirs(cat_path)

                cs = chars[i+thread_id][1:7]
                output_pattern= f'{cat_path}/%03d.{img_type}'
                proc = Process(target=KanjiImage.gen_rgb_label, args=(cs, output_pattern, num_images, thread_id))
                procs.append(proc)

            for thread_id in range(numof_thread):
                procs[thread_id].start()

            for thread_id in range(numof_thread):
                procs[thread_id].join()


if __name__ == '__main__':
    res = Generator.get_params('../params')
    Generator.generate('../../outputs')

