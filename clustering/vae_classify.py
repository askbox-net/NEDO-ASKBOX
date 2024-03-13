import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from model import VAE


fonts = [
    'TakaoGothic.ttf',
    'TakaoMincho.ttf',
    'TakaoPGothic.ttf',
    'TakaoPMincho.ttf',
]

def get_char_img(text, pixel=64):
    img = Image.new("L", (pixel, pixel), color=0x0)
    draw = ImageDraw.Draw(img)
    fonts_len = len(fonts)
    draw.font = ImageFont.truetype(f"./fonts/{fonts[random.randint(0, fonts_len-1)]}", size=pixel)
    W, H = img.size
    left, top, right, bottom = draw.font.getbbox(text)
    w, h = right - left, bottom - top
    pos = ((W - w)/2.0, (H - h*1.2)/2.0)
    draw.text(pos, text, 255)
    img = img.convert("L").convert("RGB")
    return np.array(img, dtype=np.uint8)


class CharDataset(Dataset):
    def __init__(self, chars, transform):
        self.len = len(chars)
        self.chars = chars
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = get_char_img(self.chars[idx])
        return self.transform(img)

def main():
    with open('./char.txt', 'r') as fp:
        chars = list(map(lambda x: x.strip(), fp.readlines()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE(100)
    net.load_state_dict(torch.load('vae_resnet18-100-submit.pt'))
    #net.load_state_dict(torch.load('vae_resnet18-100.pt'))

    net = net.to(device)
    net.eval()

    transform = T.Compose([
        T.ToTensor(),
    ])

    kwargs = {
            'num_workers': 8,
            'pin_memory': True,
            'batch_size': 8,
            'shuffle': False,
    } 

    char_dataset = CharDataset(chars=chars, transform=transform)
    char_dataset = DataLoader(dataset=char_dataset, **kwargs)

    mus = []
    logVARs = []
    zs = []
    for imgs in char_dataset:
        _, mu, logVAR = net(imgs.to(device))
        mus.append(mu.cpu().detach().numpy())
        logVARs.append(logVAR.cpu().detach().numpy())
        z = net.reparameterize(mu, logVAR).cpu().detach().numpy()
        zs.append(z)

    z = np.concatenate(zs)
    m = np.concatenate(mus)

    kmean = KMeans(n_clusters=1000)
    kmean.fit(m)

    pca = PCA(n_components=100)
    mpca = pca.fit_transform(m)

    pca_kmean = KMeans(n_clusters=1000)
    pca_kmean.fit(mpca)

    zkmean = KMeans(n_clusters=1000)
    zkmean.fit(z)

    preds = kmean.predict(m)
    zpreds = zkmean.predict(z)
    pca_preds = pca_kmean.predict(mpca)

    df = pd.DataFrame(preds,columns=['label'])
    df['char'] = chars
    df['zlabel'] = zpreds
    df['pca_label'] = pca_preds

    df = df.sort_values(['label','pca_label'])
    df['pca_label_diff'] = df['pca_label'].diff().fillna(-999).astype('int')
    df = df.sort_values(['label','zlabel'])
    df['zlabel_diff'] = df['zlabel'].diff().fillna(-999).astype('int')
    df = df.sort_values(['label','pca_label_diff', 'zlabel'])

    df = df.sort_values(['pca_label','label'])
    df['label_diff'] = df['label'].diff().fillna(-999).astype('int')

    df['s'] = df.apply(lambda x: int(x['label_diff'] == 0) + int(x['pca_label_diff'] == 0) + int(x['zlabel_diff'] == 0),axis=1)

    N = 6

    x = df.groupby(['pca_label']).apply(lambda x: x.nlargest(N, 's')).reset_index(drop=True)

    print("x.shape: ", x.shape)
    if x.shape[0] != N*1000:
        return False

    x['c'] = list(range(N)) * 1000

    y = pd.pivot(x, index='pca_label', columns='c', values='char').reset_index()

    y.columns = ['label'] + [ f'c{i}' for i in range(N)]
    y.to_csv('kanji_label.csv', index=False)
    return True

if __name__ == '__main__':
    while True:
        if main():
            break
