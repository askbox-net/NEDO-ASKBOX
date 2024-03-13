import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import numpy as np
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
    draw.text(pos, text, random.randint(8, 255))
    img = img.convert("L").convert("RGB")
    return np.array(img, dtype=np.uint8)


class CharDataset(Dataset):
    def __init__(self, chars, transform, max_len=80000):
        self.max_len = max_len
        self.chars = chars
        self.char_len = len(chars)
        self.transform = transform

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        img = get_char_img(self.chars[random.randint(0, self.char_len-1)])
        return self.transform(img)


if __name__ == '__main__':
    with open('./train_char.txt', 'r') as fp:
        chars = list(map(lambda x: x.strip(), fp.readlines()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE(100)
    net.to(device)

    transform = T.Compose([
        T.ToTensor(),
    ])

    kwargs = {
            'num_workers': 8,
            'pin_memory': True,
            'batch_size': 64,
            'shuffle': True,
    } 

    char_dataset = CharDataset(chars=chars, transform=transform)
    char_loader = DataLoader(dataset=char_dataset, **kwargs)

    learning_rate = 1e-3
    num_epochs = 20 

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = SGD(model, 0.1)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        for imgs in char_loader:
            imgs = imgs.to(device)
            out, mu, logVar = net(imgs)
            
            # loss function
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4:
            torch.save(net.state_dict(), f'vae_resnet18-100.pt')
        print('Epoch {}: Loss {}'.format(epoch, loss))
        scheduler.step()

    torch.save(net.state_dict(), f'vae_resnet18-100.pt')

