# メインファイルの説明

1. vae_train.py: VAEを学習するプログラム
2. vae_classify.py: csvファイル(kanji_label.csv)を作成するクラスタリング部分のプログラム
3. vae_resnet18-100-submit.pt: submitで利用したモデルファイル
4. char.txt: クラスタリング対象の文字
5. train_char.txt: 学習用の文字

# ディレクトリ構造

```bash
.
├── char.txt
├── fonts
│   ├── TakaoGothic.ttf
│   ├── TakaoMincho.ttf
│   ├── TakaoPGothic.ttf
│   └── TakaoPMincho.ttf
├── kanji_label.csv
├── model
│   └── __init__.py
├── readme.md
├── requirements.txt
├── train_char.txt
├── vae_classify.py
├── vae_resnet18-100-submit.pt
└── vae_train.py

2 directories, 13 files
```
