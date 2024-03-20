# 用 DistilBert
## 注意事項
- 執行 main.ipynb 程式後，會新增一個 data 資料夾，裡面是放訓練資料(IMDB資料集)。如果想改變存放的地點，可以到 `Load Data` 這個 Markdown 底下第２個 cell 去修改
- 目前僅支援 CPU 暫時不支援使用 GPU
- 建議使用 Python 官方的安裝器安裝 Python，不然可能會遇到 `Could not import the _lzma module` 的問題(出現在用 torchtext 導入 IMDB 資料集時)
- 鑑於權重檔太大，不上傳權重檔

## Install
使用前請先自行安裝 `Jupyter Notebook` ，之後執行 `\{path to yor environment's Python}\python -m pip install -r requirements.txt` 來安裝以下套件
- einops
- torchtext
- torch
- transformers