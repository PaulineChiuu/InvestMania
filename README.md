# 股票預測系統

這是一個結合LSTM深度學習、馬爾可夫鏈分析和技術指標的綜合股票預測系統。

## 功能特點

- 從Yahoo Finance實時獲取股票數據
- 基於LSTM的價格預測
- 技術指標分析（RSI、MACD、布林通道）
- 馬爾可夫鏈狀態預測
- 互動式網頁界面
- Excel報告生成（選配功能）

## 系統需求

- Python 3.8 或更高版本
- 請查看 `requirements.txt` 獲取所需的Python套件清單

## 安裝說明

1. 下載專案：
```bash
git clone [你的專案網址]
cd stock-prediction-system
```

2. 創建並啟動虛擬環境（建議）：
```bash
python -m venv venv
# Windows系統：
venv\Scripts\activate
# Linux/Mac系統：
source venv/bin/activate
```

3. 安裝依賴項：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 啟動應用程式：
```bash
python app.py
```

2. 使用網頁界面：
- 打開瀏覽器訪問 `http://localhost:5000`
- 進入預測表單頁面
- 輸入股票代碼（例如：2330.TW 代表台積電）
- 查看預測結果

### Excel記錄功能（選配）

如果你想要追蹤預測記錄：

1. 確保 `stock_recorder.py` 與 `app.py` 在同一目錄下
2. 系統會自動創建名為 `InvestMania準確度紀錄.xlsx` 的Excel檔案
3. 每支股票都會有獨立的工作表

## 專案結構

```
├── app.py                 # 主應用程式
├── stock_recorder.py      # Excel記錄模組（選配）
├── requirements.txt       # 套件依賴清單
├── models/               # 模型儲存目錄
└── templates/            # HTML模板
    ├── index.html
    ├── welcome.html
    └── instructions.html
```

## 詳細功能說明

### 技術分析
- 移動平均線（MA5、MA20）
- 相對強弱指標（RSI）
- 移動平均收斂散度（MACD）
- 布林通道
- 成交量分析

### 預測模型
- LSTM深度學習模型
  - 3層LSTM結構配合Dropout層
  - 使用早停法防止過擬合
  - 可自定義序列長度
- 馬爾可夫鏈狀態預測
  - 五種狀態：大幅上漲、小幅上漲、盤整、小幅下跌、大幅下跌
  - 狀態轉移機率矩陣
  - 歷史狀態分析

### 視覺化
- 股價趨勢圖
- 技術指標疊加顯示
- 預測區間置信度
- 訓練歷史圖表

## 注意事項

- 系統需要穩定的網路連接以獲取股票數據
- LSTM模型訓練可能需要一定時間，取決於硬體配置
- 請確保有足夠的硬碟空間存儲模型
- Excel記錄功能為選配，可以根據需要啟用或停用

## 問題回報

如果你發現任何問題或有改進建議，歡迎提交議題！

## 授權條款

[選擇的授權條款]