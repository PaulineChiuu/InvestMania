from app import fetch_stock_data, StockPredictor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class ModelComparator:
    def __init__(self, data, symbol):
        """
        初始化模型比較器
        
        Args:
            data: 股票數據 DataFrame
            symbol: 股票代碼
        """
        self.data = data
        self.symbol = symbol
        self.results = {}
        
    def evaluate_markov_only(self, test_size=0.2):
        """評估純馬可夫鏈模型"""
        try:
            predictor = StockPredictor(self.data.copy(), self.symbol)
            predictor.prepare_markov_chain()
            
            # 分割測試集
            test_start = int(len(self.data) * (1 - test_size))
            test_data = self.data.iloc[test_start:].copy()
            
            # 計算實際收益率，並去除第一個值（因為它是NaN）
            actual_returns = test_data['Close'].pct_change().dropna()
            
            # 預測收益率
            predicted_returns = []
            
            # 從第二天開始預測（因為第一天沒有實際收益率）
            for i in range(len(actual_returns)):
                current_state = predictor.data['State'].iloc[test_start + i]
                next_state_probs = predictor.transition_matrix.loc[current_state]
                predicted_state = next_state_probs.idxmax()
                
                # 轉換狀態為收益率
                if predicted_state == '大漲':
                    pred_return = 0.08
                elif predicted_state == '小漲':
                    pred_return = 0.03
                elif predicted_state == '平盤':
                    pred_return = 0.0
                elif predicted_state == '小跌':
                    pred_return = -0.03
                else:  # 大跌
                    pred_return = -0.08
                    
                predicted_returns.append(pred_return)
            
            # 轉換為 numpy array 並確保長度一致
            predicted_returns = np.array(predicted_returns)
            actual_returns = actual_returns.values
            
            print(f"馬可夫模型 - 預測值數量: {len(predicted_returns)}")
            print(f"馬可夫模型 - 實際值數量: {len(actual_returns)}")
            
            # 確保長度一致
            min_len = min(len(predicted_returns), len(actual_returns))
            predicted_returns = predicted_returns[:min_len]
            actual_returns = actual_returns[:min_len]
            
            # 計算誤差
            mse = mean_squared_error(actual_returns, predicted_returns)
            mape = mean_absolute_percentage_error(
                actual_returns + 1,
                predicted_returns + 1
            )
            
            self.results['markov_only'] = {
                'mse': mse,
                'mape': mape * 100
            }
            
        except Exception as e:
            print(f"馬可夫評估錯誤: {str(e)}")
            raise
            
    def evaluate_combined_model(self, test_size=0.2):
        """評估結合LSTM和技術指標的完整模型"""
        try:
            # 分割測試集
            test_start = int(len(self.data) * (1 - test_size))
            test_data = self.data.iloc[test_start:].copy()
            
            # 獲取實際收益率，並去除第一個值
            actual_returns = test_data['Close'].pct_change().dropna()
            predicted_returns = []
            
            # 從第二天開始預測
            for i in range(len(actual_returns)):
                print(f"預測第 {i+1}/{len(actual_returns)} 天")
                
                # 使用到目前為止的數據創建新的預測器
                current_data = self.data.iloc[:test_start + i + 1].copy()
                current_predictor = StockPredictor(current_data, self.symbol)
                
                try:
                    # 訓練模型並進行預測
                    current_predictor.train_lstm()
                    current_predictor.prepare_markov_chain()
                    prediction = current_predictor.predict_next_day()
                    pred_return = prediction['predicted_return'] / 100
                    predicted_returns.append(pred_return)
                    
                except Exception as e:
                    print(f"預測第 {i+1} 天時發生錯誤: {str(e)}")
                    # 如果預測失敗，使用馬可夫預測
                    current_predictor.prepare_markov_chain()
                    current_state = current_predictor.data['State'].iloc[-1]
                    next_state_probs = current_predictor.transition_matrix.loc[current_state]
                    predicted_state = next_state_probs.idxmax()
                    
                    if predicted_state == '大漲':
                        pred_return = 0.08
                    elif predicted_state == '小漲':
                        pred_return = 0.03
                    elif predicted_state == '平盤':
                        pred_return = 0.0
                    elif predicted_state == '小跌':
                        pred_return = -0.03
                    else:
                        pred_return = -0.08
                        
                    predicted_returns.append(pred_return)
            
            predicted_returns = np.array(predicted_returns)
            actual_returns = actual_returns.values
            
            print(f"組合模型 - 預測值數量: {len(predicted_returns)}")
            print(f"組合模型 - 實際值數量: {len(actual_returns)}")
            
            # 確保長度一致
            min_len = min(len(predicted_returns), len(actual_returns))
            predicted_returns = predicted_returns[:min_len]
            actual_returns = actual_returns[:min_len]
            
            # 計算誤差
            mse = mean_squared_error(actual_returns, predicted_returns)
            mape = mean_absolute_percentage_error(
                actual_returns + 1,
                predicted_returns + 1
            )
            
            self.results['combined_model'] = {
                'mse': mse,
                'mape': mape * 100
            }
            
        except Exception as e:
            print(f"組合模型評估錯誤: {str(e)}")
            raise
            
    def compare_models(self):
        """比較兩種模型的表現"""
        try:
            print("評估純馬可夫鏈模型...")
            self.evaluate_markov_only()
            
            print("\n評估組合模型...")
            self.evaluate_combined_model()
            
            # 計算改善百分比
            mse_improvement = (
                (self.results['markov_only']['mse'] - self.results['combined_model']['mse']) /
                self.results['markov_only']['mse'] * 100
            )
            
            mape_improvement = (
                (self.results['markov_only']['mape'] - self.results['combined_model']['mape']) /
                self.results['markov_only']['mape'] * 100
            )
            
            return {
                'markov_only': self.results['markov_only'],
                'combined_model': self.results['combined_model'],
                'improvements': {
                    'mse_reduction': mse_improvement,
                    'mape_reduction': mape_improvement
                }
            }
        except Exception as e:
            print(f"模型比較錯誤: {str(e)}")
            raise

# 主程式
if __name__ == "__main__":
    test_symbols = ['2330.TW']  # 先測試台積電
    
    print("開始模型比較分析...")
    print("-" * 50)
    
    for symbol in test_symbols:
        print(f"\n分析股票: {symbol}")
        try:
            data = fetch_stock_data(symbol)
            if data is not None:
                print(f"成功獲取 {symbol} 的數據")
                comparator = ModelComparator(data=data, symbol=symbol)
                results = comparator.compare_models()
                
                print("\n純馬可夫鏈模型：")
                print(f"MSE: {results['markov_only']['mse']:.6f}")
                print(f"MAPE: {results['markov_only']['mape']:.2f}%")
                
                print("\n結合LSTM和技術指標的模型：")
                print(f"MSE: {results['combined_model']['mse']:.6f}")
                print(f"MAPE: {results['combined_model']['mape']:.2f}%")
                
                print("\n改善程度：")
                print(f"MSE降低: {results['improvements']['mse_reduction']:.2f}%")
                print(f"MAPE降低: {results['improvements']['mape_reduction']:.2f}%")
                print("-" * 50)
            else:
                print(f"無法獲取 {symbol} 的數據")
        except Exception as e:
            print(f"分析 {symbol} 時發生錯誤: {str(e)}")
            
    print("\n分析完成！")