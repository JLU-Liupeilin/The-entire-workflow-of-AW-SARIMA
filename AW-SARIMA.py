# -*- coding: utf-8 -*-
"""
AW-SARIMA-model 2025.3.10
本代码所依赖库：numpy, pandas, pywavelets, statsmodels, scipy, sklearn, tqdm
本代码为示例代码，数据集为人造数据集，但是十分能够代表实际情况并且展示模型的可靠性，后续将更新论文中使用的数据集。
"""

import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox, yeojohnson
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, List
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta
import time

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 需要tqdm库显示进度条，请执行: pip install tqdm")
    exit()

warnings.filterwarnings("ignore")


class AWSARIMA:
    def __init__(self, wavelet: str = 'db4', max_decomp_level: int = 6,
                 weekly_window: int = 7, daily_window: int = 1):
        """初始化AW-SARIMA模型"""
        self.wavelet = wavelet
        self.max_decomp_level = max_decomp_level
        self.weekly_window = weekly_window
        self.daily_window = daily_window
        self.models = {}  # 存储各分量模型
        self.lmbda = None  # 变换参数
        self.decomp_level = None  # 分解层数
        self.scale_factors = {}  # 标准化参数
        self._is_trained = False
        self.alpha = 0.7  # 复合能量权重
        self.k_params = {'k_min': 0.2, 'k_max': 1.5}  # 动态权重参数
        self.training_time = 0.0  # 训练耗时

    def _calculate_aic_bic(self, coeffs: List[np.ndarray]) -> float:
        """AIC+BIC混合评价函数"""
        n = sum(len(c) for c in coeffs)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * np.var(coeffs))
        aic = 2 * (len(coeffs) + 1) - 2 * log_likelihood
        bic = np.log(n) * (len(coeffs) + 1) - 2 * log_likelihood
        return 0.6 * aic + 0.4 * bic

    def _optimize_decomp_level(self, series: np.ndarray) -> int:
        """优化小波分解层数"""
        best_j, best_score = 1, np.inf
        for j in tqdm(range(1, self.max_decomp_level + 1),
                      desc="Optimizing decomposition levels"):
            try:
                coeffs = pywt.wavedec(series, self.wavelet, level=j)
                score = self._calculate_aic_bic(coeffs)
                if score < best_score and len(coeffs[-1]) > 10:
                    best_score, best_j = score, j
            except ValueError:
                break
        return best_j

    def _wavelet_decomposition(self, series: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """小波分解"""
        self.decomp_level = self._optimize_decomp_level(series)
        coeffs = pywt.wavedec(series, self.wavelet, level=self.decomp_level)
        return coeffs[0], coeffs[1:]

    def _composite_energy(self, H: pd.Series) -> pd.Series:
        """复合能量计算"""
        short_energy = H.rolling(window=self.daily_window).var()
        long_energy = H.rolling(window=self.weekly_window).var()
        return self.alpha * short_energy + (1 - self.alpha) * long_energy

    def _dynamic_threshold(self, H: pd.Series) -> pd.Series:
        """动态阈值计算"""
        Q_weekly = H.rolling(window=self.weekly_window).quantile(0.95)
        IQR_daily = H.rolling(window=self.daily_window).apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        E = self._composite_energy(H)

        Q_low = np.percentile(E, 25)
        Q_high = np.percentile(E, 75)

        def variational_loss(params):
            s, delta = params
            k = np.zeros_like(E)
            k[E < Q_low] = self.k_params['k_min']
            mask = (E >= Q_low) & (E < Q_high)
            k[mask] = self.k_params['k_min'] + s * (E[mask] - Q_low) / (Q_high - Q_low)
            k[E >= Q_high] = self.k_params['k_max'] - delta * (E[E >= Q_high] - Q_high) / (Q_high - Q_low)
            return np.mean(np.gradient(k) ** 2)

        res = minimize(variational_loss, [0.5, 0.2], bounds=[(0, 1), (0, 0.5)])
        s_opt, delta_opt = res.x

        k = np.piecewise(E,
                         [E < Q_low, (E >= Q_low) & (E < Q_high), E >= Q_high],
                         [self.k_params['k_min'],
                          lambda x: self.k_params['k_min'] + s_opt * (x - Q_low) / (Q_high - Q_low),
                          lambda x: self.k_params['k_max'] - delta_opt * (x - Q_high) / (Q_high - Q_low)])

        return Q_weekly + k * IQR_daily

    def _denoise_signal(self, H: pd.Series) -> pd.Series:
        """信号去噪"""
        threshold = self._dynamic_threshold(H)
        denoised = np.where(np.abs(H) > threshold,
                            np.sign(H) * (np.abs(H) - threshold),
                            0)
        return denoised

    def _sarima_model_selection(self, series: np.ndarray) -> Tuple[Tuple, Tuple]:
        """SARIMA参数选择"""
        best_aic = np.inf
        best_order = (0, 0, 0)
        best_seasonal = (0, 0, 0, 0)

        param_combinations = [(p, d, q, P, D, Q, S)
                              for p in range(4)
                              for d in range(2)
                              for q in range(4)
                              for P in range(3)
                              for D in range(2)
                              for Q in range(3)
                              for S in [3, 5, 7, 15]]

        progress = tqdm(total=len(param_combinations), desc="SARIMA参数搜索")

        for p, d, q, P, D, Q, S in param_combinations:
            try:
                model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, S))
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, S)
            except:
                continue
            progress.update(1)
        progress.close()
        return best_order, best_seasonal

    def train(self, series: pd.Series) -> None:
        """模型训练（含时间统计）"""
        start_time = time.time()

        # 数据变换
        with tqdm(total=5, desc="数据预处理") as pbar:
            if series.min() > 0:
                transformed, self.lmbda = boxcox(series)
            else:
                transformed, self.lmbda = yeojohnson(series)
            pbar.update(1)

            # 小波分解
            cJ, d_bands = self._wavelet_decomposition(transformed)
            pbar.update(1)

            # 低频分量建模
            self.models['low'] = SARIMAX(cJ, order=self._sarima_model_selection(cJ)[0]).fit(disp=False)
            pbar.update(1)

            # 高频分量处理
            high_comp_progress = tqdm(d_bands, desc="处理高频分量")
            for i, d in enumerate(high_comp_progress, 1):
                d_series = pd.Series(d)
                denoised = self._denoise_signal(d_series)
                order, seasonal = self._sarima_model_selection(denoised)
                self.models[f'high_{i}'] = SARIMAX(denoised, order=order,
                                                   seasonal_order=seasonal).fit(disp=False)
            pbar.update(1)

            self._is_trained = True
            pbar.update(1)

        self.training_time = time.time() - start_time

    def predict(self, steps: int) -> np.ndarray:
        """预测"""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        start_time = time.time()

        with tqdm(total=3, desc="生成预测") as pbar:
            # 低频预测
            low_pred = self.models['low'].get_forecast(steps).predicted_mean
            pbar.update(1)

            # 高频预测
            high_preds = []
            high_models = [name for name in self.models if name.startswith('high_')]
            for name in tqdm(high_models, desc="预测高频分量"):
                pred = self.models[name].get_forecast(steps).predicted_mean
                high_preds.append(pred)
            pbar.update(1)

            # 重构与逆变换
            total_pred = low_pred + sum(high_preds)
            if hasattr(self.lmbda, 'lambda_'):
                pred = np.power(total_pred * self.lmbda.lambda_ + 1,
                                1 / self.lmbda.lambda_) - 1
            else:
                pred = (total_pred * self.lmbda + 1) ** (1 / self.lmbda)
            pbar.update(1)

        print(f"\n预测耗时: {time.time() - start_time:.2f}秒")
        return pred


# 生成非平稳时间序列数据集
def generate_nonstationary_data(num_points=1000):
    """生成包含趋势、季节性和突变的时间序列"""
    np.random.seed(42)

    with tqdm(total=5, desc="生成数据") as pbar:
        # 基础组件
        time_arr = np.arange(num_points)
        trend = 0.05 * time_arr  # 线性趋势
        pbar.update(1)

        seasonal = 10 * np.sin(2 * np.pi * time_arr / 365)  # 年周期
        pbar.update(1)

        # 突变点
        np.random.seed(42)
        for _ in range(5):
            idx = np.random.randint(100, 900)
            trend[idx:] += np.random.normal(0.2, 0.05)
        pbar.update(1)

        # 噪声（时变方差）
        noise = np.zeros(num_points)
        for i in range(num_points):
            noise[i] = np.random.normal(0, 0.1 + 0.05 * np.sin(i / 200))
        pbar.update(1)

        # 组合生成
        series = trend + seasonal + noise

        # 创建时间索引
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=int(x)) for x in time_arr]
        pbar.update(1)

    return pd.Series(series, index=dates)



if __name__ == "__main__":
    # 生成数据集
    print("=" * 40)
    print("开始生成非平稳时间序列数据集")
    data = generate_nonstationary_data()
    train_data = data.iloc[:800]  # 前800个点训练
    test_data = data.iloc[800:]  # 后200个点测试

    # 初始化模型
    print("\n" + "=" * 40)
    print("初始化AW-SARIMA模型")
    model = AWSARIMA(wavelet='db4', weekly_window=7, daily_window=1)

    # 训练模型
    print("\n" + "=" * 40)
    print("开始模型训练")
    model.train(train_data)
    print(f"\n总训练时间: {model.training_time:.2f}秒")

    # 进行预测
    print("\n" + "=" * 40)
    print("开始预测")
    predictions = model.predict(steps=len(test_data))

    # 评估结果
    print("\n" + "=" * 40)
    print("评估结果:")
    rmse = np.sqrt(mean_squared_error(test_data.values, predictions))
    mae = mean_absolute_error(test_data.values, predictions)
    mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 可视化对比
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='真实值')
    plt.plot(test_data.index, predictions, label='预测值', linestyle='--')
    plt.title("非平稳时间序列预测")
    plt.xlabel("日期")
    plt.ylabel("数值")
    plt.legend()
    plt.show()