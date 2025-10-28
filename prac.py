import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- 0. 기본 설정 ---
FILE_PATH = '/home/minjun/Downloads/OIBC_2025_DATA/train.csv'  # 1900만 행 원본 파일 경로
N_ROWS_PER_YEAR = 105120
N_TRAIN_YEARS = 100
N_VAL_YEARS = 15

TARGET = 'nins'
ID_COL = 'pv_id'
TIME_COL = 'time'

# --- 1. Train 데이터 로드 및 전처리 ---
print("--- 1. Train 데이터 로드 및 전처리 시작 ---")
n_train_rows = N_ROWS_PER_YEAR * N_TRAIN_YEARS

# 1. 원본데이터 1행 - (105120 * 5)행 분리
train_df = pd.read_csv(FILE_PATH, nrows=n_train_rows)
print(f"Train 데이터 로드 완료. Shape: {train_df.shape}")

# 1. (cont.) 결측치 보간 (ffill 사용)
# nins와 time, id를 제외한 모든 열(1시간/15분 데이터)을 ffill
features_to_fill = [col for col in train_df.columns if col not in [TARGET, TIME_COL, ID_COL]]

# id별로 그룹화하여 ffill/bfill 수행 (데이터가 섞이지 않도록)
print("Train 데이터 결측치(ffill) 채우는 중...")
train_df[features_to_fill] = train_df.groupby(ID_COL)[features_to_fill].ffill().bfill()


# --- 2. Validation 데이터 로드 및 전처리 ---
print("\n--- 2. Validation 데이터 로드 및 전처리 시작 ---")
# Train 데이터 이후의 1년치(val) 데이터 로드
# 원본 파일의 헤더(1줄) + train 데이터(n_train_rows)만큼 건너뛰기
val_df = pd.read_csv(
    FILE_PATH,
    skiprows= 1 + n_train_rows,
    nrows=N_ROWS_PER_YEAR * N_VAL_YEARS,
    names=train_df.columns,  # train_df에서 읽은 컬럼명 사용
    header=None              # 헤더를 이미 건너뛰었으므로
)
print(f"Validation 데이터 로드 완료. Shape: {val_df.shape}")

# 2. (cont.) 결측치 보간 (ffill 사용)
print("Validation 데이터 결측치(ffill) 채우는 중...")
val_df[features_to_fill] = val_df.groupby(ID_COL)[features_to_fill].ffill().bfill()


# --- 3. 특징 공학 (Feature Engineering) ---
print("\n--- 3. 특징 공학(FE) 시작 ---")

def create_features(df):
    """데이터프레임을 받아 시간/시차/이동평균 특징을 생성"""
    df_ = df.copy()
    
    # 3. 원본 time열 시계열로 변경
    df_[TIME_COL] = pd.to_datetime(df_[TIME_COL])
    
    # (A) 시간 특징
    df_['hour'] = df_[TIME_COL].dt.hour
    df_['dayofweek'] = df_[TIME_COL].dt.dayofweek
    df_['month'] = df_[TIME_COL].dt.month
    df_['hour_sin'] = np.sin(2 * np.pi * df_['hour'] / 24)
    df_['hour_cos'] = np.cos(2 * np.pi * df_['hour'] / 24)
    
    # (B) 시차(Lag) 특징 (id별로 그룹화하여 생성)
    print("Lag 특징 생성 중...")
    df_['nins_lag_1h'] = df_.groupby(ID_COL)[TARGET].shift(12)  # 1시간 전
    df_['nins_lag_24h'] = df_.groupby(ID_COL)[TARGET].shift(288) # 24시간 전
    
    # (C) 이동평균(Rolling) 특징 (id별로 그룹화하여 생성)
    print("Rolling 특징 생성 중...")
    # .reset_index(0, drop=True)는 groupby + rolling 사용 시 인덱스를 맞추기 위한 필수 작업
    df_['nins_roll_mean_1h'] = df_.groupby(ID_COL)[TARGET].rolling(window=12).mean().reset_index(0, drop=True)
    df_['nins_roll_std_1h'] = df_.groupby(ID_COL)[TARGET].rolling(window=12).std().reset_index(0, drop=True)
    
    # FE로 인해 생긴 NA값들(Lag, Rolling의 맨 앞부분)을 제거
    df_ = df_.dropna()
    
    return df_

# Train/Validation 세트에 특징 공학 함수 적용
train_df = create_features(train_df)
val_df = create_features(val_df)

print(f"FE 완료 후 Train Shape: {train_df.shape}")
print(f"FE 완료 후 Validation Shape: {val_df.shape}")

# 4. LGBM 학습을 위해 'pv_id' 타입을 string('object')에서 'category'로 변경
print("\n--- 3-1. 'pv_id' 타입을 category로 변경 ---")
train_df[ID_COL] = train_df[ID_COL].astype('category')
val_df[ID_COL] = val_df[ID_COL].astype('category')
train_df['type'] = train_df['type'].astype('category')
val_df['type'] = val_df['type'].astype('category')

# --- 4. LGBM 학습 및 손실 계산 ---
print("\n--- 4. LGBM 학습 시작 ---")

# (A) 특징(Features) 및 타겟(Target) 정의
# 원본 시간열과 타겟('nins')을 제외한 모든 것을 특징으로 사용
features = [col for col in train_df.columns if col not in [TARGET, TIME_COL]]
target = TARGET

# id와 시간 관련 열들은 범주형(Categorical)으로 지정
categorical_features = [ID_COL, 'hour', 'dayofweek', 'month']

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

# (B) LightGBM 모델 정의
lgb_model = lgb.LGBMRegressor(
    device='gpu',
    objective='regression_l1',  # MAE (L1 Loss) - nins=0이 많을 수 있으므로 RMSE(L2)보다 안정적
    metric='mae',               # 평가 지표
    n_estimators=1000,          # 최대 트리 개수
    learning_rate=0.05,
    random_state=42
)

# (C) 모델 학습 (Train/Valid Loss 동시 계산)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=True)],
    categorical_feature=categorical_features
)

# (D) 최종 손실 계산 (RMSE로도 확인)
print("\n--- 학습 완료! 최종 손실(Loss) 계산 ---")
train_pred = lgb_model.predict(X_train)
val_pred = lgb_model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"Final Training RMSE: {train_rmse:.4f}")
print(f"Final Validation RMSE: {val_rmse:.4f}")
train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)

print(f"Final Training R2 Score: {train_r2:.4f} (즉, {train_r2*100:.2f}%)")
print(f"Final Validation R2 Score: {val_r2:.4f} (즉, {val_r2*100:.2f}%)")

# (E) 특징 중요도 확인 (참고)
print("\n--- 특징 중요도 (Top 10) ---")
feature_importances = pd.Series(lgb_model.feature_importances_, index=features)
print(feature_importances.sort_values(ascending=False).head(10))