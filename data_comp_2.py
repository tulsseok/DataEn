import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- 0. 기본 설정 ---
FILE_PATH = '/home/minjun/Downloads/OIBC_2025_DATA/train.csv'  # 1900만 행 원본 파일 경로
N_ROWS_PER_YEAR = 105120
N_TRAIN_YEARS = 10
N_VAL_YEARS = 2

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


import optuna
from sklearn.metrics import mean_absolute_error # (RMSE 대신 MAE로 평가)

print("\n--- 4. Optuna 하이퍼파라미터 튜닝 시작 ---")

# (A) 특징 및 데이터셋 정의 (Objective 함수가 접근해야 함)
target = TARGET
features = [col for col in train_df.columns if col not in [TARGET, TIME_COL]]
categorical_features = [ID_COL, 'hour', 'dayofweek', 'month', 'type']

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]


lgb_train = lgb.Dataset(
    X_train, y_train, 
    categorical_feature=categorical_features, 
    free_raw_data=False # 메모리 효율을 위해 원본 데이터 참조 유지
)
lgb_val = lgb.Dataset(
    X_val, y_val, 
    categorical_feature=categorical_features, 
    reference=lgb_train, 
    free_raw_data=False
)

# --- 1. Objective 함수 정의 ---
def objective(trial):
    """Optuna가 호출할 함수. trial 객체로 파라미터를 추천받고 Loss를 반환."""
    
    # (A) 하이퍼파라미터 탐색 공간 정의
    params = {
        'objective': 'regression_l1', # MAE
        'metric': 'mae',
        'device': 'gpu', # GPU 사용
        'n_estimators': 2000, # ★★★ 트리는 크게 고정 (Early Stopping이 최적값을 찾음)
        'random_state': 42,
        
        # --- Optuna가 탐색할 파라미터들 ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    # (B) 모델 생성 및 학습
    model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=2000,callbacks=[lgb.early_stopping(100, verbose=False)] )
    val_mae=model.best_score['valid_0']['mae']
    return val_mae


# --- 2. Study 생성 ---
# 'minimize' (최소화)가 목표 (MAE를 낮춰야 하므로)
study = optuna.create_study(direction='minimize')

# --- 3. Optimize 실행 ---
# n_trials=50 : 총 50번의 조합을 시도
print(f"Optuna 탐색 시작... (총 {30}회 시도)")
study.optimize(objective, n_trials=30, show_progress_bar=True)


# --- 4. 결과 확인 ---
print("\n--- Optuna 튜닝 완료! ---")
print(f"최적의 Validation MAE: {study.best_value:.4f}")
print("최적의 하이퍼파라미터:")
print(study.best_params)

# --- 5. (최종) 찾은 파라미터로 전체 모델 재학습 ---
print("\n--- 5. 최적의 파라미터로 최종 모델 학습 ---")
best_params = study.best_params
best_params['n_estimators'] = 1000 # Early Stopping을 위해 크게 설정
best_params['device'] = 'gpu'
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'mae'
best_params['random_state'] = 42

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=True)],
    categorical_feature=categorical_features
)