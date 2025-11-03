import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- 0. 기본 설정 ---
FILE_PATH = '/home/minjun/Downloads/OIBC_2025_DATA/train.csv'
TEST_FILE_PATH = '/home/minjun/Downloads/OIBC_2025_DATA/test.csv'
SUBMISSION_FILE_PATH = '/home/minjun/Downloads/OIBC_2025_DATA/submission_sample.csv'

N_ROWS_PER_YEAR = 105120
N_TRAIN_YEARS = 100
N_VAL_YEARS = 25

TARGET = 'nins'
ID_COL = 'pv_id'
TIME_COL = 'time'

# --- 1. Train 데이터 로드 및 전처리 ---
print("--- 1. Train 데이터 로드 및 전처리 시작 ---")
n_train_rows = N_ROWS_PER_YEAR * N_TRAIN_YEARS
train_df = pd.read_csv(FILE_PATH, nrows=n_train_rows)
print(f"Train 데이터 로드 완료. Shape: {train_df.shape}")

# 결측치 보간 (원본과 동일)
features_to_fill = [col for col in train_df.columns if col not in [TARGET, TIME_COL, ID_COL]]
print("Train 데이터 결측치(ffill) 채우는 중...")
train_df[features_to_fill] = train_df.groupby(ID_COL)[features_to_fill].ffill().bfill()

# --- 2. Validation 데이터 로드 및 전처리 ---
print("\n--- 2. Validation 데이터 로드 및 전처리 시작 ---")
val_df = pd.read_csv(
    FILE_PATH,
    skiprows= 1 + n_train_rows,
    nrows=N_ROWS_PER_YEAR * N_VAL_YEARS,
    names=train_df.columns,
    header=None
)
print(f"Validation 데이터 로드 완료. Shape: {val_df.shape}")

# 결측치 보간 (원본과 동일)
print("Validation 데이터 결측치(ffill) 채우는 중...")
val_df[features_to_fill] = val_df.groupby(ID_COL)[features_to_fill].ffill().bfill()

# --- 3. 특징 공학 (Feature Engineering) - V2 (수정됨) ---
print("\n--- 3. 특징 공학(FE) V2 시작 ---")

def create_features_v2(df):
    """
    데이터프레임을 받아 'nins'나 'pv_id'에 의존하지 않는
    '일반화' 가능한 특징(시간 특징)만 생성
    """
    df_ = df.copy()
    
    # (A) 시간 특징 (원본과 동일)
    df_[TIME_COL] = pd.to_datetime(df_[TIME_COL])
    df_['hour'] = df_[TIME_COL].dt.hour
    df_['dayofweek'] = df_[TIME_COL].dt.dayofweek
    df_['month'] = df_[TIME_COL].dt.month
    df_['hour_sin'] = np.sin(2 * np.pi * df_['hour'] / 24)
    df_['hour_cos'] = np.cos(2 * np.pi * df_['hour'] / 24)
    
    # (B) & (C) nins 의존 특징 (Lag, Rolling) - ***모두 제거***
    
    # FE로 인한 NA가 없으므로 dropna() 제거
    return df_

# Train/Validation 세트에 V2 특징 공학 함수 적용
train_df = create_features_v2(train_df)
val_df = create_features_v2(val_df)

print("FE V2 완료.")

# 'type' 열을 category로 변경 (원본과 동일)
# *** 'pv_id'는 더 이상 category로 변환하지 않습니다 (사용 안 함) ***
train_df['type'] = train_df['type'].astype('category')
val_df['type'] = val_df['type'].astype('category')

# --- 4. LGBM 학습 및 손실 계산 (수정됨) ---
print("\n--- 4. LGBM 학습 시작 (V2 전략) ---")

# (A) 특징(Features) 및 타겟(Target) 정의 (수정됨)
# 'pv_id'를 특징에서 제외합니다.
features = [col for col in train_df.columns if col not in [TARGET, TIME_COL, ID_COL, 'energy']]
target = TARGET

# 'pv_id'를 범주형 특징에서 제외합니다.
categorical_features = ['type', 'hour', 'dayofweek', 'month']

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

print(f"사용되는 특징 (Features): {features}")
print(f"사용되는 범주형 특징 (Categorical): {categorical_features}")

# (B) LightGBM 모델 정의 (원본과 동일)
lgb_model = lgb.LGBMRegressor(
    device='gpu',
    objective='regression_l1',
    metric='mae',
    n_estimators=1000,
    num_leaves=63,
    learning_rate=0.01,
    random_state=42
)

# (C) 모델 학습 (수정된 categorical_feature 사용)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=True)],
    categorical_feature=categorical_features
)

# (D) 최종 손실 계산 (원본과 동일)
print("\n--- 학습 완료! 최종 손실(Loss) 계산 ---")
val_pred = lgb_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_r2 = r2_score(y_val, val_pred)
print(f"Final Validation RMSE: {val_rmse:.4f}")
print(f"Final Validation R2 Score: {val_r2:.4f} (즉, {val_r2*100:.2f}%)")

# (E) 특징 중요도 확인 (참고)
print("\n--- V2 모델 특징 중요도 (Top 10) ---")
feature_importances = pd.Series(lgb_model.feature_importances_, index=features)
print(feature_importances.sort_values(ascending=False).head(10))

# --- 5. Test 데이터 예측 및 제출 (신규 추가) ---
print("\n--- 5. Test 데이터 예측 및 제출 파일 생성 시작 ---")

# (A) Test 데이터 로드
print("Test 데이터 로드 중...")
try:
    test_df = pd.read_csv(TEST_FILE_PATH)
    # submission_sample.csv는 nins를 채우기 위한 '틀'로 사용
    submission_df = pd.read_csv(SUBMISSION_FILE_PATH)
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    print("Test 예측을 건너뜁니다.")
    # 이 스크립트 블록의 나머지 부분은 실행되지 않도록 처리
    test_df = None

if test_df is not None:
    print(f"Test 데이터 로드 완료. Shape: {test_df.shape}")

    # (B) Test 데이터 전처리 (Train과 동일하게 수행)
    
    # 1. 결측치 보간
    print("Test 데이터 결측치(ffill) 채우는 중...")
    # test.csv에는 nins 열이 없으므로 features_to_fill 리스트가 train과 동일하게 작동
    test_features_to_fill = [col for col in test_df.columns if col not in [TARGET, TIME_COL, ID_COL]]
    test_df[test_features_to_fill] = test_df.groupby(ID_COL)[test_features_to_fill].ffill().bfill()

    # 2. 특징 공학 (V2) 적용
    print("Test 데이터에 FE V2 적용 중...")
    test_df = create_features_v2(test_df)

    # 3. 'type' 열을 category로 변경
    # 중요: test_df의 'type' 카테고리가 train_df의 카테고리 정보와 일치해야 합니다.
    # .astype('category')만 사용해도 lgb이 알아서 처리하지만,
    # 명시적으로 train의 카테고리를 따르게 하려면 아래와 같이 할 수 있습니다.
    test_df['type'] = pd.Categorical(test_df['type'], categories=train_df['type'].cat.categories)

    # (C) 예측 수행
    print("Test 데이터 예측 수행 중...")
    # X_test를 정의할 때, 학습에 사용된 'features' 리스트 순서와 동일하게 선택
    X_test = test_df[features]
    test_predictions = lgb_model.predict(X_test)

    # (D) 후처리: 음수 값을 0으로 변환 (요청사항)
    print("예측 값 후처리 중 (음수 -> 0)...")
    test_predictions_clipped = np.clip(test_predictions, 0, None)

    # (E) Submission 파일 생성
    print("Submission 파일 생성 중...")
    submission_df[TARGET] = test_predictions_clipped

    # 파일 저장
    output_filename = 'submission_final.csv'
    submission_df.to_csv(output_filename, index=False)
    print(f"--- 완료! {output_filename} 파일이 저장되었습니다. ---")
    print(submission_df.head())