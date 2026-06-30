import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================================================================
# 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU BAN ĐẦU
# =========================================================================
# Đọc file dữ liệu CS:GO
df = pd.read_excel("csgo/csgo.xlsx")

# Loại bỏ các cột thời gian không cần thiết cho mô hình học
df = df.drop(columns=['day', 'month', 'year', 'date', 'wait_time_s', 'match_time_s'])

# Chuyển đổi cột phân loại 'map' thành các cột số (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['map'], dtype=int)

label_mapping = {'Tie': 0, 'Win': 1, 'Lost': 2}
df_encoded['result'] = df_encoded['result'].map(label_mapping)

# =========================================================================
# 2. TÁCH BIẾN ĐỘC LẬP (X) VÀ BIẾN MỤC TIÊU (y)
# =========================================================================
# Giả sử cột cần dự đoán là 'result' (Kết quả trận đấu)
target = "result"
x = df_encoded.drop(columns=[target])
y = df_encoded[target]

# =========================================================================
# 3. CHIA TẬP DỮ LIỆU (TRAIN / TEST SPLIT)
# =========================================================================
# Chia theo tỷ lệ 80% để huấn luyện (Train) và 20% để kiểm thử (Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# =========================================================================
# 4. ĐỊNH NGHĨA BỘ TIỀN XỬ LÝ (PREPROCESSOR)
# =========================================================================
# Danh sách các cột dạng số cần xử lý chuyên sâu
num_cols = ['team_a_rounds', 'team_b_rounds', 'ping', 'kills', 'assists', 'deaths', 'mvps', 'hs_percent', 'points']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Gộp vào ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
],
    # QUAN TRỌNG: Giữ lại các cột 'map_*' đã encode trước đó và các cột không nằm trong num_cols
    remainder = 'passthrough'
)

# =========================================================================
# 5. ĐÓNG GÓI PIPELINE TỔNG (TIỀN XỬ LÝ + MÔ HÌNH ML)
# =========================================================================
# Kết hợp bộ tiền xử lý và thuật toán phân loại RandomForest vào một chuỗi duy nhất
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42)),
])

# =========================================================================
# 6. HUẤN LUYỆN MÔ HÌNH (TRAINING)
# =========================================================================
# Pipeline tự động chạy .fit_transform() tiền xử lý dữ liệu X_train rồi nạp vào Mô hình
full_pipeline.fit(x_train, y_train)

# =========================================================================
# 7. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# =========================================================================
# Dự đoán kết quả trên tập Test (Dữ liệu test cũng tự động được tiền xử lý qua pipeline)
y_pred = full_pipeline.predict(x_test)

print("Báo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred))