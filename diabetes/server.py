from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI()

# 2. Tải mô hình đã huấn luyện (Đảm bảo file pkl này là mô hình, không phải dataframe)
with open("diabetes.pkl", "rb") as file:
    model = pickle.load(file)

# 3. Định nghĩa danh sách tên nhãn (Class names) thủ công dựa trên bài toán của bạn
# Ví dụ: 0 là Không bị tiểu đường, 1 là Bị tiểu đường
class_names = ["Không bị tiểu đường", "Bị tiểu đường"]

# 4. Tạo cấu trúc dữ liệu đầu vào (Ví dụ mô hình cần 3 tham số đầu vào)
class PatientData(BaseModel):
    # Khai báo các biến đầu vào mô hình cần ở đây, ví dụ:
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# 5. Tạo endpoint để dự đoán
@app.post("/predict")
def predict(data: PatientData):
    input_data = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]
    prediction = model.predict(input_data)
    result = class_names[int(prediction[0])]
    return {"Prediction": result}