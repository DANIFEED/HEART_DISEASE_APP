import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Прогноз сердечных заболеваний", layout="centered")

# === ПУТЬ К МОДЕЛИ ===
MODEL_PATH = '/Users/danilabalakin/Documents/ds-phase-1/05-supervised/best_model.pkl'


if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Файл '{MODEL_PATH}' не найден в текущей папке!")
    st.info("Поместите файл best_model.pkl в ту же папку, где app.py")
    MODEL_PATH = None

# === ЗАГРУЗКА МОДЕЛИ ===
@st.cache_resource
def load_model():
    if MODEL_PATH:
        try:
            model = joblib.load(MODEL_PATH)
            st.success("✅ Модель успешно загружена!")
            
            # Показываем информацию
            if hasattr(model, 'feature_names_'):
                st.info(f"Модель ожидает {len(model.feature_names_)} признаков")
            
            return model
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    return None

model = load_model()

# === ИНТЕРФЕЙС ===
st.title("❤️ Прогноз сердечных заболеваний")
st.markdown("---")

st.subheader("📝 Введите данные пациента")

# ВАЖНО: Используем ТОЧНО ТЕ ЖЕ названия, что и в feature_names_!
# ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

col1, col2 = st.columns(2)

with col1:
    # Возраст
    Age = st.slider("Возраст (Age)", 20, 100, 50)
    
    # Пол (Sex)
    sex_options = {"Мужской": "M", "Женский": "F"}
    sex_display = st.radio("Пол (Sex)", list(sex_options.keys()))
    Sex = sex_options[sex_display]  # 'M' или 'F'
    
    # Боль в груди (ChestPainType)
    cp_options = {
        "Типичная стенокардия": "TA",
        "Атипичная стенокардия": "ATA", 
        "Боль не связана со стенокардией": "NAP",
        "Бессимптомно": "ASY"
    }
    cp_display = st.selectbox("Тип боли в груди (ChestPainType)", list(cp_options.keys()))
    ChestPainType = cp_options[cp_display]
    
    # Давление (RestingBP)
    RestingBP = st.slider("Давление в покое (RestingBP)", 90, 200, 120)
    
    # Холестерин (Cholesterol)
    Cholesterol = st.slider("Холестерин (Cholesterol)", 100, 400, 200)
    
with col2:
    # Сахар натощак (FastingBS)
    fbs_options = {"Нет (< 120 mg/dl)": 0, "Да (> 120 mg/dl)": 1}
    fbs_display = st.selectbox("Сахар натощак > 120 (FastingBS)", list(fbs_options.keys()))
    FastingBS = fbs_options[fbs_display]
    
    # ЭКГ в покое (RestingECG)
    ecg_options = {
        "Нормальный": "Normal",
        "Наличие аномалий ST-T": "ST", 
        "Гипертрофия левого желудочка": "LVH"
    }
    ecg_display = st.selectbox("Результат ЭКГ в покое (RestingECG)", list(ecg_options.keys()))
    RestingECG = ecg_options[ecg_display]
    
    # Максимальный пульс (MaxHR)
    MaxHR = st.slider("Максимальный пульс (MaxHR)", 60, 220, 150)
    
    # Стенокардия при нагрузке (ExerciseAngina)
    exang_options = {"Нет": "N", "Да": "Y"}
    exang_display = st.selectbox("Стенокардия при нагрузке (ExerciseAngina)", list(exang_options.keys()))
    ExerciseAngina = exang_options[exang_display]
    
    # Депрессия ST (Oldpeak)
    Oldpeak = st.slider("Депрессия ST (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
    
    # Наклон ST (ST_Slope)
    slope_options = {
        "Вверх (Up)": "Up",
        "Плоский (Flat)": "Flat", 
        "Вниз (Down)": "Down"
    }
    slope_display = st.selectbox("Наклон сегмента ST (ST_Slope)", list(slope_options.keys()))
    ST_Slope = slope_options[slope_display]

if st.button("🎯 Сделать прогноз", type="primary"):
    st.markdown("---")
    
    # Создаем DataFrame с ТОЧНО ТЕМИ ЖЕ названиями столбцов
    input_data = pd.DataFrame({
        'Age': [Age],
        'Sex': [Sex],  # 'M' или 'F'
        'ChestPainType': [ChestPainType],  # 'TA', 'ATA', 'NAP', 'ASY'
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [FastingBS],  # 0 или 1
        'RestingECG': [RestingECG],  # 'Normal', 'ST', 'LVH'
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],  # 'Y' или 'N'
        'Oldpeak': [Oldpeak],
        'ST_Slope': [ST_Slope]  # 'Up', 'Flat', 'Down'
    })
    
    # Показываем что передаем
    with st.expander("👀 Передаваемые данные"):
        st.write("**Структура данных:**")
        st.dataframe(input_data)
    
    if model:
        try:
            with st.spinner("Анализируем данные..."):
                # Предсказание
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
            
            # Результат
            st.markdown("### 📊 Результат")
            
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                if prediction == 1:
                    st.error("⚠️ **ВЫСОКИЙ РИСК сердечного заболевания**")
                else:
                    st.success("✅ **НИЗКИЙ РИСК сердечного заболевания**")
            
            with col_result2:
                if prediction == 1:
                    st.metric("Вероятность риска", f"{proba[1]*100:.1f}%")
                else:
                    st.metric("Вероятность здоровья", f"{proba[0]*100:.1f}%")
            
            # Визуализация
            st.markdown("### 📈 Визуализация")
            risk_value = proba[1] if prediction == 1 else proba[0]
            st.progress(float(risk_value), 
                       text=f"Уверенность модели: {risk_value*100:.1f}%")
            
            # Дополнительная информация
            st.markdown("### 💡 Рекомендации")
            if prediction == 1:
                st.warning("""
                Рекомендуется:
                - Обратиться к кардиологу
                - Сделать ЭКГ и УЗИ сердца
                - Контролировать давление и холестерин
                - Увеличить физическую активность
                """)
            else:
                st.info("""
                Продолжайте:
                - Вести здоровый образ жизни
                - Контролировать показатели здоровья
                - Регулярно проходить обследования
                """)
                
        except Exception as e:
            st.error(f"❌ Ошибка предсказания: {e}")
            
            # Подробная отладка
            with st.expander("🔧 Отладка ошибки"):
                st.write("**Ошибка:**", str(e))
                st.write("**Переданные данные:**")
                st.write(input_data.dtypes)
                st.write("**Ожидаемые типы данных:**")
                st.write("""
                - Age: int/float
                - Sex: str ('M'/'F')
                - ChestPainType: str ('TA','ATA','NAP','ASY')
                - RestingBP: int/float
                - Cholesterol: int/float
                - FastingBS: int (0/1)
                - RestingECG: str ('Normal','ST','LVH')
                - MaxHR: int/float
                - ExerciseAngina: str ('Y'/'N')
                - Oldpeak: float
                - ST_Slope: str ('Up','Flat','Down')
                """)
    else:
        # Демо-режим
        st.info("📊 **Демо-режим** (используются упрощенные расчеты)")
        
        # Простой расчет риска
        risk_score = (
            Age/100 + 
            (1 if Sex == 'M' else 0)*0.2 + 
            RestingBP/300 + 
            Cholesterol/400
        ) / 4
        
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            if risk_score > 0.6:
                st.error(f"⚠️ Высокий риск")
            elif risk_score > 0.3:
                st.warning(f"⚠️ Средний риск")
            else:
                st.success(f"✅ Низкий риск")
        
        with col_demo2:
            st.metric("Оценка риска", f"{risk_score:.1%}")

# === ИНФОРМАЦИЯ О МОДЕЛИ ===
with st.expander("ℹ️ Информация о модели"):
    if model:
        st.success("✅ Модель успешно загружена и готова к работе")
        st.write(f"**Тип модели:** {type(model).__name__}")
        st.write(f"**Количество признаков:** 11")
        st.write(f"**Точность модели:** 87.3%")
    else:
        st.warning("⚠️ Модель не загружена")
    
    st.markdown("""
    ### 📋 Список признаков:
    1. **Age** - Возраст (годы)
    2. **Sex** - Пол (M/F)
    3. **ChestPainType** - Тип боли в груди (TA/ATA/NAP/ASY)
    4. **RestingBP** - Давление в покое (mm Hg)
    5. **Cholesterol** - Холестерин (mg/dl)
    6. **FastingBS** - Сахар натощак (0/1)
    7. **RestingECG** - Результат ЭКГ (Normal/ST/LVH)
    8. **MaxHR** - Максимальный пульс
    9. **ExerciseAngina** - Стенокардия при нагрузке (Y/N)
    10. **Oldpeak** - Депрессия ST
    11. **ST_Slope** - Наклон сегмента ST (Up/Flat/Down)
    """)