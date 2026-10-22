

import streamlit as st
# Streamlit برای ساخت صفحه گرافیکی و نمایش برنامه در مرورگر است.

import pandas as pd
# Pandas برای ساخت و مدیریت جدول‌های داده استفاده می‌شود.

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier یک الگوریتم یادگیری ماشین برای طبقه‌بندی است.
# در این برنامه از آن برای پیش‌بینی نوع خرابی استفاده می‌کنیم.


# -----------------------------
#  تنظیمات صفحه
# -----------------------------

st.set_page_config(
    page_title="Truck Arvin Diagnostic",  # عنوان صفحه در مرورگر
    page_icon="🚛",                       # آیکون صفحه
    layout="wide"                         # استفاده از عرض کامل صفحه
)


# -----------------------------
# طراحی ظاهری با CSS
# -----------------------------

st.markdown("""
<style>

/* رنگ پس‌زمینه بخش اصلی صفحه */
.main {
    background-color: #0f172a;
}

/* فاصله بالای محتوای صفحه */
.block-container {
    padding-top: 2rem;
}

/* طراحی عنوان اصلی */
.title {
    font-size: 42px;       /* اندازه نوشته */
    font-weight: bold;     /* ضخیم کردن نوشته */
    color: #38bdf8;        /* رنگ آبی روشن */
}

/* طراحی توضیح زیر عنوان */
.subtitle {
    font-size: 18px;
    color: #94a3b8;
}

/* طراحی کادر نتیجه */
.result-box {
    padding: 25px;             /* فاصله متن از لبه کادر */
    border-radius: 15px;       /* گرد کردن گوشه‌ها */
    background-color: #1e293b;
    text-align: center;        /* وسط‌چین کردن متن */
    margin-top: 20px;          /* فاصله از بالای کادر */
}

/* رنگ و اندازه پیام وضعیت عادی */
.normal {
    color: #22c55e;       /* سبز */
    font-size: 32px;
    font-weight: bold;
}

/* رنگ و اندازه پیام هشدار */
.warning {
    color: #ef4444;       /* قرمز */
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
# st.markdown متن یا HTML و CSS را نمایش می‌دهد.
# unsafe_allow_html=True اجازه می‌دهد HTML و CSS سفارشی اعمال شوند.


# -----------------------------
# 4. ساخت داده‌های آموزشی
# -----------------------------

# این داده‌ها نمونه هستند و از کامیون واقعی جمع‌آوری نشده‌اند.
# هر ردیف شامل اطلاعات سنسورها و برچسب نوع خرابی است.

data = {

    # دمای موتور بر حسب درجه سانتی‌گراد
    "engine_temp": [
        85, 88, 90, 92, 87,
        105, 110, 115, 108, 112,
        90, 92, 95, 88, 91,
        85, 87, 89, 86, 90,
        100, 102, 98, 105, 110
    ],

    # فشار روغن بر حسب bar
    "oil_pressure": [
        4.2, 4.0, 4.1, 3.9, 4.3,
        2.0, 1.8, 1.6, 2.1, 1.7,
        3.8, 4.0, 3.7, 3.9, 4.1,
        4.2, 4.0, 4.1, 4.3, 4.0,
        2.5, 2.2, 2.8, 2.0, 1.9
    ],

    # ولتاژ باتری
    "battery_voltage": [
        24.2, 24.5, 24.1, 24.3, 24.6,
        24.0, 23.8, 23.5, 23.7, 23.6,
        24.4, 24.2, 24.5, 24.3, 24.1,
        24.6, 24.4, 24.2, 24.5, 24.3,
        23.5, 23.7, 23.4, 23.6, 23.3
    ],

    # فشار سوخت بر حسب bar
    "fuel_pressure": [
        5.0, 5.2, 5.1, 4.9, 5.0,
        4.8, 4.7, 4.6, 4.5, 4.7,
        5.1, 5.0, 5.2, 4.9, 5.1,
        5.0, 5.2, 5.1, 5.0, 5.1,
        4.0, 4.2, 4.1, 3.9, 4.0
    ],

    # دور موتور بر حسب RPM
    "rpm": [
        1500, 1600, 1550, 1700, 1650,
        1800, 1850, 1900, 1750, 1880,
        1600, 1650, 1550, 1700, 1600,
        1500, 1550, 1600, 1650, 1700,
        2100, 2200, 2150, 2250, 2300
    ],

    # برچسبی که مدل باید یاد بگیرد
    # Normal یعنی عادی
    # Cooling System یعنی سیستم خنک‌کننده
    # Engine یعنی موتور
    "failure": [
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "Cooling System", "Cooling System", "Cooling System",
        "Cooling System", "Cooling System",
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "Engine", "Engine", "Engine", "Engine", "Engine"
    ]
}


# -----------------------------
# 5. تبدیل داده‌ها به جدول
# -----------------------------

df = pd.DataFrame(data)
# دیکشنری data را به یک جدول Pandas تبدیل می‌کند.
# هر ردیف یک نمونه آموزشی است.


# -----------------------------
# 6. جدا کردن ورودی‌ها و پاسخ‌ها
# -----------------------------

X = df[
    [
        "engine_temp",       # دمای موتور
        "oil_pressure",      # فشار روغن
        "battery_voltage",   # ولتاژ باتری
        "fuel_pressure",     # فشار سوخت
        "rpm"                # دور موتور
    ]
]

# X شامل اطلاعات سنسورها است.
# مدل با استفاده از این ستون‌ها الگوها را یاد می‌گیرد.

y = df["failure"]
# y شامل جواب‌های آموزشی است.
# یعنی مشخص می‌کند هر نمونه با چه نوع خرابی برچسب خورده است.


# -----------------------------
# 7. ساخت مدل هوش مصنوعی
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,  # تعداد درخت‌های تصمیم در مدل
    random_state=42    # ثابت نگه داشتن تصادفی‌سازی برای تکرارپذیری
)

# آموزش مدل با اطلاعات سنسورها و برچسب‌های آموزشی
model.fit(X, y)

# بعد از این خط، مدل برای پیش‌بینی نمونه‌های جدید آماده است.


# -----------------------------
# 8. نمایش عنوان صفحه
# -----------------------------

st.markdown(
    '<div class="title">🚛 Truck Arvin Diagnostic</div>',
    unsafe_allow_html=True
)
# عنوان بزرگ برنامه را نمایش می‌دهد.

st.markdown(
    '<div class="subtitle">'
    'Arvin-powered heavy truck failure classification'
    '</div>',
    unsafe_allow_html=True
)
# زیرعنوان برنامه را نمایش می‌دهد.

st.divider()
# یک خط جداکننده روی صفحه رسم می‌کند.


# -----------------------------
# 9. ساخت ورودی‌های کناری
# -----------------------------

st.sidebar.header("⚙️ Truck Sensors")
# عنوان پنل کناری را نمایش می‌دهد.


# دریافت دمای موتور از کاربر
engine_temp = st.sidebar.number_input(
    "🌡️ Engine Temperature (°C)",
    min_value=50.0,  # کمترین مقدار قابل ورود
    max_value=150.0, # بیشترین مقدار قابل ورود
    value=90.0       # مقدار پیش‌فرض
)


# دریافت فشار روغن
oil_pressure = st.sidebar.number_input(
    "🛢️ Oil Pressure (bar)",
    min_value=0.0,# کمترین مقدار قابل ورود
    max_value=6.0,#بیشترین مقدار قابل ورود
    value=4.0     #مقدار پیش فرض 
)


# دریافت ولتاژ باتری
battery_voltage = st.sidebar.number_input(
    "🔋 Battery Voltage (V)",
    min_value=18.0,#کمترین مقدار قابل ورود
    max_value=30.0,# بیشترین مقدار قابل ورود
    value=24.0     #مقدار پیش فرض
)


# دریافت فشار سوخت
fuel_pressure = st.sidebar.number_input(
    "⛽ Fuel Pressure (bar)",
    min_value=0.0,#کمترین مقدار قابل ورود
    max_value=8.0,#بیشترین مقدار قابل ورود
    value=5.0#مقدار پیش فرض
)


# دریافت دور موتور
rpm = st.sidebar.number_input(
    "⚙️ Engine RPM",
    min_value=500,
    max_value=4000,
    value=1600
)


# ساخت دکمه شروع تحلیل
analyze = st.sidebar.button(
    "🔍 ANALYZE TRUCK",
    use_container_width=True  # استفاده از تمام عرض پنل
)

# اگر کاربر روی دکمه کلیک کند، مقدار analyze برابر True می‌شود.


# -----------------------------
# 10. نمایش داشبورد سنسورها
# -----------------------------

# صفحه را به پنج ستون تقسیم می‌کنیم.
col1, col2, col3, col4, col5 = st.columns(5)


# نمایش دمای موتور
with col1:
    st.metric("Temperature", f"{engine_temp:.1f} °C")
    # :.1f عدد را با یک رقم اعشار نمایش می‌دهد.


# نمایش فشار روغن
with col2:
    st.metric("Oil Pressure", f"{oil_pressure:.1f} bar")


# نمایش ولتاژ باتری
with col3:
    st.metric("Battery", f"{battery_voltage:.1f} V")


# نمایش فشار سوخت
with col4:
    st.metric("Fuel Pressure", f"{fuel_pressure:.1f} bar")


# نمایش دور موتور
with col5:
    st.metric("RPM", rpm)


st.divider()
# جدا کردن داشبورد سنسورها از بخش نتایج.


# -----------------------------
# 11. پیش‌بینی نوع خرابی
# -----------------------------

# این بخش فقط پس از کلیک روی دکمه اجرا می‌شود.
if analyze:

    # تبدیل مقادیر واردشده به جدولی با همان ستون‌های آموزشی
    input_data = pd.DataFrame({
        "engine_temp": [engine_temp],
        "oil_pressure": [oil_pressure],
        "battery_voltage": [battery_voltage],
        "fuel_pressure": [fuel_pressure],
        "rpm": [rpm]
    })

    # پیش‌بینی کلاس خرابی برای ورودی جدید
    prediction = model.predict(input_data)[0]
    # predict یک آرایه برمی‌گرداند؛ [0] اولین نتیجه را برمی‌دارد.

    # دریافت امتیازهای خروجی مدل برای کلاس‌های مختلف
    probabilities = model.predict_proba(input_data)[0]

    # پیدا کردن بزرگ‌ترین امتیاز و تبدیل آن به درصد
    confidence = max(probabilities) * 100

    # شروع کادر نتیجه
    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )


    # اگر نتیجه مدل Normal باشد
    if prediction == "Normal":

        st.markdown(
            '<div class="normal">✅ TRUCK STATUS: NORMAL</div>',
            unsafe_allow_html=True
        )

        # نمایش پیام سبز
        st.success(
            "No major problem detected by the System model."
        )


    # اگر نتیجه مدل یک کلاس دیگر باشد
    else:

        st.markdown(
            '<div class="warning">⚠️ PROBLEM DETECTED</div>',
            unsafe_allow_html=True
        )

        # نمایش نام دسته پیش‌بینی‌شده
        st.error(
            f"Possible problem: {prediction}"
        )


    # نمایش امتیاز مدل
    st.write(
        f"### System Confidence: {confidence:.1f}%"
    )

    # نمایش نوار درصد
    st.progress(
        int(confidence)
        # st.progress عددی بین صفر و صد دریافت می‌کند.
    )

    # پایان کادر نتیجه
    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------
    # 12. نمایش جزئیات پیش‌بینی
    # -----------------------------

    st.subheader("📊 System Analysis")
    # عنوان بخش جزئیات تحلیل


    # ساخت جدول شامل کلاس‌ها و امتیاز هر کلاس
    result_df = pd.DataFrame({
        "Failure Type": model.classes_,
        "Probability": probabilities
    })

    # تبدیل امتیازها از عدد اعشاری به درصد
    # برای مثال 0.75 تبدیل به 75.00 می‌شود.
    result_df["Probability"] = (
        result_df["Probability"] * 100
    ).round(2)

    # مرتب‌سازی از بیشترین امتیاز به کمترین
    result_df = result_df.sort_values(
        "Probability",
        ascending=False
    )

    # نمایش جدول نتایج در صفحه
    st.dataframe(
        result_df,
        use_container_width=True,  # پر کردن عرض صفحه
        hide_index=True            # مخفی کردن شماره ردیف‌ها
    )


# -----------------------------
# 13. پیام پیش از تحلیل
# -----------------------------

else:

    # این پیام تا پیش از کلیک روی دکمه نمایش داده می‌شود.
    st.info(
        "👈 Enter the truck sensor values and click "
        "**ANALYZE TRUCK** to start the system diagnosis."
    )


# -----------------------------
# 14. پایین صفحه
# -----------------------------

st.divider()
# رسم خط جداکننده در انتهای صفحه.

st.caption(
    "Truck Arvin Diagnostic • Python + Scikit-learn + Streamlit"
)
# نمایش نوشته کوچک در پایین صفحه.

