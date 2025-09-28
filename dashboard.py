# dashboard.py
import streamlit as st
import pandas as pd
import re

st.title("📊 داشبورد Observability - CNN مدل CIFAR10")

log_file = "app.log"

try:
    with open(log_file, "r") as f:
        logs = f.readlines()
except FileNotFoundError:
    st.error("❌ فایل app.log پیدا نشد. اول app.py را اجرا کن و چند درخواست بده.")
    st.stop()

data = []
for line in logs:
    if "Prediction" in line:
        match = re.search(r"Prediction: (\w+), Latency=([\d.]+)s", line)
        if match:
            label, latency = match.groups()
            data.append({
                "label": label,
                "latency": float(latency)
            })

if data:
    df = pd.DataFrame(data)
    st.subheader("📄 آخرین درخواست‌ها")
    st.write(df.tail(10))
    st.subheader("⏱️ نمودار Latency")
    st.line_chart(df["latency"])
    st.success("✅ داشبورد آپدیت شد.")
else:
    st.warning("هیچ داده‌ای در لاگ‌ها پیدا نشد.")
