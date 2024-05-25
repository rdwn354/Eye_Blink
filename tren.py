import numpy as np
import matplotlib.pyplot as plt

# Data yang diberikan
data = [20, 15, 19, 10, 47, 39, 20, 22, 13, 19, 29, 28, 18, 27, 28, 26, 25, 19, 40, 47, 22, 5, 6, 6, 7, 13, 22, 14, 28, 23]

# Menghitung garis tren
x = np.arange(len(data))
y = np.array(data)
slope, intercept = np.polyfit(x, y, 1)

# Menentukan arah tren
if slope > 0:
    trend_direction = "up"
elif slope < 0:
    trend_direction = "down"
else:
    trend_direction = "flat"

# Membuat data untuk garis tren
trendline = intercept + slope * x

# Visualisasikan data dan garis tren
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Data')
plt.plot(x, trendline, label=f'Garis Tren ({trend_direction})', color='red', linewidth = 1)
plt.xlabel('Index')
plt.ylabel('Nilai')
plt.title('Data dengan Garis Tren')
plt.legend()
plt.show()

# Cetak hasil
print(f"The trend line is going {trend_direction}.")
