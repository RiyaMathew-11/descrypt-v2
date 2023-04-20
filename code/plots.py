import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv('./results/bert-base-multilingual-cased-1.csv')
data2 = pd.read_csv('./results/bert-base-multilingual-cased-2.csv')
# Sample data
result_set_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]

result_set_y1 = list(data1.score)
result_set_y2 = list(data2.score)

plt.plot(result_set_x, result_set_y1, label='Without Siamese', marker='o')
plt.plot(result_set_x, result_set_y2, label='With Siamese', marker='o')

plt.title('MODEL : bert-base-multilingual-cased')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.legend()

# Save the plot
plt.savefig('./plots/model-bert-base-multilingual-cased.png', dpi=300)

