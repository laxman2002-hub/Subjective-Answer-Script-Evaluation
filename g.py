import matplotlib.pyplot as plt
x = [5]
y1 = [100-16.67]

y2 = [100-5.56]

bar_width = 2

plt.bar(x, y1, color='blue', width=bar_width, label='Reference model accuracy')
plt.bar([xi + bar_width for xi in x], y2, color='red', width=bar_width, label='Our model accuracy')

# Adding text labels on top of each bar
for xi, yi in zip(x, y1):
    plt.text(xi, yi + 0.5, str(yi), ha='center', va='bottom', color='black')

for xi, yi in zip(x, y2):
    plt.text(xi + bar_width, yi + 0.5, str(yi), ha='center', va='bottom', color='black')


# print(y_total)
# plt.xlabel()
plt.ylabel('Accuracy for above example')

plt.legend()
plt.savefig('accuracy.png')
plt.show()