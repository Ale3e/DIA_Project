import  matplotlib.pyplot as plt

prices = []

for i in range(200, 600, 25):
    prices.append(i)

probs = [5, 30, 50, 55, 55, 60, 60, 55, 55, 50, 45, 40, 35, 30, 20, 10, 5]

plt.figure(0)
plt.ylabel("Conversion Rate")
plt.xlabel("Price")
plt.xticks([i for i in range(0, len(prices))], prices)
plt.plot(probs)
plt.plot([i - 10 for i in probs])
plt.legend(["General", "Worker", "Student", "Retired"])
plt.show()
