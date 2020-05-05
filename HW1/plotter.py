import matplotlib
import matplotlib.pyplot as plt

def plot(test_accuracies):
  plt.plot(range(len(test_accuracies)),test_accuracies)
  plt.xlabel("Step x100")
  plt.ylabel("Test accuracy")
  plt.show()