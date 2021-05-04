from matplotlib import pyplot as plt

def plot_c(c, alpha, threshold):
    p = [c]
    while c > threshold:
        c = c*alpha
        p.append(c)
    plt.plot(p)
    plt.show()
    