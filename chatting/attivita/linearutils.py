import matplotlib.pyplot as plt

def get_params(model, rounding=None):
    return [round(p.item(), rounding) if rounding else p.item() for p in reversed(list(model.parameters()))]

def plot(x, y, title=None, model=None):
    plt.close('all')
    xx, yy = x.numpy(), y.numpy()
    plt.plot(xx, yy, 'o')
    if title:
        plt.title(title)
    if model:
        [k0, k1] = get_params(model)
        plt.plot(xx, k0 + k1 * xx)
    plt.show()
