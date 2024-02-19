import environment as env
import matplotlib.pyplot as plt

def show_board(envi):
    plt.figure()
    plt.imshow(envi.board_image())
    plt.show()

envi = env.Environment()

print(envi.move([30, 26]))
print(envi.move([15, 20]))
print(envi.move([26, 15]))
show_board(envi)