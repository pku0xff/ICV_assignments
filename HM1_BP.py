import numpy as np
from utils import read_img

if __name__ == "__main__":

    # input
    input_vector = np.zeros((10, 784))
    for i in range(10):
        input_vector[i, :] = read_img("mnist_subset/" + str(i) + ".png").reshape(-1) / 255.
    gt_y = np.zeros((10, 1))
    gt_y[0] = 1

    np.random.seed(14)

    # Intialization MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784, 16)
    MLP_layer_2 = np.random.randn(16, 1)
    lr = 1e-1
    loss_list = []

    '''
    (input) --> |MLP L1| --> (o1) --> |sigmoid| --> (o1a) --> 
                |MLP L2| --> (o2) --> |sigmoid| --> (pred) -->
                |CE Loss| --> (loss)
    '''

    for i in range(50):
        # Forward
        output_layer_1 = input_vector.dot(MLP_layer_1)  # 10*16
        output_layer_1_act = 1 / (1 + np.exp(-output_layer_1))  # sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)  # 10*1
        pred_y = 1 / (1 + np.exp(-output_layer_2))  # sigmoid activation function
        loss = -(gt_y * np.log(pred_y) + (1 - gt_y) * np.log(1 - pred_y)).sum()  # cross-entroy loss
        print(loss)
        print("iteration: %d, loss: %f" % (i + 1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        loss_to_pred = -gt_y / pred_y + (1 - gt_y) / (1 - pred_y)  # 10*1
        pred_to_o2 = pred_y * (1 - pred_y)  # 10*1
        o2_to_o1a = MLP_layer_2  # 16*1
        o1a_to_o1 = output_layer_1_act * (1 - output_layer_1_act)  # 10*16
        o1_to_input = MLP_layer_1  # 784*16

        loss_to_o2 = loss_to_pred * pred_to_o2  # 10*1
        grad_layer_2 = output_layer_1_act.T @ loss_to_o2  # 16*10 @ 10*1 = 16*1

        loss_to_o1a = loss_to_o2 @ o2_to_o1a.T  # 10*16
        loss_to_o1 = loss_to_o1a * o1a_to_o1  # 10*16
        grad_layer_1 = input_vector.T @ loss_to_o1  # 784*10

        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2

    np.savetxt("result/HM1_BP.txt", loss_list)
