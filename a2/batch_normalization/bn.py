'''
JOB LOG:
反向传播 x_hat 对 x 求导时，miu & sigma 也是关于 x 的函数，需要对x求导
gamma, beta 也和其他参数一样通过反向传播学习得到
test time的 mean & var 则是训练时 mean & var 的积累更新
'''
import numpy as np
import cv2

# eps may help you to deal with numerical problem
eps = 1e-5


def bn_forward_test(x, gamma, beta, mean, var):
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out


def bn_forward_train(x, gamma, beta):
    '''
    x: input data, B*C
    gamma, beta: C
    mean, var: C
    '''
    # print_info(x)
    sample_mean = np.mean(x, axis=0)  # miu
    sample_var = np.var(x, axis=0)  # sigma
    # print(sample_mean, sample_var)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta

    # save intermidiate variables for computing the gradient when backward
    cache = (gamma, x, sample_mean, sample_var, x_hat)
    return out, cache


def bn_backward(dout, cache):
    '''
    dout: B*C
    dx: B*C
    dgamma: C
    dbeta: C
    '''
    gamma, x, sample_mean, sample_var, x_hat = cache
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma
    N = x.shape[0]
    std = np.sqrt(sample_var + eps)
    dsigma2 = np.sum(dx_hat * (x - sample_mean) / (-2 * std ** 3), axis=0)
    dmiu = np.sum(-dx_hat / std, axis=0)
    dx = dx_hat / std + dmiu / N + dsigma2 * 2 * (x - sample_mean) / N
    return dx, dgamma, dbeta


# This function may help you to check your code
def print_info(x):
    print('mean:', np.mean(x, axis=0))
    print('var:', np.var(x, axis=0))
    print('------------------')
    return


if __name__ == "__main__":

    # input data
    train_data = np.zeros((9, 784))
    for i in range(9):
        train_data[i, :] = cv2.imread("mnist_subset/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.
    gt_y = np.zeros((9, 1))
    gt_y[0] = 1

    val_data = np.zeros((1, 784))
    val_data[0, :] = cv2.imread("mnist_subset/9.png", cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.
    val_gt = np.zeros((1, 1))

    np.random.seed(14)

    # Intialize MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784, 16)
    MLP_layer_2 = np.random.randn(16, 1)

    # Initialize gamma and beta
    gamma = np.random.randn(16)
    beta = np.random.randn(16)

    lr = 1e-1
    loss_list = []

    # compute mean and var for testing
    # add codes anywhere as you need
    mean = 0
    var = 0
    rho = 0.8

    # training 
    for i in range(50):
        # Forward
        output_layer_1 = train_data.dot(MLP_layer_1)
        output_layer_1_bn, cache = bn_forward_train(output_layer_1, gamma, beta)
        output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn))  # sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1 + np.exp(-output_layer_2))  # sigmoid activation function

        # update mean and var
        _, _, sample_mean, sample_var, _ = cache
        mean = rho * mean + (1 - rho) * sample_mean
        var = rho * var + (1 - rho) * sample_var

        # compute loss 
        loss = -(gt_y * np.log(pred_y) + (1 - gt_y) * np.log(1 - pred_y)).sum()
        print("iteration: %d, loss: %f" % (i + 1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        grad_pred_y = -(gt_y / pred_y) + (1 - gt_y) / (1 - pred_y)
        grad_activation_func = grad_pred_y * pred_y * (1 - pred_y)
        grad_layer_2 = output_layer_1_act.T.dot(grad_activation_func)
        grad_output_layer_1_act = grad_activation_func.dot(MLP_layer_2.T)
        grad_output_layer_1_bn = grad_output_layer_1_act * (1 - output_layer_1_act) * output_layer_1_act
        grad_output_layer_1, grad_gamma, grad_beta = bn_backward(grad_output_layer_1_bn, cache)
        grad_layer_1 = train_data.T.dot(grad_output_layer_1)

        # update parameters
        gamma -= lr * grad_gamma
        beta -= lr * grad_beta
        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2

    # validate
    output_layer_1 = val_data.dot(MLP_layer_1)
    output_layer_1_bn = bn_forward_test(output_layer_1, gamma, beta, mean, var)
    output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn))  # sigmoid activation function
    output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
    pred_y = 1 / (1 + np.exp(-output_layer_2))  # sigmoid activation function
    loss = -(val_gt * np.log(pred_y) + (1 - val_gt) * np.log(1 - pred_y)).sum()
    print("validation loss: %f" % (loss))
    loss_list.append(loss)

    np.savetxt("../results/bn_loss.txt", loss_list)
