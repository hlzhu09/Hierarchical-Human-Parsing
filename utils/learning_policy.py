import math

# poly lr
# def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
#     if method == 'poly':
#         current_step = epoch * iters_per_epoch + i_iter
#         max_step = args.epochs * iters_per_epoch
#         lr = args.learning_rate * ((1 - current_step / max_step) ** 0.9)
#     else:
#         lr = args.learning_rate
#     optimizer.param_groups[0]['lr'] = lr
#     return lr

def cosine_decay(base_learning_rate, global_step, warm_step, decay_steps, alpha=0.0001):
    # warm_step = 5 * iters_per_epoch
    # warm_lr = 0.01 * learning_rate
    # current_step = epoch * iters_per_epoch + i_iter
    alpha = alpha/base_learning_rate
    if global_step < warm_step:
        lr = base_learning_rate*global_step/warm_step
        # lr = base_learning_rate
    else:
        global_step = min(global_step, decay_steps)-warm_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / (decay_steps-warm_step)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = base_learning_rate * decayed
    return lr


def restart_cosine_decay(base_learning_rate, global_step, warm_step, decay_steps, alpha=0.0001):
    # warm_step = 5 * iters_per_epoch
    # warm_lr = 0.01 * learning_rate
    # current_step = epoch * iters_per_epoch + i_iter
    alpha = alpha/base_learning_rate
    restart_step = int((warm_step+decay_steps)/2)
    if global_step < warm_step:
        lr = base_learning_rate*global_step/warm_step
    elif global_step <restart_step:
        end_steps = restart_step-warm_step
        cur_step = global_step-warm_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / end_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = base_learning_rate * decayed
    else:
        end_steps = decay_steps - restart_step
        cur_step = min(global_step, decay_steps) - restart_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / end_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = base_learning_rate * decayed
    return lr


if __name__ == '__main__':
    epochs = 150
    total_steps = 150
    learning_rate_base = 0.007
    iters_per_epoch = 1
    lr_list = []
    for epoch in range(epochs):
        lr = restart_cosine_decay(base_learning_rate=0.007, global_step=epoch,
                      warm_step=10, decay_steps=epochs, alpha=0.0001)
        lr_list.append(lr)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.array(lr_list))
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('lr', fontsize=20)
    plt.axis([0, total_steps, 0, learning_rate_base*1.1])
    plt.xticks(np.arange(0, total_steps, 10))
    plt.grid()
    plt.title('Cosine decay with warmup', fontsize=20)
    plt.show()
