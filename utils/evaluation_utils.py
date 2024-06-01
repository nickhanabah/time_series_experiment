import matplotlib.pyplot as plt
from utils.metrics import metric

def plot_multistep_forecast(test_data, neural_net, future_steps, number_of_forecasts= 100): 
    output_list = []
    target_list = []
    for i, data in enumerate(test_data):
        inputs, labels = data
        output = neural_net(inputs)
        if i > number_of_forecasts: 
            break
        if future_steps > 1: 
            [output_list.append(out) for out in output.tolist()]
        else: 
            [output_list.append(out[0]) for out in output.tolist()]

        [target_list.append(tar) for tar in labels.squeeze(1,2).tolist()]

    target = []
    if future_steps > 1: 
        for i in range(len(target_list)): 
            if i == 0: 
                target = target_list[i]
            else: 
                target.append(target_list[i][len(target_list[i])-1])
        alpha = 0.2
    else: 
        target = target_list
        alpha = 1
    
    fig = plt.figure(figsize=(15, 15))
    if future_steps > 1: 
        plt.plot(range(0, len(target)), target, 'g', label='target time series', alpha=0.9)
        for i, output in enumerate(output_list, start=0): 
            if i == 0:
                plt.plot(range(i, i +future_steps), output, color='#F39C12',linewidth=1, linestyle='-.',alpha=alpha, label='pred time series' + "\n" + f'{future_steps} each')
            else: 
                plt.plot(range(i, i +future_steps), output, color='#F39C12',linewidth=1, linestyle='-.',alpha=alpha)
    else:
        print(len(target))
        print(len(output_list))
        plt.plot(range(0, len(target)), target, 'g', label='target time series', alpha=0.9)
        plt.plot(range(0, len(output_list)), output_list, color='#F39C12',linewidth=1, linestyle='-.',alpha=alpha)

    plt.legend(loc="upper left")
    plt.xlabel("Time Steps")
    plt.ylabel("Oil Temparature (Target variable)")

def evaluate_on_test_data(test_data, neural_net): 
    running_test_mae  = 0.
    running_test_mse  = 0.
    running_test_mape = 0.
    test_counter = 0

    for i, data in enumerate(test_data):
        inputs, test_labels = data
        test_labels = test_labels.squeeze(0).float()
        output = neural_net(inputs)
        output_array = output.detach().cpu().numpy()
        test_labels_array = test_labels.squeeze(2).detach().cpu().numpy()
        mae, mse, mape= metric(pred=output_array, true=test_labels_array)
        running_test_mae  += mae
        running_test_mse  += mse
        running_test_mape += mape
        test_counter += neural_net.batch_size

    print("Test metrics: -------")
    print(f"Validation MAE is {running_test_mae/test_counter}.")
    print(f"Validation MSE is {running_test_mse/test_counter}.")
    print(f"Validation MAPE is {running_test_mape/test_counter}.")
    print("---------------------------")