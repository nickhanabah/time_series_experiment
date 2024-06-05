import matplotlib.pyplot as plt
from utils.metrics import metric


def plot_multistep_forecast(
    test_data, neural_net, future_steps, number_of_forecasts=100
):
    output_list = []
    target_list = []
    for i, data in enumerate(test_data):
        inputs, labels = data
        output = neural_net(inputs)
        if i > number_of_forecasts:
            break
        else:
            if future_steps > 1:
                [output_list.append(out) for out in output.tolist()]
            else:
                [output_list.append(out[0]) for out in output.tolist()]

        [target_list.append(tar) for tar in labels.squeeze(1, 2).tolist()]

    target = []
    if future_steps > 1:
        for i in range(len(target_list)):
            if i == 0:
                target = target_list[i]
            else:
                target.append(target_list[i][len(target_list[i]) - 1])
        if future_steps > 0:
            alpha = 1
        if future_steps > 10:
            alpha = 0.5
        if future_steps > 30:
            alpha = 0.4
        if future_steps > 50:
            alpha = 0.3
        if future_steps > 80:
            alpha = 0.2
        if future_steps > 110:
            alpha = 0.1
    else:
        target = target_list
        alpha = 1

    fig = plt.figure(figsize=(15, 15))
    if future_steps > 1:
        plt.plot(
            range(0, len(target)), target, "g", label="target time series", alpha=0.9
        )
        for i, output in enumerate(output_list, start=0):
            if i == 0:
                plt.plot(
                    range(i, i + future_steps),
                    output,
                    color="#F39C12",
                    linewidth=1,
                    linestyle=".",
                    alpha=alpha,
                    label="pred time series" + "\n" + f"{future_steps} each",
                )
            else:
                plt.plot(
                    range(i, i + future_steps),
                    output,
                    color="#F39C12",
                    linewidth=1,
                    linestyle=".",
                    alpha=alpha,
                )
    else:
        plt.plot(
            range(0, len(target)), target, "g", label="target time series", alpha=0.9
        )
        plt.plot(
            range(0, len(output_list)),
            output_list,
            color="#F39C12",
            linewidth=1,
            linestyle=".",
            alpha=alpha,
            label="pred time series" + "\n" + f"{future_steps} each",
        )

    plt.title(
        f"Prediction plot for {number_of_forecasts} forecasts of {neural_net.model}, {future_steps} each"
    )
    plt.legend(loc="upper left")
    plt.xlabel("Time Steps")
    plt.ylabel("Oil Temparature (Target variable)")

    plt.savefig(
        f"/workspaces/time_series_experiment/plots/{neural_net.model}_{future_steps}fs_{number_of_forecasts}fcs_{neural_net.model}plag.png"
    )
    plt.savefig(
        f"/workspaces/time_series_experiment/plots/{neural_net.model}_{future_steps}fs_{number_of_forecasts}fcs_{neural_net.model}plag.pdf"
    )
