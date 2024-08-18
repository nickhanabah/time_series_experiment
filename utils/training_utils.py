import torch
from utils.model_utils import ARNet, set_seed, normal_loss, MoE
from torch.utils.data import DataLoader
from utils.data_utils import TimeSeriesDataset
from utils.metrics import metric

def train_expert_or_moe(
        net,
        train_data:DataLoader, 
        modelling_task: str, 
        n_continous_features:int, 
        batch_size:int, 
        val_loss_list:list, 
        val_data:DataLoader, 
        train_loss_list:list, 
        i:int, 
        density: bool = False, 
        num_of_experts: int = 2, 
        epochs:int = 2,
        moe: bool = False): 
    
    if moe == False: 
        print(f'Started training expert {i +1}/{num_of_experts}')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        train_counter = 0
        val_counter = 0
        running_train_loss = 0.0
        running_val_loss = 0.0
        running_train_mae = 0.0
        running_train_mse = 0.0
        running_val_mae = 0.0
        running_val_mse = 0.0

        if epoch + 1 != 1 and (epoch + 1) % 2 == 0:
            learning_rate = learning_rate / 2
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        print(f"Current learning rate is : {learning_rate}")
        print("---------------------------")
        for i, data in enumerate(train_data):
            inputs, labels = data
            labels = labels.squeeze(0).float()

            optimizer.zero_grad()
            outputs = net(inputs)
            if density:
                loss = normal_loss(outputs, labels.squeeze(1))
            else:
                if modelling_task == "multivariate":
                    loss = net.criterion(
                        outputs,
                        labels[:, 0:(n_continous_features), :].reshape(outputs.shape),
                        )
                else:
                    loss = net.criterion(outputs, labels.squeeze(1))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            if density == False:
                outputs_array = outputs.detach().cpu().numpy()
                labels_array = labels.squeeze(2).detach().cpu().numpy()
                mae, mse = metric(pred=outputs_array, true=labels_array)
                running_train_mae += mae
                running_train_mse += mse

            running_train_loss += loss.item()
            train_counter += batch_size
            if train_counter % 5000 == 0:
                print(
                    f"Current (running) training loss at iteration {train_counter} : {running_train_loss/train_counter}"
                    )
        train_loss_list.append(running_train_loss / train_counter)

        if density == False:
            for i, data in enumerate(val_data):
                inputs, test_labels = data
                test_labels = test_labels.squeeze(0).float()
                output = net(inputs)
                if modelling_task == "multivariate":
                    val_loss = net.criterion(
                        outputs,
                        test_labels[:, 0:(n_continous_features), :].reshape(
                            outputs.shape
                            ),
                        )
                else:
                    val_loss = net.criterion(output, test_labels.squeeze(1))

                running_val_loss += val_loss.item()

                output_array = output.detach().cpu().numpy()
                test_labels_array = test_labels.squeeze(2).detach().cpu().numpy()
                mae, mse = metric(pred=output_array, true=test_labels_array)
                running_val_mae += mae
                running_val_mse += mse
                val_counter += batch_size
            val_loss_list.append(running_val_loss / val_counter)

            if epoch % 1 == 0:
                print("")
                print(f"Epoch {epoch}: ")
                print("")
                print("Train metrics: -------")
                print(f"Running (training) loss is {running_train_loss/train_counter}.")
                print(f"Training MAE is {running_train_mae/train_counter}.")
                print(f"Training MSE is {running_train_mse/train_counter}.")
                print("")
                print("Test metrics: -------")
                print(f"Running (test) loss is {running_val_loss/val_counter}.")
                print(f"Test MAE is {running_val_mae/val_counter}.")
                print(f"Test MSE is {running_val_mse/val_counter}.")
                print("---------------------------")
    return net

def train(
    epochs,
    p_lag,
    future_steps,
    n_continous_features,
    n_categorial_features,
    training_df,
    validation_df,
    feature_columns,
    dataset_name: str,
    target_column=["OT"],
    learning_rate=1.0e-4,
    decomp_kernel_size=7,
    batch_size=8,
    model="rlinear",
    moe: bool = False, 
    num_of_experts: int = 1, 
    modelling_task="univariate",
    density=False,
    depth = 'shallow'
):

    #set_seed()
    untrained_experts = []
    for _ in range(num_of_experts): 
        net = ARNet(
                p_lag=p_lag,
                n_continous_features=n_continous_features,
                n_categorial_features=n_categorial_features,
                future_steps=future_steps,
                decomp_kernel_size=decomp_kernel_size,
                batch_size=batch_size,
                model=model,
                modelling_task=modelling_task,
                density=density,
                depth = depth
            )
        untrained_experts.append(net)

    train_data = DataLoader(
        TimeSeriesDataset(
            training_df,
            future_steps=future_steps,
            feature_columns=feature_columns,
            target_column=target_column,
            p_lag=p_lag,
            modelling_task=modelling_task,
        ),
        batch_size=batch_size,
        drop_last=True,
    )
    train_loss_list = []
    val_data = DataLoader(
        TimeSeriesDataset(
            validation_df,
            future_steps=future_steps,
            feature_columns=feature_columns,
            target_column=target_column,
            p_lag=p_lag,
            modelling_task=modelling_task,
        ),
        batch_size=batch_size,
        drop_last=True,
    )
    val_loss_list = []

    torch.set_grad_enabled(True)
    
    trained_experts = []
    for i in range(num_of_experts): 
        net = train_expert_or_moe(
                 num_of_experts=num_of_experts, 
                 net=untrained_experts[i], 
                 epochs=epochs, 
                 train_data=train_data, 
                 density=density, 
                 modelling_task=modelling_task, 
                 n_continous_features=n_continous_features, 
                 batch_size=batch_size, 
                 val_loss_list=val_loss_list, 
                 val_data=val_data, 
                 train_loss_list=train_loss_list, 
                 i=i, 
                 moe = False)
        trained_experts.append(net)
    if moe: 
        moe_model = MoE(trained_experts)
        moe_model = train_expert_or_moe(
            num_of_experts=num_of_experts, 
            net=moe_model, 
            epochs=epochs, 
            train_data=train_data, 
            density=density, 
            modelling_task=modelling_task, 
            n_continous_features=n_continous_features, 
            batch_size=batch_size, 
            val_loss_list=val_loss_list, 
            val_data=val_data, 
            train_loss_list=train_loss_list, 
            i=0, 
            moe = True)
        return moe_model
    
    else: 
        return net
