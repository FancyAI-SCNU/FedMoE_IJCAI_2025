import torch
from torch import nn
from torch.distributions import Normal

from data.utils import infor_entropy, l2loss, moe_test, infor_entropy_nb
import torch.nn.functional as F


def fedmoe_train(model, scaler, dataset, batch_size, optimizer, epoch, entropy_loss_weight, aux_weight, l2_weight, device_index,
                 entropy_loss_flag=True, flag=0):
    tensor = torch.zeros(model.num_experts, model.output_size, device=device_index)  # 记录数据在专家上的分布情况
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    l2_loss = l2loss()
    device = torch.device(device_index)
    gate_count = {i: 0 for i in range(model.num_experts)}
    gate_prob = {i: 0 for i in range(model.num_experts)}
    gates_list = []
    flag = flag
    model.train()
    model.to(device)
    model_CE_loss = 0.0
    model_aux_loss = 0.0
    model_l2_loss = 0.0
    model_entropy_loss = 0.0
    for e in range(epoch):

        for i, data in enumerate(trainloader, 0):
            if model.epsilon > model.epsilon_min and model.random_gate:
                model.epsilon -= model.epsilon_decay
            # get the inputs; data is a list of [inputs, labels]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            optimizer.zero_grad()
            # inputs = inputs.view(inputs.shape[0], -1)
            outputs, aux_loss, gates = model(input_ids)
            if e == epoch - 1:
                for _, (gate, label) in enumerate(zip(gates, labels)):
                    if label.item() == flag:
                        gates_list.append({'label': label.item(), 'gate': gate.detach()})
                        flag = flag + 1
            aux_loss = aux_loss * aux_weight
            loss = criterion(outputs, labels)
            reg_loss = l2_loss(model) * l2_weight

            model_CE_loss += loss.item()
            model_aux_loss += aux_loss.item()
            model_l2_loss += reg_loss.item()

            if entropy_loss_flag:
                entropy_loss = infor_entropy(gates, labels, outputs.size(1)) * entropy_loss_weight
                model_entropy_loss += entropy_loss.item()
                total_loss = loss + aux_loss + reg_loss + entropy_loss
            else:
                total_loss = loss + aux_loss + reg_loss
            # total_loss = loss + aux_loss

            total_loss.backward()
            optimizer.step()
            if e == epoch - 1:
                for _, (gate, label) in enumerate(zip(gates, labels)):
                    tensor[:, label.item()] += gate.squeeze()
                for b in range(input_ids.size(0)):
                    client_index = torch.argsort(gates[b], descending=True)
                    for index in range(model.k):
                        gate_count[client_index[index].item()] += 1
                        gate_prob[client_index[index].item()] += gates[b][client_index[index].item()].item()

        model_CE_loss = model_CE_loss / len(trainloader)
        model_entropy_loss = model_entropy_loss / len(trainloader)
        model_aux_loss = model_aux_loss / len(trainloader)
        model_l2_loss = model_l2_loss / len(trainloader)
    print('CE loss:', model_CE_loss,
          'aux loss:', model_aux_loss,
          'l2 loss:', model_l2_loss,
          'entropy loss:', model_entropy_loss)
    print('gate count:', gate_count)
    print('gate prob:', gate_prob)
    print('gate list:', gates_list)
    print('gate on experts:', tensor.to(torch.int))
    return model_CE_loss, model_l2_loss, model_aux_loss, model_entropy_loss


def server_gate(model, fedavgloader, device_index):
    tensor = torch.zeros(model.num_experts, model.output_size, device=device_index)
    device = torch.device(device_index)
    server_gate_logits = []
    model.eval()
    model.to(device)
    gate_count = {i: 0 for i in range(model.num_experts)}
    gate_prob = {i: 0 for i in range(model.num_experts)}
    with torch.no_grad():
        for data in fedavgloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            outputs, _, gates = model(input_ids)
            for _, (gate, label) in enumerate(zip(gates, labels)):
                tensor[:, label.item()] += gate.squeeze()
            server_gate_logits.append(gates)
            for b in range(input_ids.size(0)):
                server_index = torch.argsort(gates[b], descending=True)
                for index in range(model.k):
                    # 记录每个专家的激活次数
                    gate_count[server_index[index].item()] += 1
                    # 记录每个专家的激活值的sum
                    gate_prob[server_index[index].item()] += gates[b][server_index[index].item()].item()
    print(gate_count, gate_prob)
    tensor = tensor.to(torch.int)
    # print(server_gate_logits)
    return server_gate_logits, tensor


def client_logits(client_models, fedavgloader, device_index):
    # device = torch.device(device_index)
    # client_logits = []
    # for data in fedavgloader:
    #     images, labels = data
    #     images, labels = images.to(device), labels.to(device)
    #     client_logits_batch = []
    #     for model in client_models:
    #         model.to(device)
    #         model.eval()
    #         with torch.no_grad():
    #             outputs = model(images.view(images.shape[0], -1))
    #             outputs = F.softmax(outputs, dim=1)
    #             class_probabilities = outputs[range(outputs.size(0)), labels].unsqueeze(0)
    #
    #             client_logits_batch.append(class_probabilities)
    #     client_logits_batch_combine = torch.cat(client_logits_batch, dim=0)
    #     client_logits.append(client_logits_batch_combine)
    # client_logits_combine = torch.cat(client_logits, dim=-1)
    # print(client_logits_combine.shape)
    # top_logits, top_indices = client_logits_combine.topk(2, dim=0)
    #
    # top_k_gates = F.softmax(top_logits, dim=0)
    # zeros = torch.zeros_like(client_logits_combine)
    # p = zeros.scatter(0, top_indices, top_k_gates)
    # return p

    device = torch.device(device_index)
    client_logits = []
    for data in fedavgloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)
        client_logits_batch = []
        for model in client_models:
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids)
                outputs = F.softmax(outputs, dim=1)
                class_probabilities = outputs[range(outputs.size(0)), labels].unsqueeze(0)
                client_logits_batch.append(class_probabilities)
        client_logits_batch_combine = torch.cat(client_logits_batch, dim=0)
        client_logits.append(client_logits_batch_combine)
    client_logits_combine = torch.cat(client_logits, dim=-1)
    return client_logits_combine


def expert_fix_logits(model, fedavgloader, device_index):
    device = torch.device(device_index)
    client_logits = []
    for data in fedavgloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)
        client_logits_batch = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            outputs = F.softmax(outputs, dim=1)
            class_probabilities = outputs[range(outputs.size(0)), labels].unsqueeze(0)
            client_logits_batch.append(class_probabilities)
        client_logits_batch_combine = torch.cat(client_logits_batch, dim=0)
        client_logits.append(client_logits_batch_combine)
    client_logits_combine = torch.cat(client_logits, dim=-1)
    return client_logits_combine


def client_moe_logits(client_moe_models, fedavgloader, device_index):
    device = torch.device(device_index)
    client_logits = []
    gates_logits = []
    for data in fedavgloader:
        gatas_all = {i: [] for i in range(len(client_moe_models))}
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        client_logits_batch = []
        for i, model in enumerate(client_moe_models):
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs, _, gates = model(images.view(images.shape[0], -1))
                gatas_all[i].append(gates.unsqueeze(0))
                outputs = F.softmax(outputs, dim=1)
                class_probabilities = outputs[range(outputs.size(0)), labels].unsqueeze(0)
                client_logits_batch.append(class_probabilities)
        client_logits_batch_combine = torch.cat(client_logits_batch, dim=0)
        client_logits.append(client_logits_batch_combine)
        max_indices = torch.argmax(client_logits_batch_combine, dim=0)
        gates_logits_batch = []
        for i in range(labels.size(0)):
            gates_logits_batch.append(gatas_all[max_indices[i].item()][0][:, i])
        gates_logits_batch_combine = torch.cat(gates_logits_batch, dim=0)
        gates_logits.append(gates_logits_batch_combine)

    client_logits_combine = torch.cat(client_logits, dim=-1)
    max_indices = torch.argmax(client_logits_combine, dim=0).unsqueeze(1)
    gates_logits_combine = torch.cat(gates_logits, dim=0)

    second_indices = 2*max_indices+1
    gates_indices = torch.cat((2*max_indices, second_indices), dim=1)
    print(gates_indices[-1])
    zeros = torch.zeros(200, 10, device=device)
    p = zeros.scatter(1, gates_indices, gates_logits_combine)
    return p.T
