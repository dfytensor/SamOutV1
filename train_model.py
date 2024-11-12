import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from model import SamOut

import polars as pl
from collections import Counter


def train():
    voc = pd.read_pickle("total_voc.pkl")

    net = SamOut(len(voc["voc"]), 768, 32, 16)
    print(sum([i.shape[0] * i.shape[1] for i in net.parameters() if len(i.shape) > 1]) + sum(
        [i.shape[0] for i in net.parameters() if len(i.shape) == 1]))

    net.load_state_dict(torch.load("pretrain_768.pth"))
    net.to("cuda")

    opt = torch.optim.Adam(params=net.parameters(), lr=0.00002)
    loss_func0 = torch.nn.CrossEntropyLoss(ignore_index=3)

    bar = tqdm(range(10))
    steps = 0
    epoch_loss = []
    batch_size = 30

    for epoch in bar:
        paths = glob("./pre_data_set_*.pkl")
        data_set = []
        for ii in range(0, len(paths), 2):

            for one_path in paths[ii:ii + 2]:

                data_set = pd.read_pickle(one_path)
                np.random.shuffle(data_set)
                loss_list = []
                for i in range(0, len(data_set), batch_size):
                    # weights.append(list(net.state_dict().values())[0])
                    j = i + batch_size
                    input_one = data_set[i:j]

                    out0, _ = net(torch.Tensor(input_one)[:, :-1].int().to("cuda"))
                    loss = loss_func0(out0.reshape([-1, out0.shape[-1]]),
                                      torch.Tensor(input_one)[:, 1:].reshape([-1]).long().to("cuda"))

                    loss_list.append(loss.item())
                    bar.set_description(
                        "epoch___{}____loss___{:.6f}____steps___{}".format(epoch, np.mean(loss_list), steps))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    steps += batch_size

                torch.save(net.state_dict(), "pretrain_768.pth")
                # eval_model()
                epoch_loss.append(np.mean(loss_list))
                pd.to_pickle(epoch_loss, "loss916")


def gen_one_voc():
    data = pd.read_csv("pretrain_data.csv")

    data = data["text"].values.tolist()
    data = "".join(data)
    count = Counter()
    for ii in tqdm(range(0, len(data), len(data) // 8)):
        jj = ii + len(data) // 8
        for k, v in Counter(data[ii:jj]).items():
            count[k] = count.get(k, 0) + v

    data = ""
    data0 = pd.read_csv("sft_data_multi.csv")
    for ii in tqdm(range(0, len(data0), len(data0) // 8)):
        jj = ii + len(data0) // 8
        for k, v in Counter(data0[ii:jj]).items():
            count[k] = count.get(k, 0) + v
    data0 = ""
    data1 = pd.read_csv("sft_data_single.csv")
    for ii in tqdm(range(0, len(data1), len(data1) // 8)):
        jj = ii + len(data1) // 8
        for k, v in Counter(data1[ii:jj]).items():
            count[k] = count.get(k, 0) + v
    data1 = ""

    # plt.plot(sorted(count.values()))
    # plt.show()
    count = pd.DataFrame({"voc": count.keys(), "count": count.values()})
    voc = count.loc[count["count"] > 100, "voc"].values.tolist()
    voc0 = [[[["<|pos_{}_{}|>".format(jj, ii) for jj, ii in enumerate(list(str(i)))], j] for i, j in
             enumerate(count.loc[count["count"] <= 100, "voc"].values.tolist())]]
    pd.to_pickle(voc, "voc.pkl")
    pd.to_pickle(voc0, "voc0.pkl")


def gen_voc():
    voc = pd.read_pickle("voc.pkl")
    voc0 = pd.read_pickle("voc0.pkl")
    voc0 = {j: i for i, j in voc0[0]}
    for i in range(6):
        for j in range(10):
            voc.append("<|pos_{}_{}|>".format(i, j))
    voc = ["<|sos|>", "<|user|>", "<|agent|>", "<|pad|>", "<|history|>"] + sorted(voc)

    pd.to_pickle({"voc": voc, "voc0": voc0}, "total_voc.pkl")


def gen_pre_data_align(num, total_num):
    voc = pd.read_pickle("total_voc.pkl")
    voc["voc0"] = [[i, [voc["voc"].index(j) for j in ii]] for i, ii in voc["voc0"].items()]
    voc["voc"] = [i for i in voc["voc"]]
    voc = {"voc": voc["voc"] + [i for i, j in voc["voc0"]],
           "voc_id": [[i] for i in list(range(len(voc["voc"])))] + [j for i, j in voc["voc0"]]}
    voc = pd.DataFrame(voc)
    # voc=pl.DataFrame(voc)

    pre_data = pl.read_csv("pretrain_data.csv")
    pre_data = pre_data["text"].to_numpy().tolist()
    count = len(pre_data) // total_num
    pre_data = pre_data[(num - 1) * count:count * num]
    data_set = []
    bar = tqdm(range(len(pre_data)))

    while pre_data:
        bar.update()
        one = pre_data.pop()
        one = pd.merge(pd.DataFrame({"voc": list(one)}), voc, on="voc", how="left")

        thr = np.hstack(one["voc_id"].to_numpy()).tolist()

        thr += (518 - len(thr)) * [3]
        thr = thr[:512]
        data_set.append(thr)
    pd.to_pickle(data_set, "pre_data_set_{}.pkl".format(num))


def gen_sft_single_data_align():
    voc = pd.read_pickle("total_voc.pkl")
    voc["voc0"] = {i: [voc["voc"].index(j) for j in ii] for i, ii in voc["voc0"].items()}
    voc["voc"] = {v: i for i, v in enumerate(voc["voc"])}

    pre_data = pl.read_csv("sft_data_single.csv")
    pre_data = pre_data.to_numpy().tolist()
    data_set = []
    index_id = 0
    for h, q, a in tqdm(pre_data):
        index_id += 1
        one = ["<|user|>"] + list(q) + ["<|agent|>"] + list(a)
        one_list = []
        for i in one:
            voc_id = voc["voc"].get(i, None)
            if voc_id != None:
                one_list.append(voc_id)
            else:
                one_list += voc["voc0"].get(i, [3])
        one_list += (512 - len(one_list)) * [3]
        data_set.append(one_list[:512])
        if len(data_set) > 1000000:
            pd.to_pickle(data_set, "sft_data_single_{}.pkl".format(index_id))
            data_set = []
    pd.to_pickle(data_set, "sft_data_single_{}.pkl".format(index_id))


def train_single():
    voc = pd.read_pickle("total_voc.pkl")

    net = SamOut(len(voc["voc"]), 512, 32, 8)

    net.load_state_dict(torch.load("pretrain_sft_single.pth"))
    net.to("cuda")

    opt = torch.optim.Adam(params=net.parameters(), lr=0.000003)
    loss_func0 = torch.nn.CrossEntropyLoss(ignore_index=3)

    bar = tqdm(range(2))
    steps = 0
    epoch_loss = []

    for epoch in bar:
        paths = glob("./sft_data_*.pkl")
        np.random.shuffle(paths)
        for o in range(0, len(paths), 2):
            data_set = []
            for one_path in paths[o:o + 2]:
                data_set += pd.read_pickle(one_path)

            np.random.shuffle(data_set)

            loss_list = []
            for i in range(0, len(data_set), 80):
                # weights.append(list(net.state_dict().values())[0])
                j = i + 80
                input_one = data_set[i:j]

                out0, _ = net(torch.Tensor(input_one)[:, :-1].int().to("cuda"))
                loss = loss_func0(out0.reshape([-1, out0.shape[-1]]),
                                  torch.Tensor(input_one)[:, 1:].reshape([-1]).long().to("cuda"))

                loss_list.append(loss.item())
                bar.set_description(
                    "epoch___{}____loss___{:.6f}____steps___{}".format(epoch, np.mean(loss_list), steps))
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 80

            torch.save(net.state_dict(), "pretrain_sft_single.pth")
            # eval_model()
            epoch_loss.append(np.mean(loss_list))
            pd.to_pickle(epoch_loss, "loss916")


def load_model_and_voc(device="cpu"):
    voc = pd.read_pickle("total_voc.pkl")

    net = SamOut(len(voc["voc"]), 768, 32, 16)
    # net = SamOut(len(voc["voc"]), 512, 32, 8)
    print(sum([i.shape[0] * i.shape[1] for i in net.parameters() if len(i.shape) > 1]) + sum(
        [i.shape[0] for i in net.parameters() if len(i.shape) == 1]))

    # net.load_state_dict(torch.load("pretrain_768.pth", map_location=device))
    # net.load_state_dict(torch.load("pretrain_sft_single.pth", map_location=device))
    net.load_state_dict(torch.load("pretrain_sft_single_768.pth", map_location=device))
    # net.load_state_dict(torch.load("pretrain.pth", map_location=device))
    net.to(device)
    net.eval()
    return net, voc


def gen_token(prompt, max_len, rp=1.2, temp=0.7, top_k=12,device="cpu"):
    model, voc = load_model_and_voc()
    for _ in  tqdm(range(max_len)):

        prompt_list = []
        for i in prompt:
            if i not in voc["voc"]:
                prompt_list += [voc["voc"].index(ii) for ii in voc["voc0"].get(i)]
            else:

                prompt_list.append(voc["voc"].index(i))
        out,_=model(torch.Tensor([prompt_list]).to(device).long())
        out=out[:,-1:]
        # 重复抑制
        for token_id in enumerate(prompt_list):
            out[:,:,token_id]/=rp
        # 温度

        out = out / temp

        v, _ = torch.topk(out, min(top_k, out.size(-1)))
        out[out < v[:,:, [-1]]] = -float('Inf')

        probs = torch.nn.functional.softmax(out, dim=-1)
        idx_next = torch.multinomial(probs.reshape(-1), num_samples=1, generator=None)
        if voc["voc"][idx_next.item()]=="<|sos|>":
            break
        prompt+=[voc["voc"][idx_next.item()]]
        print("".join(prompt))

if __name__ == '__main__':
    # print(pd.read_pickle("loss916"))
    # gen_one_voc()
    # gen_voc()
    # for i in range(17,18):
    #     gen_pre_data_align(i, 16)

    # train()
    # gen_sft_single_data_align()
    # train_single()
    # sft 推理  一本正经的胡说八道已练成
    gen_token(["<|user|>"]+list("你是谁开发的？")+["<|agent|>"], 320)
