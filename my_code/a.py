import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--abs", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--list", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)
    parser.add_argument("--sep", type=int, default=5)

    return parser.parse_args()


def fetch_all_locs(file, absp):
    ret = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            ret.append((absp + "/" + line).replace("\r", "").replace("\n", ""))
    return ret


def make_list_txt(ret_list, names, lists):
    rets = {}
    for ret in ret_list:
        path_ret_root = ret
        for item in zip(names, lists):
            if rets.get(item[0], None) is None:
                rets[item[0]] = []
            rets[item[0]].append(path_ret_root + "/" + item[1])

    return rets


def save_all(ret_dict, sep):
    for k, v in ret_dict.items():
        file_name = k

        sep_train, sep_val = v[:sep], v[sep:]

        with open(f"{file_name}_1.txt", "w") as f:
            f.writelines([item + "\n" for item in sep_train])

        with open(f"{file_name}_2.txt", "w") as f:
            f.writelines([item + "\n" for item in sep_val])


opt = parse_args()
ret_list = fetch_all_locs(opt.file, opt.abs)
print(json.dumps(make_list_txt(ret_list, eval(opt.names), eval(opt.list)), indent=4))
ret_list2 = make_list_txt(ret_list, eval(opt.names), eval(opt.list))
save_all(ret_list2, opt.sep)
