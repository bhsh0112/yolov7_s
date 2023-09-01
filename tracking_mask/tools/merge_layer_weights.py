layer_names_file = "/home/omnisky/programfiles/tracking/pysot/tools/cache/layer_names.txt"
layer_weights_file = layer_names_file.replace("names", "weights")
with open(layer_names_file, "r") as f:
    raw_lines = f.readlines()

layer_names = [l.strip() for l in raw_lines]

with open(layer_weights_file, "r") as f:
    raw_lines = f.readlines()

layer_weights = [l.strip() for l in raw_lines]


def process_prototxt(prototxt_file):
    with open(prototxt_file, "r") as f:
        raw_lines = f.readlines()
    raw_lines = raw_lines[:-1]

    return raw_lines


def merge_str(prototxt):
    s = ""
    for p in prototxt:
        s += p
    s += "}\n"
    return s

n = 0
prototxt_root = "/home/omnisky/programfiles/tracking/pysot/tools/cache/layer_with_weights/"

def find_bn(layer_name, layer_weights, n, weight_str):
    layer_name = layer_name.split(".bn")[0]
    running_mean = layer_name + ".running_mean"
    for layer_weight in layer_weights:
        if running_mean in layer_weight:
            # print(running_mean==layer_weight)
            n += 1
            prototxt_file = prototxt_root + layer_weight + ".prototxt"
            prototxt = process_prototxt(prototxt_file)
            weight_str += "{}\n".format(layer_weight)
    running_var = layer_name + ".running_var"
    for layer_weight in layer_weights:
        if running_var in layer_weight:
            # print(running_var==layer_weight)
            n += 1
            prototxt_file = prototxt_root + layer_weight + ".prototxt"
            prototxt += process_prototxt(prototxt_file)
            weight_str += "{}\n".format(layer_weight)
    return n, weight_str, prototxt


def find_sc(layer_name, layer_weights, n, weight_str):
    layer_name = layer_name.split(".sc")[0]
    weight = layer_name + ".weight"
    for layer_weight in layer_weights:
        if weight in layer_weight:
            # print(weight==layer_weight)
            n += 1
            prototxt_file = prototxt_root + layer_weight + ".prototxt"
            prototxt = process_prototxt(prototxt_file)
            # print(prototxt)
            weight_str += "{}\n".format(layer_weight)
    bias = layer_name + ".bias"
    for layer_weight in layer_weights:
        if bias in layer_weight:
            # print(bias)
            n += 1
            prototxt_file = prototxt_root + layer_weight + ".prototxt"
            prototxt += process_prototxt(prototxt_file)
            weight_str += "{}\n".format(layer_weight)
    return n, weight_str, prototxt

f = open("net.prototxt", "w")
for l, layer_name in enumerate(layer_names):
    prototxt_file = prototxt_root + layer_name + ".prototxt"
    prototxt = process_prototxt(prototxt_file)
    # f.write(prototxt)
    # break
    weight_str = ""
    if "_temp" in layer_name:
        layer_name = layer_name.split("_temp")[0]
    elif "_srch" in layer_name:
        layer_name = layer_name.split("_srch")[0]
    if ".bn" in layer_name:
        n, weight_str, p = find_bn(layer_name, layer_weights, n, weight_str)
        prototxt += p
    elif ".sc" in layer_name and not (".sc_relu" in layer_name) and not (".sc_reshape" in layer_name):
        n, weight_str, p = find_sc(layer_name, layer_weights, n, weight_str)
        prototxt += p
    else:
        weight = layer_name + ".weight"
        for layer_weight in layer_weights:
            if weight in layer_weight:
                # print(weight==layer_weight)
                n += 1
                prototxt_file = prototxt_root + layer_weight + ".prototxt"
                prototxt += process_prototxt(prototxt_file)
                # f.write(prototxt)
                weight_str += "{}\n".format(layer_weight)
        bias = layer_name + ".bias"
        for layer_weight in layer_weights:
            if bias in layer_weight:
                # print(bias)
                n += 1
                prototxt_file = prototxt_root + layer_weight + ".prototxt"
                prototxt += process_prototxt(prototxt_file)
                weight_str += "{}\n".format(layer_weight)
    prototxt_str = merge_str(prototxt)
    print("layer_name ---- {}\n".format(layer_name))
    if weight_str is not None:
        print("weight_str ---- {}\n".format(weight_str))
        # f.write("{}".format(weight_str))
    else:
        print("weight_str ---- None\n")
    f.write("{}".format(prototxt_str))
    print("**"*20)
    # if l ==2:
    #     break
f.close()