import yaml
import io

# config for retail
config = {
    'env_name' : "retail",
    'train_root' : "/home/kanya/Data/MMPdata/train",
    'train_views' : 6,  # num camera
    'train_list' : {"63am/retail_0": 16908, "63am/retail_2": 27660, "63am/retail_3": 27240,
                        "63am/retail_4": 34812, "63am/retail_5": 27864, "63am/retail_6": 28782,
                        "63am/retail_8": 28470, "64am/retail_0": 34686, "64am/retail_1": 24534,
                        "64am/retail_2": 28752, "64am/retail_3": 28152, "64am/retail_4": 37530,
                        "64am/retail_5": 32172, "64am/retail_6": 26550, "64am/retail_7": 31800,
                        "64am/retail_8": 26214},  # file paths and number of frames
    'grid_size' : [327, 425]
}

# {"1": [390, -51, 443, 69], "2": [298, 34, 343, 169], "3": [5d14, 109, 700, 255]}


if __name__ == '__main__':
    with io.open('./retail.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    with open("./retail.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        print(data_loaded)