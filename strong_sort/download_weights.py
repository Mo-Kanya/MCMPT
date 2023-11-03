import gdown

#find your model_url in path:`Yolov5_StrongSORT_OSNet/strong_sort/deep/reid_model_factory.py`
model_url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"  
weights = "./osnet_x0_25_msmt17.pt"  ##The suffix of the file name is pt

gdown.download(model_url, str(weights), quiet=False)