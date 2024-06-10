#  参数设置
class Config(object):

    # 数据地址
    train_y = "F:\python\机器学习\Data\mnist_m_train_labels.txt"
    train_x = "F:\python\机器学习\Data\mnist_m_train"
    test_y = "F:\python\机器学习\Data\mnist_m_test_labels.txt"
    test_x = "F:\python\机器学习\Data\mnist_m_test"

    # 源数据地址
    train_dir = r"F:\python\机器学习\SData\train"
    test_dir = r"F:\python\机器学习\SData\test"

    save_path = f"F:/python/机器学习/model_dict/source.pth"
    epoch_source = 10
    epoch_target = 10
    batch_size = 32

    lr = 1e-3
    eps=1e-8 
    device = 'cuda'

    random_seed = 10
