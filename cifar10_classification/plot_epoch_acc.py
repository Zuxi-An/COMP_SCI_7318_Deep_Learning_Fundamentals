import matplotlib.pyplot as plt
import pickle
import os

def plot_full_image_acc_epoch(save_dir):
    # for arch in ["EfficientNetB0", "MobileNetV2", "ResNet18", "SENet18"]:
    acc_EfficientNetB0 = pickle.load(open("{}/acc_{}.pkl".format(class_type, "EfficientNetB0"), "rb"))
    acc_EfficientNetB0 = [a.item() for a in acc_EfficientNetB0]

    acc_MobileNetV2 = pickle.load(open("{}/acc_{}.pkl".format(class_type, "MobileNetV2"), "rb"))
    acc_MobileNetV2 = [a.item() for a in acc_MobileNetV2]

    acc_ResNet18 = pickle.load(open("{}/acc_{}.pkl".format(class_type, "ResNet18"), "rb"))
    acc_ResNet18 = [a.item() for a in acc_ResNet18]

    acc_SENet18 = pickle.load(open("{}/acc_{}.pkl".format(class_type, "SENet18"), "rb"))
    acc_SENet18 = [a.item() for a in acc_SENet18]


    plt.figure(figsize=(15, 5), dpi=240)
    plt.plot(list(range(len(acc_EfficientNetB0))), acc_EfficientNetB0, linewidth=2, label="EfficientNetB0")
    plt.plot(list(range(len(acc_MobileNetV2))), acc_MobileNetV2, linewidth=2, label="MobileNetV2")
    plt.plot(list(range(len(acc_ResNet18))), acc_ResNet18, linewidth=2, label="ResNet18")
    plt.plot(list(range(len(acc_SENet18))), acc_SENet18, linewidth=2, label="SENet18")


    plt.xticks(range(len(acc_SENet18)))
    plt.legend(loc='lower right')
    plt.title("acc")
    plt.savefig("{}/acc_result.png".format(save_dir))
    plt.show()
    plt.close()

def plot_full_image_loss_epoch(arch, save_dir):
    train_loss = pickle.load(open("{}/train_loss_{}.pkl".format(class_type, arch), "rb"))
    val_loss = pickle.load(open("{}/val_loss_{}.pkl".format(class_type, arch), "rb"))

    plt.figure(figsize=(15, 5), dpi=240)
    plt.plot(list(range(len(train_loss))), train_loss, linewidth=2, label="train loss")
    plt.plot(list(range(len(train_loss))), val_loss, linewidth=2, label="val loss")

    plt.xticks(range(len(train_loss)))
    plt.legend(loc='lower right')
    plt.title("{}".format(arch))
    plt.savefig("{}/loss_{}_result.png".format(save_dir, arch))
    # plt.show()
    plt.close()

if __name__ == "__main__":
    class_type = "checkpoints"
    save_dir = "{}/acc_loss".format(class_type)
    os.makedirs(save_dir, exist_ok=True)
    plot_full_image_acc_epoch(save_dir)
    for arch in ["EfficientNetB0", "MobileNetV2", "ResNet18", "SENet18"]:
        plot_full_image_loss_epoch(arch, save_dir)