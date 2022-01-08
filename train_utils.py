import os
import datetime
import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from model import Generator
from model import Discriminator


def get_loader(image_dir, resize=100, batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    dataset = ImageFolder(image_dir, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(c_org, c_dim, device):
    """ Generate target domain labels for debugging and testing. """
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=weight,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def update_lr(g_optimizer, d_optimizer, g_lr, d_lr):
    """Decay learning rates of the generator and discriminator."""
    for param_group in g_optimizer.param_groups:
        param_group["lr"] = g_lr
    for param_group in d_optimizer.param_groups:
        param_group["lr"] = d_lr


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def reset_grad(g_optimizer, d_optimizer):
    """Reset the gradient buffers."""
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


def train(config):
    """Train StarGAN within a single dataset."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).to(device)
    D = Discriminator(config.resize, config.d_conv_dim, config.c_dim, config.d_repeat_num).to(device)

    g_optimizer = torch.optim.Adam(
        G.parameters(), config.g_lr, [config.beta1, config.beta2]
    )
    d_optimizer = torch.optim.Adam(
        D.parameters(), config.d_lr, [config.beta1, config.beta2]
    )

    # 学習過程可視化用 画像のセット
    data_loader = get_loader(image_dir=f"./data/{config.dataset_name}", resize=config.resize, batch_size=config.batch_size)  # set dataloader
    data_iter = iter(data_loader)  # to iterator
    x_fixed, c_org = next(data_iter)  # sampling
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, config.c_dim, device)  # make label

    # Learning rate cache for decaying.
    g_lr = config.g_lr
    d_lr = config.d_lr

    # train log 用データフレーム
    df = pd.DataFrame(columns={'epoch', 'D/loss_real', 'D/loss_fake', "D/loss_cls", "D/loss_gp", "G/loss_fake", "G/loss_rec", "G/loss_cls"})

    # Start training from scratch or resume training.
    start_iters = 0
    # Start training.
    print("Start training...")
    start_time = time.time()
    for i in range(start_iters, config.num_iters):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        # Fetch real images and labels.
        try:
            x_real, label_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, label_org = next(data_iter)

        # ターゲットラベルをランダムに決定
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = label2onehot(label_org, config.c_dim)  # 元のラベル(one-hot)
        c_trg = label2onehot(label_trg, config. c_dim)  # 変換先ラベル(one-hot)

        x_real = x_real.to(device)  # Input images.
        c_org = c_org.to(device)  # Original domain labels.
        c_trg = c_trg.to(device)  # Target domain labels.

        label_org = label_org.to(device)  # Labels for computing classification loss.
        label_trg = label_trg.to(device)  # Labels for computing classification loss.

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #

        # Compute loss with real images.
        out_src, out_cls = D(x_real)  # 識別器に元画像を入力 -> real or fake, pred class
        d_loss_real = -torch.mean(out_src)  # 識別能のロス計算 (全部本物の画像だから平均にマイナスを乗算するとlossになる)
        d_loss_cls = F.cross_entropy(out_cls, label_org)  # クラス分類のロス計算

        # Compute loss with fake images.
        x_fake = G(x_real, c_trg)  # fake画像の生成
        out_src, out_cls = D(x_fake.detach())  # 識別器にfake画像入力 -> real or fake, pred class
        d_loss_fake = torch.mean(out_src)  # 識別能のロス計算 (全部fake画像だから平均取るだけでlossになる)

        # Compute loss for gradient penalty. (8)式 最終項の右側の計算
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = D(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, device)  # (8)式 最終項の右側

        # Backward and optimize.
        d_loss = (
            d_loss_real
            + d_loss_fake
            + config.lambda_cls * d_loss_cls
            + config.lambda_gp * d_loss_gp
        )

        # 識別器の更新
        reset_grad(g_optimizer, d_optimizer)  # 勾配リセット
        d_loss.backward()  # 逆伝播
        d_optimizer.step()  # パラメータ更新

        # Logging.
        loss = {}
        loss["epoch"] = i
        loss["D/loss_real"] = d_loss_real.item()
        loss["D/loss_fake"] = d_loss_fake.item()
        loss["D/loss_cls"] = d_loss_cls.item()
        loss["D/loss_gp"] = d_loss_gp.item()

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        if (i + 1) % config.n_critic == 0:
            # Original-to-target domain. (オリジナル -> ターゲットに変換)
            x_fake = G(x_real, c_trg)  # fake画像の生成
            out_src, out_cls = D(x_fake)  # 識別器にfake画像入力 -> real or fake, pred class
            g_loss_fake = -torch.mean(out_src)  # ロス計算
            g_loss_cls = F.cross_entropy(out_cls, label_trg)  # クラス分類ロス計算
            # Target-to-original domain. (フェイクから元の画像に変換)
            x_reconst = G(x_fake, c_org)  # fake画像 -> 元のラベルに変換
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))  # (元の画像との差の絶対値を計算) cycle consistency loss

            # edges loss
            batch_size = x_real.size(1)
            origin_edges = torch.squeeze(x_real, 1).reshape(batch_size, -1).sum(1)
            fake_edges = torch.squeeze(x_fake, 1).reshape(batch_size, -1).sum(1)

            edges_loss = (origin_edges - fake_edges).sum().item() * config.lambda_edges

            # Backward and optimize.
            g_loss = (
                g_loss_fake
                + config.lambda_rec * g_loss_rec
                + config.lambda_cls * g_loss_cls + edges_loss
            )

            # 生成器の更新
            reset_grad(g_optimizer, d_optimizer)
            g_loss.backward()
            g_optimizer.step()
            # Logging.
            loss["G/loss_fake"] = g_loss_fake.item()
            loss["G/loss_rec"] = g_loss_rec.item()
            loss["G/loss_cls"] = g_loss_cls.item()

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #

        # Print out training information.
        if (i + 1) % config.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(
                et, i + 1, config.num_iters
            )
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

        # Translate fixed images for debugging.
        if (i + 1) % config.sample_step == 0:
            with torch.no_grad():
                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(
                    f"{config.parent_dir}/sample", "{}-images.jpg".format(i + 1)
                )
                save_image(
                    x_concat.data.cpu(), sample_path, nrow=1, padding=0
                )
                print("Saved real and fake images into {}...".format(sample_path))

        # Save model checkpoints.
        if (i + 1) % config.model_save_step == 0:
            G_path = os.path.join(f"{config.parent_dir}/model", "{}-G.ckpt".format(i + 1))
            D_path = os.path.join(f"{config.parent_dir}/model", "{}-D.ckpt".format(i + 1))
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)
            print("Saved model checkpoints into {}/model...".format(config.parent_dir))

        # Decay learning rates.
        if (i + 1) % config.lr_update_step == 0 and (i + 1) > (
            config.num_iters - config.num_iters_decay
        ):
            g_lr -= g_lr / float(config.num_iters_decay)
            d_lr -= d_lr / float(config.num_iters_decay)
            update_lr(g_optimizer, d_optimizer, g_lr, d_lr)
            print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))

        # log
        df = df.append(pd.Series(loss), ignore_index=True)

    df.to_csv(f"{config.parent_dir}/loss_log.csv", index=False)
