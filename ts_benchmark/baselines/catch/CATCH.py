import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion, frequency_loss
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "cf_dim": 64,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "individual": 0,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "patch_stride": 8,
    "patch_size": 16,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "dc_lambda": 0.005,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 128,
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 192,
    "pct_start": 0.3,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "lradj": "type1",
    "use_adaptive_bottleneck": True,
    "bn_dims": [64, 128, 256, 512],
    "k_experts": 2,
    "bn_loss_coef": 0.01,
    "bottleneck_residual": "scaled",
    "routing_level": "patch",
    "use_frequency_router_features": True,
    "freq_pos_dim": 8,
    "mask_update_interval": 1,
    "early_stop_loss": "total", #total\rec
    "threshold_source": "combined", #combined\test
    "baseline_compatible": True,
}


class TransformerConfig:
    def __init__(self, **kwargs):
        self._user_keys = set(kwargs.keys())
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class CATCH:
    def __init__(self, **kwargs):
        super(CATCH, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        self.model = None
        self.latest_test_gates = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return getattr(self, "model_name", self.__class__.__name__)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception:
            freq = "S"

        if freq is None:
            raise ValueError("Irregular time intervals")
        if freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def _losses(self, inputs, outputs, output_complex, auxiliary_loss):
        rec_loss = self.criterion(outputs, inputs)
        norm_input = self.model.latest_norm_input
        if norm_input is None:
            norm_input = self.model.revin_layer(inputs, "transform")
        auxi_loss = self.auxi_loss(output_complex, norm_input)
        total_loss = rec_loss + auxiliary_loss + self.config.auxi_lambda * auxi_loss
        return rec_loss, auxi_loss, total_loss

    def detect_validate(self, valid_data_loader, criterion=None):
        rec_losses = []
        total_losses = []
        self.model.eval()

        with torch.no_grad():
            for inputs, _ in valid_data_loader:
                inputs = inputs.float().to(self.device)
                outputs, output_complex, auxiliary_loss = self.model(inputs)
                rec_loss, _, total_loss = self._losses(
                    inputs,
                    outputs,
                    output_complex,
                    auxiliary_loss,
                )
                rec_losses.append(rec_loss.detach().cpu().item())
                total_losses.append(total_loss.detach().cpu().item())

        self.model.train()
        return float(np.mean(rec_losses)), float(np.mean(total_losses))

    def _average_mask_generator_grads(self, num_batches):
        if num_batches <= 1:
            return
        for param in self.model.mask_generator.parameters():
            if param.grad is not None:
                param.grad.div_(num_batches)

    def _apply_baseline_compatibility(self):
        if self.config.use_adaptive_bottleneck or not self.config.baseline_compatible:
            return
        if "early_stop_loss" not in self.config._user_keys:
            self.config.early_stop_loss = "rec"
        if "threshold_source" not in self.config._user_keys:
            self.config.threshold_source = "combined"

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        _ = test_data

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self._apply_baseline_compatibility()
        self.model = CATCHModel(self.config).to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )
        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        if config.use_adaptive_bottleneck:
            print("Adaptive Bottleneck enabled:")
            print(f"  - Expert dimensions: {config.bn_dims}")
            print(f"  - Active experts (k): {config.k_experts}")
            print(f"  - Routing level: {config.routing_level}")
            print(f"  - Balance loss coefficient: {config.bn_loss_coef}")

        self.early_stopping = EarlyStopping(patience=config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        main_params = [
            param
            for name, param in self.model.named_parameters()
            if "mask_generator" not in name
        ]
        self.optimizer = torch.optim.Adam(main_params, lr=config.lr)
        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=config.Mlr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=config.pct_start,
            epochs=config.num_epochs,
            max_lr=config.lr,
        )
        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=config.pct_start,
            epochs=config.num_epochs,
            max_lr=config.Mlr,
        )

        mask_update_interval = max(1, int(config.mask_update_interval or 1))

        time_now = time.time()
        for epoch in range(config.num_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()
            self.optimizerM.zero_grad()
            mask_accum_count = 0

            for i, (inputs, _) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                if mask_update_interval == 1:
                    self.optimizerM.zero_grad()

                inputs = inputs.float().to(self.device)
                outputs, output_complex, auxiliary_loss = self.model(inputs)
                rec_loss, auxi_loss, loss = self._losses(
                    inputs,
                    outputs,
                    output_complex,
                    auxiliary_loss,
                )

                loss.backward()
                self.optimizer.step()
                mask_accum_count += 1

                if (i + 1) % mask_update_interval == 0 or (i + 1) == train_steps:
                    self._average_mask_generator_grads(mask_accum_count)
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()
                    mask_accum_count = 0

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0 or (i + 1) == train_steps:
                    print(
                        "\titers: {0}, epoch: {1} | reconstruction loss: {2:.7f} | "
                        "frequency loss: {3:.7f} | auxiliary loss: {4:.7f}".format(
                            i + 1,
                            epoch + 1,
                            rec_loss.item(),
                            auxi_loss.item(),
                            auxiliary_loss.item(),
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((config.num_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_rec_loss, valid_total_loss = self.detect_validate(self.valid_data_loader)
            early_stop_loss = str(config.early_stop_loss).lower()
            valid_loss = valid_total_loss if early_stop_loss == "total" else valid_rec_loss
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Val Rec Loss: {3:.7f} Val Total Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    valid_rec_loss,
                    valid_total_loss,
                )
            )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, config, printout=False)

    def _load_best_model(self):
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        if hasattr(self, "early_stopping") and hasattr(self.early_stopping, "check_point"):
            self.model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

    def _collect_anomaly_scores(self, data_loader, print_examples=False, collect_gates=False):
        attens_energy = []
        all_gates = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x)
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = temp_score + self.config.score_lambda * freq_score
                attens_energy.append(score.detach().cpu().numpy())

                if collect_gates and hasattr(self.model, "adaptive_bottleneck"):
                    gates = getattr(self.model.adaptive_bottleneck, "latest_gates", None)
                    if gates is not None:
                        all_gates.append(gates.detach().cpu())

                if print_examples:
                    print(
                        "\t testing time loss: {0} | \n\t testing fre loss: {1}".format(
                            temp_score.detach().cpu().numpy()[0, :5],
                            freq_score.detach().cpu().numpy()[0, :5],
                        )
                    )

        scores = np.concatenate(attens_energy, axis=0).reshape(-1)
        if collect_gates:
            gates = torch.cat(all_gates, dim=0) if all_gates else None
            return np.array(scores), gates
        return np.array(scores)

    def _scaled_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.transform(data.values),
            columns=data.columns,
            index=data.index,
        )

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = self._scaled_frame(test)
        self._load_best_model()

        config = self.config
        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.temp_anomaly_criterion = nn.MSELoss(reduction="none")
        self.freq_anomaly_criterion = frequency_criterion(config)
        test_energy, self.latest_test_gates = self._collect_anomaly_scores(
            self.thre_loader,
            print_examples=True,
            collect_gates=True,
        )
        return test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = self._scaled_frame(test)
        self._load_best_model()

        config = self.config
        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.temp_anomaly_criterion = nn.MSELoss(reduction="none")
        self.freq_anomaly_criterion = frequency_criterion(config)

        train_energy = self._collect_anomaly_scores(self.train_data_loader)
        test_energy, self.latest_test_gates = self._collect_anomaly_scores(
            self.thre_loader,
            print_examples=True,
            collect_gates=True,
        )

        threshold_source = getattr(config, "threshold_source", "train").lower()
        if threshold_source == "train":
            threshold_energy = train_energy
        elif threshold_source == "combined":
            threshold_energy = np.concatenate([train_energy, test_energy], axis=0)
        else:
            raise ValueError("threshold_source must be 'train' or 'combined'")

        if not isinstance(config.anomaly_ratio, list):
            config.anomaly_ratio = [config.anomaly_ratio]

        preds = {}
        for ratio in config.anomaly_ratio:
            threshold = np.percentile(threshold_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
