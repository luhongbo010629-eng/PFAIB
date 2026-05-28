python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_sub_mix0.0574.csv" --model-name "catch.CATCH" --model-hyper-params '{"Mlr": 0.0005, "auxi_lambda": 0.3, "batch_size": 64, "cf_dim": 16, "d_ff": 16, "d_model": 32, "dc_lambda": 0.5, "dropout": 0.05, "e_layers": 1, "head_dim": 32, "head_dropout": 0.1, "lr": 0.0005, "n_heads": 1, "num_epochs": 2, "patch_size": 16, "patch_stride": 16, "score_lambda": 0.01, "seq_len": 96, "anomaly_ratio": 3.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/CATCH"

--config-path
"unfixed_detect_label_multi_config.json"
--data-name-list
"synthetic_sub_mix0.0574.csv"
--model-name
"catch.CATCH"
--model-hyper-params
"{\"Mlr\": 0.0005, \"auxi_lambda\": 0.3, \"batch_size\": 64, \"cf_dim\": 16, \"d_ff\": 16, \"d_model\": 32, \"dc_lambda\": 0.5, \"dropout\": 0.05, \"e_layers\": 1, \"head_dim\": 32, \"head_dropout\": 0.1, \"lr\": 0.0005, \"n_heads\": 1, \"num_epochs\": 2, \"patch_size\": 16, \"patch_stride\": 16, \"score_lambda\": 0.01, \"seq_len\": 96, \"anomaly_ratio\": 3.0,\"bn_dims\": [2, 4, 8, 16], \"k_experts\": 1,  \"bn_loss_coef\": 0.01}"
--gpus
0
--num-workers
1
--timeout
60000
--save-path
"label/CATCH"