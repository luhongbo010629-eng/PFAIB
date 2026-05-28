python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "Genesis.csv" --model-name "catch.CATCH" --model-hyper-params '{"Mlr": 1e-05, "auxi_lambda": 0.3, "batch_size": 32, "cf_dim": 32, "d_ff": 256, "d_model": 256, "dc_lambda": 0.2, "e_layers": 2, "head_dim": 32, "lr": 0.005, "n_heads": 1, "num_epochs": 4, "patch_size": 16, "patch_stride": 16, "seq_len": 192, "anomaly_ratio": 0.5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/CATCH"

 --config-path
"unfixed_detect_label_multi_config.json"
--data-name-list
"Genesis.csv"
--model-name
"catch.CATCH"
--model-hyper-params
"{\"Mlr\": 1e-05, \"auxi_lambda\": 0.3, \"batch_size\": 32, \"cf_dim\": 32, \"d_ff\": 256, \"d_model\": 16, \"dc_lambda\": 0.2, \"e_layers\": 2, \"head_dim\": 32, \"lr\": 1e-4, \"n_heads\": 1, \"num_epochs\": 5, \"patch_size\": 16, \"patch_stride\": 16, \"seq_len\": 192, \"anomaly_ratio\": 0.5,\"bn_dims\": [8,16,32,64], \"k_experts\": 2,  \"bn_loss_coef\": 0.001}"
--gpus
0
--num-workers
1
--timeout
60000
--save-path
"label/CATCH"