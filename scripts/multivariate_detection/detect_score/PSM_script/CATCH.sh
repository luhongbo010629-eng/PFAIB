python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "PSM.csv" --model-name "catch.CATCH" --model-hyper-params '{"Mlr": 0.001, "auxi_lambda": 0.01, "batch_size": 128, "cf_dim": 16, "d_ff": 32, "d_model": 16, "dc_lambda": 0.05, "dropout": 0.3, "e_layers": 1, "head_dim": 32, "inference_patch_size": 96, "lr": 0.005, "n_heads": 4, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.5, "seq_len": 192}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/CATCH"

--config-path
unfixed_detect_score_multi_config.json
--data-name-list
PSM.csv
--model-name
catch.CATCH
--model-hyper-params
"{\"Mlr\": 0.001, \"auxi_lambda\": 0.01, \"batch_size\": 128, \"cf_dim\": 16, \"d_ff\": 32, \"d_model\": 16, \"dc_lambda\": 0.05, \"dropout\": 0.3, \"e_layers\": 1, \"head_dim\": 32, \"inference_patch_size\": 96, \"lr\": 0.005, \"n_heads\": 4, \"num_epochs\": 3, \"patch_size\": 16, \"patch_stride\": 8, \"score_lambda\": 0.5, \"seq_len\": 192,\"bn_dims\": [128, 256, 512, 1024],\"k_experts\": 2,\"bn_loss_coef\": 1e-2}"
--gpus
0
--num-workers
1
--timeout
60000
--save-path
score/CATCH