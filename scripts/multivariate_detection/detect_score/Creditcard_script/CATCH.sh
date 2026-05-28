python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "Creditcard.csv" --model-name "catch.CATCH" --model-hyper-params '{"Mlr": 1e-05, "batch_size": 128, "cf_dim": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "head_dim": 64, "inference_patch_size": 16, "lr": 0.0001, "n_heads": 16, "num_epochs": 5, "patch_size": 32, "patch_stride": 32, "seq_len": 192}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/CATCH"

--config-path
unfixed_detect_score_multi_config.json
--data-name-list
Creditcard.csv
--model-name
catch.CATCH
--model-hyper-params
"{\"Mlr\": 1e-05, \"batch_size\": 128, \"cf_dim\": 64, \"d_ff\": 128, \"d_model\": 128, \"e_layers\": 3, \"head_dim\": 64, \"inference_patch_size\": 16, \"lr\": 0.0001, \"n_heads\": 16, \"num_epochs\": 5, \"patch_size\": 32, \"patch_stride\": 32, \"seq_len\": 192,\"bn_dims\": [128, 256, 512, 1024],  \"k_experts\": 2,  \"bn_loss_coef\": 0.0001}"
--gpus
0
--num-workers
1
--timeout
60000
--save-path
score/CATCH