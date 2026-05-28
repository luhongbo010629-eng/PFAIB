python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "SWAT.csv" --model-name "catch.CATCH" --model-hyper-params '{"auxi_lambda": 0, "batch_size": 32, "inference_patch_size": 256, "inference_patch_stride": 32, "patch_size": 256, "patch_stride": 64, "score_lambda": 0, "seq_len": 2048}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/CATCH"

--config-path
"unfixed_detect_score_multi_config.json"
--data-name-list
"SWAT.csv"
--model-name
"catch.CATCH"
--model-hyper-params
"{\"auxi_lambda\": 0, \"batch_size\": 32, \"inference_patch_size\": 256, \"inference_patch_stride\": 32, \"patch_size\": 256, \"patch_stride\": 64, \"score_lambda\": 0, \"seq_len\": 2048,\"bn_dims\": [128,256,512,1024], \"k_experts\": 2, \"bn_loss_coef\": 0.01}"
--gpus
0
--num-workers
1
--timeout
60000
--save-path
"score/CATCH"