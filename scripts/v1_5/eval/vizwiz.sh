#!/bin/bash
CKPT=avg-llava
python -m llava.eval.model_vqa_loader \
    --model-path zhibinlan/AVG-LLaVA \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}.json
