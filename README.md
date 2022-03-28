# Non-autoregressive Translation with Dependency-Aware Decoder
The code for paper Non-autoregressive Translation with Dependency-Aware Decoder

Some codes are borrowed from GLAT (https://github.com/FLC777/GLAT).

## Requirements
* Python >= 3.7
* Pytorch >= 1.5.0
* Fairseq 1.0.0a0

## Data Preparation
### Download distilled training data
For WMT14EN-DE, the dataset can be downloaded at [WMT14EN-DE](http://statmt.org/wmt14/translation-task.html#Download).

For WMT16EN-RO and IWSLT16DE-EN, the datasets can be downloaded at [IWSLT16EN-DE](https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing) and [WMT16EN-RO](https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing)

Create a new folder "data-bin" and download the datasets into it.

### Binarize the distilled training data
Take the WMT14En-De dataset as example:

```
input_dir=data-bin/wmt14ende
data_dir=data-bin/wmt14ende/bin
src=en
tgt=de
python3 fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir}/ \
    --workers 32 --src-dict ${input_dir}/dict.${src}.txt --tgt-dict {input_dir}/dict.${tgt}.txt
 ```
 
## Model Training
### Training VanillaNAT w/ DAD
The parameter "--input-transform" means using our method "Input Transformation".

The parameter "curriculum-type" refers to the choice of different training phase in our method "Forward Backward Dependency Modeling", which can be set to "at-forward", "at-backward", and "nat" sequentially. Except for the first phase, the parameter "--reset-optimizer" should be set in other training phases.

 ```
python train.py ${data_dir} --arch vanilla_nat --noise full_mask --share-all-embeddings \
    --criterion nat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 \
    --save-dir ${save_path} --src-embedding-copy --pred-length-offset --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_winth_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir dad_plugins \
    --curriculum-type nat --choose-data ende --input-transform
 ```
### Training GLAT w/ DAD

```
python3 train.py ${data_dir} --arch glat --noise full_mask --share-all-embeddings \
    --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_winth_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir dad_plugins \
    --curriculum-type nat --choose-data ende --input-transform
```

### Inference
* VanillaNAT w/ DAD
```
checkpoint_path=path_t_checkpoint
python3 fairseq_cli/generate.py ${data_dir} --path ${checkpoint_path} --user-dir dad_plugins \
    --task translation_lev --remove-bpe --max-sentences 20 --source-lang ${src} --target-lang ${tgt} \
    --quiet --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test
```

* GLAT w/ DAD
```
checkpoint_path=path_t_checkpoint
python3 fairseq_cli/generate.py ${data_dir} --path ${checkpoint_path} --user-dir dad_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang ${src} --target-lang ${tgt} \
    --quiet --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test
```
