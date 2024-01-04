# MALCOM
#### Source code for paper: Thai Le, Suhang Wang, Dongwon Lee. MALCOM: Generating Malicious Comments to Attack Neural Fake News Detection Models 20th IEEE International Conference on Data Mining (ICDM), Virtual, November 17-20, 2020.
https://ieeexplore.ieee.org/abstract/document/9338282/

##### Better Late than Never. Sorry all for uploading all the codes late. 
##### Thanks other author(s) that have utilized and contacted for feedbacks on the codes.

# Instruction

#### A few important notes on prerequisite
Nvidia-Driver and CUDA version:  
NVIDIA-SMI 530.30.02  
Driver Version: 530.30.02
CUDA Version: 12.1

python==3.5  
scikit-learn==0.21.3  
pytorch==2.0.0+cu117  

### MLE (Pre-training) Training (yes, this is before we have all the fancy pre-trained models :))
python main.py --dataset rumors20x --vocab_size 15000 --max_num_data 100000 --load_gen_pretrain 0 --gen_pretrain 1 --batch_size 8 --mle_batch_size 8 --max_num_eval 128

* if you run on modern GPU with a lot of storage, you can increase the --mle_batch_size to increase the speed (but not too much). However, you should keep --batch_size sufficiently low such that there are at least a total of batches for evaluationg during training.

* Training should show better metrics overtime, such as below:  
[MLE-GEN] epoch 0 : pre_loss = 7.8850, BLEU-[2] = [0.139], NLL_gen = 4.4054 | AvgTime: 9.1916  
[MLE-GEN] epoch 50 : pre_loss = 3.0950, BLEU-[2] = [0.13], NLL_gen = 3.0191 | AvgTime: 9.2615  
...  
Saved MLE model to .//pretrain/rumors20x/gen_MLENEW_pretrain_rumors20x_relgan_RMC_sl20_sn10000_hs256_maxsn100000_vs15003.pt

### Fake News Classifier Training
```
python main.py --dataset rumors20x --max_num_data 100000 \
--load_gen_pretrain 0 --gen_pretrain 0 \
--batch_size 64 --flg_train_f 1 --f_batch_size 64 \
--f_dropout_comment 0.59 --f_dropout_content 0.95 --f_type RNN2 \
--F_patience_epoch 3 --max_num_comment 10 --f_embed_dim 64 --F_clf_lr 0.01
```

* Training should improve the fake news classifier overtime. It will early-stop when meeting criteria for overfitting prevention  
[PRE-CLAS] epoch 0: c_loss = 0.7504, c_acc = 0.5959, eval_loss = 0.6265, eval_acc = 0.6615, min_eval_loss = 0.6265  
[PRE-CLAS] epoch 1: c_loss = 0.5689, c_acc = 0.7170, eval_loss = 0.4599, eval_acc = 0.7884, min_eval_loss = 0.4599  
[PRE-CLAS] epoch 2: c_loss = 0.3318, c_acc = 0.8657, eval_loss = 0.3175, eval_acc = 0.8721, min_eval_loss = 0.3175  
[PRE-CLAS] epoch 3: c_loss = 0.2039, c_acc = 0.9252, eval_loss = 0.3193, eval_acc = 0.8924, min_eval_loss = 0.3175  
...  
[PRE=CLAS] SAVED classifier to: .//pretrain/rumors20x/F_pretrain_relgan_RMC_sl20_sn10000_vs15003_10cmt_GloveFalse_RNN2.pt.0.1861B

### GAN Training
```
python main.py --dataset rumors20x --max_num_data 100000 \
--load_gen_pretrain 1 --gen_pretrain 0 \
--flg_train_f 0 --f_embed_dim 64 --f_batch_size 128 \
--f_dropout_comment 0.59 --f_dropout_content 0.95 --f_type RNN2 --F_patience_epoch 3 --max_num_comment 10  --F_clf_lr 0.01 \
--attack_f_path ./pretrain/rumors20x/F_pretrain_relgan_RMC_sl20_sn10000_vs15003_10cmt_GloveFalse_RNN2.pt.0.1861B \
--batch_size 128 --h_batch_size 32 --g_f_step 3 --g_f_alpha 0.1 --head_size 256
```

This will save the model to a specific folder. Here I have "./save/20240104/rumors20x/relgan_RMC_lt-rsgan_sl20_temp1000_T0104_1828_44/gen_ADV-F_00100.pt" as an example but only after 100 iterations.

* prefix and suffix "h": often denoted as parameters for the topic discriminator
* prefix and suffix "d": often denoted as parameters for the style discriminator (D in GAN)
* prefix and suffix "g": often denoted as parameters for the generator (G in GAN)
* I turned off all the visualization via visdom, but please uncomment all "*visualization*" lines if you want to do so


### Evaluation (BlackBox)
```
python main.py --dataset rumors20x \
--max_num_data 100000 \
--load_gen_pretrain 1 --gen_pretrain 0 \
--attack_f_path ./pretrain/rumors20x/F_pretrain_relgan_RMC_sl20_sn10000_vs15003_10cmt_GloveFalse_RNN2.pt.0.1861B \
--f_type RNN2 --f_embed_dim 64 \
--eval_g_path ./save/20240104/rumors20x/relgan_RMC_lt-rsgan_sl20_temp1000_T0104_1828_44/models/gen_ADV-F_00100.pt \
--if_eval_score 0 --if_eval 1 --eval_num 3 \
--num_topics 14 --flg_adv_t 0 \
--max_num_eval 2000 --flg_adv_f 1 \
--max_num_comment_test 1 --max_num_comment 10 \
--atk_num_comment 1 --batch_size 128 \
--blackbox_using_path ./pretrain/rumors20x/F_pretrain_relgan_RMC_sl20_sn10000_vs15003_10cmt_GloveFalse_RNN2.pt.0.1861B \
--flg_eval_g 1 --if_show_sentence 1
```

#### Example of Outputs

```
Title:  rcmp advises people to stay away from parliament hill due to ongoing police incident .
ID:   rumors-4128
User: advises people to stay away from hill due to ongoing police incident .
User: rt advises people to stay away from parliament hill due to ongoing police incident .,
User: advises people to stay away from parliament hill due to ongoing police incident . ”
User: because god forbid they do their job and protect those reporters from the active shooter on the loose ...
User: you mean like police pointing guns are entire groups of reporters ? does not sound safe at all .
User: advises people to stay away from parliament hill due to ongoing police incident . ”
User: my <unk> tom camp myself live close by parliament hill over here in ottawa , <unk> thank u
User: and we have seen a dozen of every level of police <unk> over to parliament <unk> memorial
User: keep yourselves and the public safe !
User: so is 3rd hand information . which is known as hearsay ... yet no actual complaint made hmm ...
Malcom: escaping goodness god goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness goodness
True Label:0
Before Attack: Real
After Attack: Fake
````
The results from Malcom will be better when the **GAN Training** step converges. Please refer full details in the paper.

### Citation
```
@inproceedings{le2020malcom,
  title={Malcom: Generating malicious comments to attack neural fake news detection models},
  author={Le, Thai and Wang, Suhang and Lee, Dongwon},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  pages={282--291},
  year={2020},
  organization={IEEE}
}
```




