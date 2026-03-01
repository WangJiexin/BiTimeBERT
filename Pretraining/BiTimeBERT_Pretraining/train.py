import os
import math
import pickle 
import argparse
import collections
import multiprocessing
from tqdm import tqdm, trange
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from fairseq.tasks.temp_bert import TempBertTask
from transformers import AdamW
from transformers import BertTokenizer,BertConfig,BertForPreTraining
from torch.nn import CrossEntropyLoss
from transformers import BertModel
torch.cuda.set_device(0)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--bert_model', default="bert-base-cased", type=str)    
    parser.add_argument('--add_external_temp_tokens', action='store_true')
    parser.add_argument('--temp_granularity', default='Day', choices=['Year', 'Month', 'Week', 'Day'])
    parser.add_argument('--no_nsp', default=False, action='store_true')
    parser.add_argument('--model_type', default='mask_tempbert', choices=['mask_tempbert'])
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=2.0, type=float)
    parser.add_argument('--tokens_per_sample', default=512, type=int, help='max number of total tokens over all segments per sample for BERT dataset')
    parser.add_argument('--warmup_proportion', default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--temp_masked_ratio', default=0.3, type=float)
    
    return parser

class TempBert(nn.Module):
    def __init__(self, bert_model_name, cls_output_num):
        super(TempBert, self).__init__()
        self.config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertForPreTraining.from_pretrained(bert_model_name, config = self.config)
        self.bert.cls.seq_relationship = nn.Linear(self.config.hidden_size, cls_output_num)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_model_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states = True)
        return bert_model_output

def main(args):
    for temp_masked_ratio in [3]:
        args.temp_masked_ratio = temp_masked_ratio/10
        
        show_steps = 10000
        args.save_dir = os.path.join(args.save_dir, args.model_type)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        task = TempBertTask.setup_task(args)

        print("*********************Start Preparing Dataset*********************")
        # Load dataset splits
        task.load_dataset("train")
        #task.load_dataset("valid") 

        if args.model_type=="mask_tempbert":
            cls_output_num = task.date_num
        else:
            raise
        device = torch.device("cuda")
        model = TempBert(args.bert_model, cls_output_num)
        model.to(device)
        #model.bert.load_state_dict(state_dict=torch.load("..../.pt")) #load checkpoint here
        print('| model {}, model params {}'.format(args.bert_model, sum(p.numel() for p in model.parameters())))

        print(f"*********************Start Training {args.model_type} Model*********************")
        # Prepare DataLoader
        os.makedirs(args.save_dir, exist_ok=True)
        padding_idx = 0 #padding_idx in lm_label_ids
        train_dataset = task.datasets["train"]
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count())
        num_train_steps = int(len(train_dataset)/args.batch_size/args.gradient_accumulation_steps*args.num_train_epochs)
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate)

        logger.info("***** Running training *****")
        logger.info("Num of batch size = %d", args.batch_size)
        logger.info("Num of examples = %d", len(train_dataset))
        logger.info("Num of gradient_accumulation_steps = %d", args.gradient_accumulation_steps)
        logger.info("Num of epochs = %d", args.num_train_epochs)
        logger.info("Num of train steps: {}".format(num_train_steps))
        logger.info("Num of predict cls_output_num: {}".format(cls_output_num))

        global_step = 0
        before = 10
        learning_rate = args.learning_rate
        epoch_batch_results = dict() 
        epoch_results = [] 

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            in_epoch_step = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", position=0)):
                with torch.no_grad():
                    batch = (item.cuda(device=device) for item in batch)
                input_ids, attention_mask, token_type_ids, lm_label_ids, ts_ids, ts_labels, index = batch
                pred_labels = ts_labels
                    
                bert_model_output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                prediction_scores = bert_model_output.prediction_logits
                seq_relationship_score = bert_model_output.seq_relationship_logits
                lm_loss_fct = CrossEntropyLoss(ignore_index=padding_idx)
                pred_loss_fct = CrossEntropyLoss()
                masked_lm_loss = lm_loss_fct(prediction_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
                pred_loss = pred_loss_fct(seq_relationship_score.view(-1, cls_output_num), pred_labels.view(-1))
                loss = masked_lm_loss + pred_loss 

                if step%show_steps==0:
                    if step==0:
                        avg_tr_loss=loss.item()
                    else:
                        avg_tr_loss=tr_loss/step*args.gradient_accumulation_steps
                    epoch_batch_results[f"{epoch}-{step}"] = [loss.item(), masked_lm_loss.item(), pred_loss.item(), avg_tr_loss]
                    print('Epoch[{}] Batch[{}](batch_size:{}) MaskTempRatio:{} - loss:{:.6f} - masked_lm_loss: {:.6f} - pred_loss: {:.6f}'.format(epoch, step, args.batch_size , args.temp_masked_ratio, loss.item(), masked_lm_loss.item(), pred_loss.item()))
                    print(f'average tr_loss:{avg_tr_loss}')
                if args.gradient_accumulation_steps > 1:
                    loss = loss/args.gradient_accumulation_steps
                tr_loss += loss.item()
                loss.backward()
                
                if (step+1)%args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    #num_train_steps = int(len(train_dataset)/args.batch_size/args.gradient_accumulation_steps*args.num_train_epochs)
                    if global_step/num_train_steps < args.warmup_proportion:
                        lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    in_epoch_step += 1

            averloss=tr_loss/in_epoch_step 
            epoch_results.append(averloss) 
            print("epoch: %d\taverageloss: %f\tglobal_step: %d "%(epoch,averloss,global_step))
            print("current learning_rate: ", learning_rate)
            if global_step/num_train_steps > args.warmup_proportion and averloss > before - 0.01:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                    learning_rate = param_group['lr']
                print("Decay learning rate to: ", learning_rate)
            before=averloss

            if True:
                # Save a trained model
                logger.info(f"********** Saving Model At Epoch {epoch} **********")
                checkpoint_prefix = 'checkpoint' + str(args.temp_masked_ratio).replace(".","_") + str(epoch)
                output_dir = os.path.join(args.save_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_output_dir = output_dir + f'/{args.model_type}.pt'
                torch.save(model.bert.state_dict(), model_output_dir)

            save_results = [epoch_batch_results, epoch_results] 
            with open(os.path.join(args.save_dir,"save_results.pkl"), 'wb') as f:
                pickle.dump(save_results, f)
        print(epoch_results)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

