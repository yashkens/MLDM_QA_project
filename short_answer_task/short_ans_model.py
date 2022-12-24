import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import f1_score
from torch.nn import DataParallel


class ShortAnswerModel:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model = DataParallel(self.model).to(device)

    def __call__(self, tokenized_input, question_lens):

        self.model.eval()
        with torch.no_grad():

            ids = tokenized_input['input_ids'].to(self.device).squeeze(dim=1)
            mask = tokenized_input['attention_mask'].to(self.device).squeeze(dim=1)

            output = self.model(input_ids=ids, attention_mask=mask)
            logits = output['logits']
            pred = torch.argmax(logits, axis=-1)

            ind_preds = []
            for i in range(len(pred)):
                start, end = self.get_start_end_tokens(pred[i],
                                                       tokenized_input['offset_mapping'][i],
                                                       question_lens[i])
                ind_preds.append([start, end])

        return ind_preds

    def plot_log(self, train_losses, train_fscores, val_losses, val_fscores, log_name):
        clear_output()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(9, 3))
        fig.suptitle('Training Log', fontsize=8)
        ax1.plot(train_losses)
        ax1.set_title('Train Loss', fontsize=8)
        ax1.tick_params(labelsize=6)
        ax2.plot(train_fscores)
        ax2.set_title('Train F1', fontsize=8)
        ax2.tick_params(labelsize=6)

        ax3.plot(val_losses)
        ax3.set_title('Val Loss', fontsize=8)
        ax3.tick_params(labelsize=6)
        ax4.plot(val_fscores)
        ax4.set_title("Val F1", fontsize=8)
        ax4.tick_params(labelsize=6)
        plt.savefig(f'log_{log_name}.png')
        plt.show()

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss, val_fscore = 0, 0
        golds, preds = [], []

        with torch.no_grad():
            for batch in val_dataloader:

                tokens, labels, question_lens = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device).squeeze(dim=1)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                val_loss += loss.item()

                pred = torch.argmax(logits, axis=-1)

                for i in range(len(labels)):
                    gold_start, gold_end = self.get_start_end_tokens(labels[i],
                                                                     tokens['offset_mapping'][i],
                                                                     question_lens[i])

                    pred_start, pred_end = self.get_start_end_tokens(pred[i],
                                                                     tokens['offset_mapping'][i],
                                                                     question_lens[i])

                    golds.append(str(gold_start) + ':' + str(gold_end))
                    preds.append(str(pred_start) + ':' + str(pred_end))

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_f1 = f1_score(golds, preds, average='micro')
        return avg_val_loss, avg_val_f1

    def train(self, train_dataloader, val_dataloader, n_epoch, optimizer, checkpoint_step, model_save_name):

        train_losses, val_losses = [], []
        train_fscores, val_fscores = [], []

        for epoch in range(n_epoch):

            start_time = time.time()

            self.model.train()

            train_loss, train_fscore = 0, 0

            golds, preds = [], []
            step_train_losses, step_train_fscores, step_val_losses, step_val_fscores = [], [], [], []
            for step_num, batch in enumerate(train_dataloader):

                tokens, labels, question_lens = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device).squeeze(dim=1)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits, axis=-1)

                for i in range(len(labels)):
                    gold_start, gold_end = self.get_start_end_tokens(labels[i],
                                                                     tokens['offset_mapping'][i],
                                                                     question_lens[i])

                    pred_start, pred_end = self.get_start_end_tokens(pred[i],
                                                                     tokens['offset_mapping'][i],
                                                                     question_lens[i])

                    golds.append(str(gold_start) + ':' + str(gold_end))
                    preds.append(str(pred_start) + ':' + str(pred_end))

                fscore = f1_score(golds, preds, average='micro')
                train_fscore += fscore
                # print(f"Step num: {step_num}")
                # print(f"Loss: {train_loss / (step_num + 1):.5f}")
                # print(f"F1: {fscore:.4f}")

                if step_num % 200 == 0:
                    print(f"Step №{step_num}!")

                if step_num % checkpoint_step == 0:
                    step_train_losses.append(train_loss / (step_num + 1))
                    step_train_fscores.append(train_fscore / (step_num + 1))
                    print(f"Step №{step_num}")
                    print("Running validation...")
                    val_loss_tmp, val_f1_tmp = self.validate(val_dataloader)
                    if len(step_val_fscores) > 0 and step_val_fscores[-1] < val_f1_tmp:
                        torch.save(self.model.state_dict(), model_save_name)
                    step_val_losses.append(val_loss_tmp)
                    step_val_fscores.append(val_f1_tmp)
                    self.plot_log(step_train_losses, step_train_fscores,
                                  step_val_losses, step_val_fscores, f"{epoch}_{step_num}")
                    print(f"Train loss: {step_train_losses[-1]:.4f}")
                    print(f"Train F-score: {step_train_fscores[-1]:.4f}")
                    print(f"Val loss: {val_loss_tmp:.4f}")
                    print(f"Val F-score: {val_f1_tmp:.4f}")

            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_f1 = f1_score(golds, preds, average='micro')

            avg_val_loss, avg_val_f1 = self.validate(val_dataloader)

            train_losses.append(avg_train_loss)
            train_fscores.append(avg_train_f1)
            val_losses.append(avg_val_loss)
            val_fscores.append(avg_val_f1)

            self.plot_log(train_losses, train_fscores, val_losses, val_fscores, f"EPOCH_{epoch}")

            print(f'Epoch {epoch}')
            print(f'Train loss: {avg_train_loss:.3f}')
            print(f'Train micro F1: {avg_train_f1:.3f}')
            print(f'Validation loss: {avg_val_loss:.3f}')
            print(f'Validation micro F1: {avg_val_f1:.3f}')
            curr_time = time.time() - start_time
            print(f'Epoch time: {curr_time:.3f}s')

    def get_start_end_tokens(self, bio_tags, offset_mapping, question_len):

        token_mapping = self.get_tokenization_mapping(offset_mapping)

        answer_inds = torch.where(bio_tags != 0)[0].cpu()

        true_mask = []
        for ind in range(len(answer_inds)):
            if ind == 0:
                true_mask.append(answer_inds[ind])
                continue
            if answer_inds[ind - 1] != answer_inds[ind] - 1:
                break
            true_mask.append(answer_inds[ind])
        true_mask = np.array(true_mask)

        answer_tokens = []
        for ind in true_mask:
            for group_ind, token_group in enumerate(token_mapping):
                if ind in token_group:
                    answer_tokens.append(group_ind)
                    break
        if answer_tokens:
            start = answer_tokens[0] - question_len
            end = answer_tokens[-1] - question_len + 1
        else:
            start = -1
            end = -1
        return start, end

    # TODO: don't duplicate this function!
    def get_tokenization_mapping(self, offset_mapping):
        d = []
        for i, pair in enumerate(offset_mapping[0]):
            if pair[0] == 0 and pair[1] == 0:
                continue
            if pair[0] == 0:
                d.append([i])
            else:
                if len(d) == 0:
                    d.append([i])
                else:
                    d[-1].append(i)
        return d
