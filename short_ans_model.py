import time
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import f1_score


class ShortAnswerModel():

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, dataloader):
        # TODO
        return None

    def plot_log(self, train_losses, val_losses, val_fscores):
        clear_output()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
        fig.suptitle('Training Log', fontsize=8)
        ax1.plot(train_losses)
        ax1.set_title('Train Loss', fontsize=8)
        ax1.tick_params(labelsize=6)
        ax2.plot(val_losses)
        ax2.set_title('Val Loss', fontsize=8)
        ax2.tick_params(labelsize=6)
        ax3.plot(val_fscores)
        ax3.set_title("Val F1", fontsize=8)
        ax3.tick_params(labelsize=6)
        plt.show()

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss, val_fscore = 0, 0

        with torch.no_grad():
            for batch in val_dataloader:
                tokens, labels, long_starts, question_lens = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device).squeeze(dim=1)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                val_loss += loss.item()

                flattened_gold = labels.view(-1)
                active_logits = logits.view(-1, self.model.num_labels)
                flattened_pred = torch.argmax(active_logits, axis=1)

                mask = labels.view(-1) != -100
                gold = torch.masked_select(flattened_gold, mask)
                pred = torch.masked_select(flattened_pred, mask)

                fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), average='micro')
                # TODO: compare start+end indices, not token labels
                val_fscore += fscore

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_f1 = val_fscore / len(val_dataloader)
        return avg_val_loss, avg_val_f1

    def train(self, train_dataloader, val_dataloader, n_epoch, optimizer):

        train_losses, val_losses = [], []
        val_fscores = []

        for epoch in range(n_epoch):

            start_time = time.time()

            self.model.train()

            train_loss, train_fscore = 0, 0

            for batch in train_dataloader:
                tokens, labels, long_starts, question_lens = batch
                ids = tokens['input_ids'].to(self.device).squeeze(dim=1)
                mask = tokens['attention_mask'].to(self.device).squeeze(dim=1)
                labels = labels.to(self.device).squeeze(dim=1)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output['loss']
                logits = output['logits']

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                # compute accuracy
                flattened_gold = labels.view(-1)
                active_logits = logits.view(-1, self.model.num_labels)
                flattened_pred = torch.argmax(active_logits, axis=1)

                mask = labels.view(-1) != -100
                gold = torch.masked_select(flattened_gold, mask)
                pred = torch.masked_select(flattened_pred, mask)

                fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), average='micro')
                # TODO: compare start+end indices, not token labels
                train_fscore += fscore

            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_f1 = train_fscore / len(train_dataloader)

            avg_val_loss, avg_val_f1 = self.validate(val_dataloader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_fscores.append(avg_val_f1)

            self.plot_log(train_losses, val_losses, val_fscores)

            print(f'Epoch {epoch}')
            print(f'Train loss: {avg_train_loss:.3f}')
            print(f'Train micro F1: {avg_train_f1:.3f}')
            print(f'Validation loss: {avg_val_loss:.3f}')
            print(f'Validation micro F1: {avg_val_f1:.3f}')
            curr_time = time.time() - start_time
            print(f'Epoch time: {curr_time:.3f}s')

    def get_start_end_tokens(self, bio_tags, offset_mapping, long_start, question_len):

        token_mapping = self.get_tokenization_mapping(offset_mapping)

        answer_inds = torch.where(bio_tags != 2)[0]
        answer_tokens = []
        # TODO: check what's going on with inds here
        for ind in answer_inds:
            for ind, token_group in enumerate(token_mapping):
                if ind in token_group:
                    answer_tokens.append(ind)
                    break

        start = answer_tokens[0] + long_start - question_len
        end = answer_tokens[-1] + long_start - question_len + 1
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
                d[-1].append(i)
        return d
