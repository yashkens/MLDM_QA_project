import re

from torch.utils.data import Dataset
from nltk.corpus import stopwords
from sklearn.utils import resample, shuffle
import pandas as pd


class LongAnswerDatasetBase(Dataset):
    SAMPLE_RATE = 5
    HTML_PATTERN = re.compile(r'<.*?>')
    
    def _get_nq_tokens(self, simplified_nq_example):
        if "document_text" not in simplified_nq_example:
            raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

        return simplified_nq_example["document_text"].split(" ")
    
    def _clean_token(self, token):
        return re.sub(u" ", "_", token["token"])

    def _remove_html_byte_offsets(self, span):
        if "start_byte" in span:
            del span["start_byte"]

        if "end_byte" in span:
            del span["end_byte"]

        return span

    def _clean_annotation(self, annotation):
        annotation["long_answer"] = self._remove_html_byte_offsets(
            annotation["long_answer"])
        annotation["short_answers"] = [
            self._remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation
    
    def _simplify_nq_example(self, nq_example):
        text = " ".join([self._clean_token(t) for t in nq_example["document_tokens"]])

        simplified_nq_example = {
          "question_text": nq_example["question_text"],
          "example_id": nq_example["example_id"],
          "document_url": nq_example["document_url"],
          "document_text": text,
          "long_answer_candidates": [
              self._remove_html_byte_offsets(c)
              for c in nq_example["long_answer_candidates"]
          ],
          "annotations": [self._clean_annotation(a) for a in nq_example["annotations"]]
        }

        if len(self._get_nq_tokens(simplified_nq_example)) != len(
          nq_example["document_tokens"]):
            raise ValueError("Incorrect number of tokens.")

        return simplified_nq_example
    
    def _get_question_and_document(self, line):
        question = line['question_text']
        text = line['document_text'].split(' ')
        long_answer_candidates = line['long_answer_candidates']
        example_id = line['example_id']

        return question, text, long_answer_candidates, example_id


    def _get_long_candidate(self, i, annotations, candidate):
        if i == annotations['long_answer']['candidate_index']:
            label = 1
        else:
            label = 0

        long_start = candidate['start_token']
        long_end = candidate['end_token']

        return label, long_start, long_end
    
    def _preprocess_data(self, data):
        rows = []

        for line in data:
            if not self._kaggle_format:
                line = self._simplify_nq_example(line)
            question, text, long_answe, example_id = self._get_question_and_document(line)
            for i, candidate in enumerate(line['long_answer_candidates']):
                label, long_start, long_end = self._get_long_candidate(i, annotations, candidate)

                if label == True or (i % self.SAMPLE_RATE == 0):
                    rows.append(
                        self._form_data_row(question, label, text, long_start, long_end)
                    )

        return pd.DataFrame(rows)
    
    def _remove_stopwords(self, sentence):
        words = sentence.split()
        words = [word for word in words if word not in stopwords.words('english')]

        return ' '.join(words)

    def _remove_html(self, sentence):
        return  self.HTML_PATTERN.sub(r'', sentence)

    def _clean_df_by_column(self, df, column):
        # df[column] = df[column].apply(lambda x : self._remove_stopwords(x))
        df[column] = df[column].apply(lambda x : self._remove_html(x))
        return df

    def _clean_df(self, df):
        df = self._clean_df_by_column(df, 'long_answer')
        df = self._clean_df_by_column(df, 'question')
        return df
    
    def __getitem__(self, idx):
        raise NotImplementedError('method __getitem__ is not implemented')
    
    def __len__(self):
        raise NotImplementedError('method __len__ is not implemented')
        
        
class TrainLongAnswerDataset(LongAnswerDatasetBase):
    def __init__(self, data, tokenizer, max_len=150, kaggle_format=True, balance=True):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._kaggle_format = kaggle_format
        
        data = self._preprocess_data(data)
        data = self._clean_df(data)
        if balance:
            data = self._balance_data(data)
        self._questions = data.question.values
        self._long_answers = data.long_answer.values
        self._targets = data.is_long_answer.values
        
    
    def _balance_data(self, data):
        data_unbalanced_majority = data[data.is_long_answer == 0]
        data_unbalanced_minority = data[data.is_long_answer == 1]
        majority_size = data_unbalanced_majority.shape[0]
        data_balanced_minority = resample(data_unbalanced_minority, 
                                          replace=True,
                                          n_samples=majority_size)
        data_balanced = pd.concat([data_unbalanced_majority, data_balanced_minority])
        return shuffle(data_balanced).reset_index(drop=True)
        

    def _form_data_row(self, question, label, text, long_start, long_end):
        row = {
            'question': question,
            'long_answer': ' '.join(text[long_start:long_end]),
            'is_long_answer': label,
        }

        return row
    
    def __getitem__(self, idx):
        input_tokens = self._questions[idx].split()
        input_tokens.append(' ' + self._tokenizer.sep_token + ' ')
        long_answer_tokens = self._long_answers[idx].split()
        input_tokens.extend(long_answer_tokens)
        encoding = self._tokenizer(input_tokens,
                          is_split_into_words=True,
                          return_offsets_mapping=False,
                          return_token_type_ids=False,
                          padding='max_length',
                          truncation=True,
                          max_length=self._max_len,
                          return_tensors='pt')
        return encoding, self._targets[idx]

    def __len__(self):
        return self._targets.shape[0]
    
    
class TrainLongAnswerDatasetDicreasing(TrainLongAnswerDataset):
    def _balance_data(self, data):
        data_unbalanced_majority = data[data.is_long_answer == 0]
        data_unbalanced_minority = data[data.is_long_answer == 1]
        minority_size = data_unbalanced_minority.shape[0]
        data_balanced_majority = resample(data_unbalanced_majority, 
                                          replace=False,
                                          n_samples=minority_size * 2)
        data_balanced = pd.concat([data_balanced_majority, data_unbalanced_minority])
        return shuffle(data_balanced).reset_index(drop=True)
    
    
class TestLongAnswerDataset(LongAnswerDatasetBase):
    def __init__(self, data, tokenizer, max_len=150, kaggle_format=True):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._kaggle_format = kaggle_format
        
        data = self._preprocess_data(data)
        data = self._clean_df(data)
        self._data = {index: question_df for index, (question, question_df) in enumerate(data.groupby('question'))}

    def _form_data_row(self, question, label, text, long_start, long_end):
        row = {
            'question': question,
            'long_answer': ' '.join(text[long_start:long_end]),
            'is_long_answer': label,
            'long_start': long_start,
            'long_end': long_end
        }

        return row

    def __getitem__(self, idx):
        labels, texts, indices = [], [], []
        current_data = self._data[idx]
        for i in range(current_data.shape[0]):
            question = current_data.question.iloc[i]
            answer = current_data.long_answer.iloc[i]
            start = current_data.long_start.iloc[i]
            end = current_data.long_end.iloc[i]
            
            texts.append(question + self._tokenizer.sep_token + answer)
            labels.append(current_data.is_long_answer.iloc[i])
            indices.append(f"{start}:{end}")
            
        encoding = self._tokenizer(texts,
                                   return_offsets_mapping=False,
                                   return_token_type_ids=False,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self._max_len,
                                   return_tensors='pt')
        return encoding, labels, indices
   
    def __len__(self):
        return len(self._data)
