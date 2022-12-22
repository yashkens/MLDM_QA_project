import re
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# this func is taken from official NaturalQuestions repo
def get_nq_tokens(simplified_nq_example):

    if "document_text" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

    return simplified_nq_example["document_text"].split(" ")


# this func is taken from official NaturalQuestions repo too
def simplify_nq_example(nq_example):

    def _clean_token(token):
        return re.sub(u" ", "_", token["token"])

    text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

    def _remove_html_byte_offsets(span):
        if "start_byte" in span:
            del span["start_byte"]

        if "end_byte" in span:
            del span["end_byte"]

        return span

    def _clean_annotation(annotation):
        annotation["long_answer"] = _remove_html_byte_offsets(
            annotation["long_answer"])
        annotation["short_answers"] = [
            _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation

    simplified_nq_example = {
      "question_text": nq_example["question_text"],
      "example_id": nq_example["example_id"],
      "document_url": nq_example["document_url"],
      "document_text": text,
      "long_answer_candidates": [
          _remove_html_byte_offsets(c)
          for c in nq_example["long_answer_candidates"]
      ],
      "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
    }

    if len(get_nq_tokens(simplified_nq_example)) != len(
      nq_example["document_tokens"]):
        raise ValueError("Incorrect number of tokens.")

    return simplified_nq_example


class YesNoAnswerDataset(Dataset):

    def __init__(self, data, simplify_fn, tokenizer, max_len, should_simplify, balance=True, test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simplify = should_simplify
        self.simplify_fn = simplify_fn
        self.balance = balance
        self.test = test
        self.label_dict = {"NO": 0, "YES": 1, "NONE": 2}
        qs, l_ans, labels = self.preprocess_data(data)

        self.questions = qs
        self.long_answers = l_ans
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        input_tokens = self.questions[idx]
        input_tokens += f" {self.tokenizer.sep_token} "
        input_tokens += self.long_answers[idx]
        encoding = self.tokenizer(input_tokens,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        if self.test:
            return encoding
        return encoding, self.labels[idx]

    def process_example(self, ex):
        simple_ex = ex
        if self.simplify:
            simple_ex = self.simplify_fn(ex)
        simple_ex['document_text'] = simple_ex['document_text'].replace('\ufeff', ' - ')
        tokens = simple_ex['document_text'].split()

        long_answer_info = simple_ex['annotations'][0]['long_answer']
        if long_answer_info['candidate_index'] == -1:
            return [], [], []

        long_answer = tokens[long_answer_info['start_token']:long_answer_info['end_token']]
        long_answer = ' '.join(long_answer)
        question = simple_ex['question_text']

        if self.test:
            return question, long_answer, ''

        yes_no_answer = simple_ex['annotations'][0]['yes_no_answer']
        label = self.label_dict[yes_no_answer]

        return question, long_answer, label

    def preprocess_data(self, data):
        questions, long_answers, labels = [], [], []
        for ex in tqdm(data):
            q, text, ans = self.process_example(ex)
            if not text:
                continue
            long_answers.append(text)
            labels.append(ans)
            questions.append(q)

        if self.balance:
            nones = np.where(np.array(labels) == self.label_dict["NONE"])
            random_nones = random.sample(list(nones[0]), 4000)
            normal_answers = np.where(np.array(labels) != self.label_dict["NONE"])
            normal_answers = list(normal_answers[0])
            normal_answers.extend(list(random_nones))
            random.shuffle(normal_answers)
            long_answers_new = []
            for i in range(len(long_answers)):
                if i in normal_answers:
                    long_answers_new.append(long_answers[i])
            #             long_answers = list(np.array(long_answers)[normal_answers])
            labels = list(np.array(labels)[normal_answers])
            questions = list(np.array(questions)[normal_answers])

        return questions, long_answers, labels
