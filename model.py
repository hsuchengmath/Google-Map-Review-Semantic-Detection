

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel




class Pairwise_Sentence_Model:
    def __init__(self, target_sent=None, threshold=0.95):
        # language model part
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = AutoModel.from_pretrained("bert-base-chinese")
        # similarity calculation part
        self.cos_fuc = nn.CosineSimilarity(dim=1, eps=1e-6)
        # init
        self.threshold = threshold 
        self.target_sent = target_sent
        self.target_sent_emb = self.multi_sent_to_emb_func(self.target_sent)


    def multi_sent_to_emb_func(self, multi_sent=None):
        inputs = self.tokenizer(multi_sent,padding=True,truncation=True, return_tensors="pt")
        multi_sent_emb = self.model(**inputs)[1] # it is torch[N,768] and N is sent num
        return multi_sent_emb


    def calculate_similarity(self, sent_emb=None, source_sent_unit_emb=None):
        similarity = self.cos_fuc(sent_emb, source_sent_unit_emb).tolist()
        return similarity # it is list and N element is target_sent


    def threshold_bar_determine(self, similarity=None):
        determine_bool, matched_sent = False, list()
        for i, sent in enumerate(self.target_sent):
            sim = similarity[i]

            if self.threshold is not None and sim >= self.threshold:
                determine_bool = True
                matched_sent.append((sim, sent))
            elif self.threshold is None:
                determine_bool = True
                matched_sent.append((sim, sent))
        if len(matched_sent) != 0:
            return determine_bool, max(matched_sent)[1]
        else:
            return determine_bool, matched_sent


    def forward(self, source_multi_sent=None):
        in_std_faq, match_std_feq, not_in_std_faq = set(), set(), set()
        source_multi_sent_emb = self.multi_sent_to_emb_func(source_multi_sent)
        for i, customer_sent in enumerate(source_multi_sent):
            sent_unit_emb = source_multi_sent_emb[i].view(1,-1)
            similarity = self.calculate_similarity(self.target_sent_emb, sent_unit_emb)
            determine_bool, matched_sent = self.threshold_bar_determine(similarity)
            if determine_bool is True:
                in_std_faq.add(customer_sent)
                match_std_feq.add(matched_sent)
            else:
                not_in_std_faq.add(customer_sent)
        return list(in_std_faq), list(match_std_feq), list(not_in_std_faq)


class Target_Label_Detector_base_on_PSM:
    def __init__(self, total_target_label=None):
        self.target_label_to_psm_obj = dict()
        for target_label in total_target_label:
            target_sent = [target_label+'是良好',target_label+'是普通',target_label+'是不好的',target_label+'是極差的']
            self.target_label_to_psm_obj[target_label] = Pairwise_Sentence_Model(target_sent=target_sent,threshold=None)


    def forward(self, target_label=None, source_sent=[None]):
        if target_label in self.target_label_to_psm_obj:
            psm_obj = self.target_label_to_psm_obj[target_label]
            _,matched_sent,_ = psm_obj.forward(source_multi_sent=source_sent)
            return matched_sent[0]
        else:
            return 0

