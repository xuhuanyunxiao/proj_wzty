#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division
import re
import jieba
from string import digits

stopwords = {}
stw = open("corpus/stop_words_cor.txt", encoding='UTF-8')
for ws in stw:
    ws = ws.replace("\n", "")
    ws = ws.replace("\r", "")
    stopwords[ws] = 1
stw.close()

jieba.load_userdict('corpus/company.txt')
jieba.load_userdict('corpus/user_dict.txt')
jieba.load_userdict('corpus/bank_dict.txt')


def handle_contents(l_contents):
    lines = []
    for line in l_contents:
        lines.append(handle_content(line))
    return lines


def handle_content(content):
    content = str(content)
    raw = content.strip()
    line = ""
    if raw != "":
        word_list_1 = []
        line = ""
        remove_words = []
        raw = clear_sen(raw)
        word_list = filter(lambda x: len(x) > 0, map(etl, jieba.cut(raw, cut_all=False)))
        ll = list(word_list)
        for wd in ll:
            if wd in stopwords:
                remove_words.append(wd)

        for l in ll:
            if not (l in remove_words):
                word_list_1.append(l)

        for wd in word_list_1:
            line = line + wd + " "
    return line


def clear_sen(sent):
    sent = sent.replace("\n", "")
    sent = sent.replace('\r','')
    sent = sent.replace('\r\n','')
    reobj = re.compile('//@(.*?)[:\s]')
    sent = reobj.sub("", sent)
    reobj = re.compile("@(.*?)[:\s]")
    sent = reobj.sub("", sent)
    reobj = re.compile(r"\[[^\[\]]*?\]")
    sent = reobj.sub("", sent)

    sent = sent.replace("，", ",")
    sent = sent.replace("。", ".")
    sent = sent.replace("！", "!")
    sent = sent.replace("？", "?")
    reobj = re.compile("//(.*?)[:\s]")
    sent = reobj.sub("", sent)
    return sent


def etl(s):  # remove 标点和特殊字符
    regex = re.compile(r"[^\u4e00-\u9f5aa-zA-Z0-9]")
    s = regex.sub('', s)
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res