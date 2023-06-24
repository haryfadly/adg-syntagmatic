from flask import Flask, redirect, url_for, render_template, request, session, flash
from datetime import timedelta
from collections import Counter

import os
import re
import string
import requests
import pyinflect
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag, word_tokenize, sent_tokenize

## ssl for nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

## uncomment for download
# nltk.download('universal_tagset')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

app = Flask (__name__)
app.secret_key = "spasi"
app.permanent_session_lifetime = timedelta(minutes=5)

# global sentence
# global correct_answer, words, word, target_word, collocations, collocation, scores, parameters, word_pos, rels, relation, basis_candidates, candidates, basis_word, basis_pos, target_pos, datas, result, distractors, api_request_sent, api_request_list
# global word_pos_list, rels_list, relation_list, candidates_list, basis_word_list, basis_pos_list

def get_sentence(passage, target_word):
    # global sentence, clause
    #join passage in one line
    sentences = " ".join(line.strip() for line in passage.splitlines())

    #split sentences with target word
    sentence_target = [sent + '.' for sent in sentences.split('.') if target_word in sent]
    sentence = str(sentence_target[0])

    #split clause with target word
    clauses = [clau for clau in sentence.split(',') if target_word in clau]
    clause = str(clauses[0])

    # sentences_list = sent_tokenize(sentences)
    return sentence, clause

def preprocess_text(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text.lower())
    # tokens = nltk.word_tokenize(text)
    
    # Remove stop words and punctuations (can not remove (') due to word particle)
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    word_to_tag = [i for i in tokens if i not in stop_words and i not in exclude]

    # # Lemmatize the words
    wordnet_lemmatizer = WordNetLemmatizer()
    words = pos_tag((word_to_tag), tagset = 'universal')
    preprocessed_text = []
    for word, tag in words:
        match tag:
                case "VERB":
                    wpos = "v"
                case "NOUN":
                    wpos = "n"
                case "ADV":
                    wpos = "r"
                case "ADJ":
                    wpos = "a"
                case _:
                    wpos = ""
        if wpos != "":
            preprocessed_text.append(wordnet_lemmatizer.lemmatize(word,wpos))
        else:
            preprocessed_text.append(wordnet_lemmatizer.lemmatize(word))

    # words = [wordnet_lemmatizer.lemmatize(word) for word in words]

    # # Stem the words (can not perform well)
    # stemmer = SnowballStemmer("english")
    # words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    text = " ".join(preprocessed_text)
    return text

def get_one_word(words):
    ## find word pos from collocation sent
    word_pos = []
    word_pos = pos_tag(word_tokenize(words), tagset = 'universal')

    # take verb and noun first as main word in collocation
    word_list = [] 
    tag = ""
    for word, tag in word_pos:
        if tag == "VERB":
            word_list.append(word)
        elif tag == "NOUN":
            word_list.append(word)

    if len(word_list) == 0:
        for word, tag in word_pos:
            if tag == "ADJ":
                word_list.append(word)
            elif tag == "ADV":
                word_list.append(word)

    return word_list[0]

def get_collocations(passage, stem, pmi_rank):
    sp = spacy.load('en_core_web_sm')
    ## get target word from stem
    # target_word is correct_answer
    target = re.findall('"([^"]*)"', stem)
    target_word = target[0]

    ## get sentence from passage
    (sentence, clause) = get_sentence(passage, target_word)

    ## delete punctuation
    tokens = nltk.word_tokenize(clause.lower())
    exclude = set(string.punctuation)
    words = [i for i in tokens if i not in exclude]

    ## find collocation based on target_word
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(words)
    finder.apply_ngram_filter(lambda w1, w2, w3: target_word not in (w1, w2, w3))

    ## for select collocation 1 pmi rank
    # for i in finder.nbest(trigram_measures.pmi,1):
    #     collocation = (' '.join(i))
    # else:
    #     word_pos = pos_tag(word_tokenize(collocation), tagset = 'universal')
    
    collocations = []
    # collocation = []
    scores = []
    f = 1
    word_pos=[]
    while f <=3:
        for i in finder.nbest(trigram_measures.pmi,f):
            found = (' '.join(i))
        if found not in collocations:
            collocations.append(found)
        if f == pmi_rank:
            word_pos = pos_tag(word_tokenize(found), tagset = 'universal')
            # collocation = found
        f +=1

    ## get scoring
    scores = finder.score_ngrams(trigram_measures.pmi)

    ## find target pos
    target_pos = []
    for word, tag in word_pos:
        if word == target_word:
            target_pos = tag

    ## find spacy_tag
    sp_sentence = sp(sentence)
    for word in sp_sentence:
        if word.text == target_word:
            # target_word_POS = word.pos_
            target_word_TAG = word.tag_
            spacy_tag = target_word_TAG

    return sentence, target_word, collocations, scores, target_pos, spacy_tag

def get_parameter(collocation, target_word, target_pos):
    ## find word pos from collocation sent
    word_pos = []
    word_pos = pos_tag(word_tokenize(collocation), tagset = 'universal')

    ## find relation(s) for API filtering parameter
    rels=[]
    for word, tag in word_pos:
        match tag:
            case "VERB":
                rel = "V"
            case "NOUN":
                rel = "N"
            case "ADV":
                rel = "A"
            case "ADJ":
                rel = "A"
            case "ADP":
                rel = "prep"
            #handle conj case 'to try and find'
            case "CONJ":
                rel = "C"
            case _:
                rel = tag
        rels.append(rel)
    
    frequency = Counter(rels)
    relation = ""
    if (frequency["A"]) == 2:
        if (frequency["V"]):
            relation = "V:mod:Adv"
        else:
            relation = "Adj:mod:Adv"
    elif (frequency["V"]) == 2:
        relation = "V:sc:Vinf"
    elif (frequency["N"]) == 2:
        if (frequency["prep"]):
            relation = "N:prep:N"
        elif (frequency["V"]):
            relation = "V:obj1+2:N"
        elif (frequency["A"]):
            relation = "N:mod:Adj"
        else:
            relation = "N:nn:N"
    elif (frequency["V"]) and (frequency["N"]):
        if (frequency["prep"]):
            relation = "V:prep:N"
        elif (frequency["C"]):
            relation = "V:sc:Vinf"
        else: #handling VobjN dan VscN
            if target_pos == "VERB":
                #target_pos = V VscN
                relation = "V:obj:N"
            elif target_pos == "NOUN":
                #target_pos = N VobjN
                relation = "V:subj:N"
            ## 6 line below unuses because basisword and basispos searching based on relation
            # if basis_pos == "V":
            #     #basis = V VscN
            #     relation = "V:subj:N"
            # elif basis_pos == "N":
            #     #basis = N VobjN
            #     relation = "V:obj:N"
    elif (frequency["N"]) and (frequency["A"]):
            relation = "N:mod:Adj"
    elif (frequency["V"]) and (frequency["A"]):
            relation = "V:mod:Adv"
    else:
        relation = ""

    ## find basisword for searching
    candidates = [] 
    tag = ""
    basis_word = ""
    pronouns = ['i', 'me', 'we', 'us', 'you', 'your', 'yourself', 'yourselves', 'he', 'his', 'him', 'himself', 'she', 'her', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']
    tobe = ['is','am','are','was','were','has','have','had']
    
    ## basisword selection base on relation
    if relation != "":
        candidates = []
        for word, tag in word_pos:
            if tag == "VERB" or tag == "NOUN" or tag == "ADJ" or tag == "ADV":
                candidates.append((word,tag))

        if relation == "V:sc:Vinf":
            for word,tag in candidates:
                if tag == "VERB" and word != target_word and word not in tobe:
                    basis_word = word
        elif relation == "N:prep:N" or relation == "N:nn:N":
            for word,tag in candidates:
                if tag == "NOUN" and word != target_word:
                    basis_word = word
        elif relation == "Adj:mod:Adv":
            if target_pos == "ADJ":
                basis_word = [word for word, tag in candidates if "ADV" in tag]
                if basis_word == []:
                    basis_word = [word for word, tag in candidates if "ADJ" in tag][0]
                else:
                    basis_word = basis_word[0]
            elif target_pos == "ADV":
                basis_word = [word for word, tag in candidates if "ADJ" in tag]
                if basis_word == []:
                    basis_word = [word for word, tag in candidates if "ADV" in tag][0]
                else:
                    basis_word = basis_word[0]
        elif relation == "V:mod:Adv":
            if target_pos == "VERB":
                basis_word = [word for word, tag in candidates if "ADV" in tag]
                if basis_word == []:
                    basis_word = [word for word, tag in candidates if "ADJ" in tag][0]
                else:
                    basis_word = basis_word[0]
            elif target_pos == "ADV" or target_pos == "ADJ":
                for word,tag in candidates:
                    if tag == "VERB":
                        basis_word = word
        elif relation == "N:mod:Adj":
            if target_pos == "NOUN":
                basis_word = [word for word, tag in candidates if "ADJ" in tag]
                if basis_word == []:
                    basis_word = [word for word, tag in candidates if "ADV" in tag][0]
                else:
                    basis_word = basis_word[0]
            elif target_pos == "ADJ" or target_pos == "ADV":
                for word,tag in candidates:
                    if tag == "NOUN":
                        basis_word = word
        elif relation == "V:prep:N" or relation == "V:subj:N" or relation == "V:obj:N":
                if target_pos == "VERB":
                    for word,tag in candidates:
                        if tag == "NOUN":
                            basis_word = word
                elif target_pos == "NOUN":
                    for word,tag in candidates:
                        if tag == "VERB":
                            basis_word = word
        # need to select noun that is not a person pronoun
        elif relation == "V:obj1+2:N":
                if target_pos == "VERB":
                    for word,tag in candidates:
                        if tag == "NOUN":
                            basis_word = word
                elif target_pos == "NOUN":
                    for word,tag in candidates:
                        if tag == "VERB":
                            basis_word = word
    else:
        # take verb and noun first as basis word candidate if any
        candidates = []
        for word, tag in word_pos:
            if tag == "VERB" or tag == "NOUN":
                candidates.append((word,tag))
            
        if len(candidates) == 2 and target_word != "":
            ## remove target_word from candidates 
            ## fail to check any target word?
            for candidate in candidates:
                # delete empty candidate
                if candidate == [] or candidate == ():
                        candidates.remove(candidate)
                # take candidate that is not target word
                for word, tag in candidate:
                    if word == target_word:
                        candidates.remove(candidate)
                    elif candidates != [()]:
                        basis_word = candidate[0][0]
                    # else:
                    #     basis_word = ""

        else:
            ## select basis word with pronoun in collocation 
            candidates = []
            candidates = word_pos
            for (word, tag) in candidates:
                # for word, tag in candidate:
                    if word == any(pronouns) or word == target_word:
                        # delete pronoun from candidates
                        candidates.remove((word, tag))
            if len(candidates) == 1:
                basis_word = candidates[0][0]
            else:
                ## select basis word base on position on collocation
                candidates = []
                candidates = word_pos
                # for candidate in candidates:
                for (word, tag) in candidates:
                    if word == target_word:
                        if candidates.index((word, tag)) == 1:
                            basis_word = candidates[3][0]
                        elif candidates.index((word, tag)) == 3:
                            basis_word = candidates[1][0]
                        else:
                            basis_word = candidates[2][0]

    ## find basis pos
    ## basis_pos for API call, pos for lemmatize basis word
    basis_pos = []
    pos = ""
    for word, tag in word_pos:
        if word == basis_word:
            # basis_pos = tag
            match tag:
                case "VERB":
                    basis_pos = "V"
                    pos = "v"
                case "NOUN":
                    basis_pos = "N"
                    pos = "n"
                case "ADV":
                    basis_pos = "Adv"
                    pos = "r"
                case "ADJ":
                    basis_pos = "Adj"
                    pos = "a"
                case _:
                    basis_pos = ""
                    pos = ""

    # lemmatize basisword
    # can be commented if basisword is not needed in a lemmatized form
    wordnet_lemmatizer = WordNetLemmatizer()
    if pos != "":
        basis_word = wordnet_lemmatizer.lemmatize(basis_word, pos)
    else:
        basis_word = wordnet_lemmatizer.lemmatize(basis_word)


    return word_pos, rels, relation, candidates, basis_word, basis_pos

def generate_distractor(collocations, target_word, target_pos, correct_answer, spacy_tag):
    collocated_word_list = []
    word_pos_list = []
    rels_list = []
    relation_list = []
    candidates_list = []
    basis_word_list = []
    basis_pos_list = []
    datas_dict = {}
    sorted_datas_dict = {}
    filtered_datas_dict = {}
    related_word_dict = {}
    filtered_datas_list = []
    distractors = []
    api_request_sent = {}
    api_request_list = []
    trial = ""

    sp = spacy.load('en_core_web_sm')

    # processing each collocation
    for collocation in collocations:
        #getting parameter from collocation for API request
        (word_pos, rels, relation, candidates, basis_word, basis_pos) = get_parameter(collocation, target_word, target_pos)

        for word,tag in word_pos:
            if word not in collocated_word_list:
                collocated_word_list.append(word)

        if basis_word != "":
        # line below in a thinking progress
        # if basis_word != "" and (api_request_list == [] or api_request_sent not in api_request_list):
            (datas, api_request_sent, api_request_list, trial) = API_request_controller(basis_word, relation, basis_pos, trial, api_request_list)

            word_pos_list.append(word_pos)
            rels_list.append(rels)
            relation_list.append(relation)
            candidates_list.append(candidates)
            # if basis_word not in basis_word_list:
            basis_word_list.append(basis_word)
            basis_pos_list.append(basis_pos)

            
            # if datas != {}:
            for key, value in datas.items():
            # if data not in datas_dict:
                datas_dict.update({key:value})

            sorted_datas_dict = dict(reversed(sorted(datas_dict.items(), key = lambda item: item[1])))

    ## getting distractor candidate from sorted_distractor_dict
    filtered_datas_dict = {}
    for key, value in sorted_datas_dict.items():
        str_key = str(key)
        key_pos = pos_tag(word_tokenize(str_key), tagset = 'universal')
        ## exclude for distractor
        exc  = ["i","to","too", "so","of", "'", "{", "}"]
        # excl = ["i", "'", "{", "}"]
        # exclu  = ['i','to','too', 'so', "'", "{", "}"]
        # collocation_tag = ['NOUN', 'VERB', 'ADV', 'ADJ']
        pronouns = ['i', 'me', 'we', 'us', 'you', 'your', 'yourself', 'yourselves', 'he', 'his', 'him', 'himself', 'she', 'her', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']

        ## THIS SECTION BELOW CAN BE CHOSEN AS DISTRACTOR DIFFICULTY LEVEL
        ## easy distractor with POS != target_pos
        ## medium distractor with POS = all collocation_tag
        ## hard distractor with POS = target_pos

        ## SECTION 1 EASY
        ## get distractor with pos != target_pos
        # for word, tag in key_pos:
        #     if word not in basis_word_list and word not in correct_answer and word not in target_word and word not in exc and word not in filtered_datas_dict and word not in candidates and tag not in target_pos:
        #         filtered_datas_dict[word] = value

        ## SECTION 2 MEDIUM
        ## get distractor with all collocation_tag pos
        # for word, tag in key_pos:
        #     if word not in basis_word_list and word not in correct_answer and word not in target_word and word not in exc and word not in filtered_datas_dict and word not in candidates and tag in collocation_tag:
        #         filtered_datas_dict[word] = value

        ## SECTION 3 HARD
        ## get distractor with pos = target pos
        ## resulting some synonym distractors
        for word, tag in key_pos: # some of this "if criteria" should be move down to cleaned key criteria
            if word not in basis_word_list and word not in correct_answer and word not in target_word and word not in exc and word not in filtered_datas_dict and word not in candidates and tag in target_pos:
                cleaned_key = ' '.join([word for word,tag in key_pos if word not in basis_word_list and word not in collocated_word_list and word not in exc and word not in pronouns ])
                filtered_datas_dict[cleaned_key] = value

                related_word = ' '.join([word for word,tag in key_pos if word not in cleaned_key and word not in exc and word not in pronouns ])
                related_word_dict[value] = [cleaned_key,related_word]

        ### old backup
        # for word, tag in key_pos:
        #     if word not in basis_word_list and word not in correct_answer and word not in target_word and word not in exc and tag in target_pos:
        #         filtered_datas_dict[word] = value

    for key, value in filtered_datas_dict.items():
        filtered_datas_list.append(key)

    ## optimal exclusion below
    inflected_datas_list = []
    collocation_pos = ['VERB', 'NOUN', 'ADV', 'ADJ']
    # collocation_pos_3 = ['VERB', 'ADV', 'ADJ']
    # collocation_pos_2 = ['VERB', 'ADV']
    collocation_pos_1 = ['VERB']
    # spacy_tag_list_3 = ["VBG", "VBD", "VBN"]
    spacy_tag_list_4 = ["VBG", "VBD", "VBN", "NNS"]
    # spacy_tag_list_5 = ["VBG", "VBD", "VBN", "NN", "NNS"]
    # sp = spacy.load('en_core_web_sm')
    # for dat in datas:
    for dat in filtered_datas_list:
        sp_dat = sp(dat)
        if len(sp_dat) == 1:
            for sd in sp_dat:
                if sd.pos_ in collocation_pos:
                    if spacy_tag in spacy_tag_list_4:
                        sdi = sd._.inflect(spacy_tag)
                    else:
                        sdi = sd
                    dat = dat.replace(str(sd),str(sdi))
        else:
            for sd in sp_dat:
                if sd.pos_ in collocation_pos_1:
                    if spacy_tag in spacy_tag_list_4:
                        sdi = sp_dat[0]._.inflect(spacy_tag)
                    else:
                        sdi = sp_dat[0]
                    dat = dat.replace(str(sd),str(sdi))

        inflected_datas_list.append(dat)

    datas_list_length = len(inflected_datas_list)
    if datas_list_length >= 3:
        for d in range(3):
            distractors.append(inflected_datas_list[d])
    else:
        for distractor in inflected_datas_list:
            distractors.append(distractor)
        else:
            for e in range(3 - datas_list_length):
                distractors.append("err")

    return collocated_word_list, word_pos_list, rels_list, relation_list, candidates_list, basis_word_list, basis_pos_list, api_request_list, sorted_datas_dict, filtered_datas_dict, related_word_dict, distractors

def API_request_controller(basis_word, relation, basis_pos, trial, api_request_list):
    ## used if ask_API result is empty array
    empty_basis_pos = ""
    empty_relation = ""
    datas = {}
    api_request_sent = {}
    
    # request with params
    if trial == "":
        (datas, api_request_sent, api_request_list, trial) = ask_API(basis_word, relation, basis_pos, trial, api_request_list)
    # request without basis_pos
    elif trial == "empty_basis_pos" and basis_pos != "":
        (datas, api_request_sent, api_request_list, trial) = ask_API(basis_word, relation, empty_basis_pos, trial, api_request_list)
    # request without relation
    elif trial == "empty_relation" and relation != "":
        (datas, api_request_sent, api_request_list, trial)  = ask_API(basis_word, empty_relation, basis_pos, trial, api_request_list)
    # # request without relation and basis_pos
    # # this one is not recommended because collocation data will have no relation to the sentence context except the basis_word itself
    # elif trial == "empty_params":
    #     (datas, api_request_sent, api_request_list, trial)  = ask_API(basis_word, empty_relation, empty_basis_pos, trial, api_request_list)

    return datas, api_request_sent, api_request_list, trial

def ask_API(basis_word, relation, basis_pos, trial,api_request_list):
# def ask_API(basis_word, relation, basis_pos):
    api_request_sent = {}
    # # # MAIN API 

    # # # v1 
    # # url = "https://linguatools-english-collocations.p.rapidapi.com/bolls/"

    # v2
    # url = "https://linguatools-english-collocations.p.rapidapi.com/bolls/v2"
    url = os.environ.get("API-url")

    querystring = {"lang":"en","query":basis_word,"max_results":"30","relation":relation,"pos":basis_pos}

    headers = {
        "X-RapidAPI-Key": os.environ.get("X-RapidAPI-Key"),
        "X-RapidAPI-Host": os.environ.get("X-RapidAPI-Host")
    }
    
    response = requests.get(url, headers=headers, params=querystring)

    api_request_sent = querystring
    api_request_list.append(api_request_sent)

    ## try catch
    # try:
    #     response = requests.get(url, headers=headers, params=querystring)

    # except requests.exceptions.ConnectionError as e:
    #     flash(f"API connection error. Can't connect to API provider.")
    #     return redirect(url_for("services"))
    #     # pass

    # data = json.loads(response.text)
    data = response.json()

    # # skip API
    # response = True
    
    datas = {}
    if response:
    # if response == 200:
        # result = "Ok."
        # datas = {}  
        # response header 429
        # {"message": "You have exceeded the rate limit per hour for your plan, BASIC, by the API provider"}
        # if data(['message']): 
            # flash(f"You have exceeded the rate limit per hour for your plan, BASIC, by the API provider")
            #return redirect(url_for("services"))

        if data != []:
        # if data != [] and trial == "":
            for item in data:
                datas.update({item['collocation']:item['significance']})
                # # datas for skip API
                # datas = {'to let decide': 7008, 'to find sit': 4483, 'to find stand': 3409, 'to find wonder': 2328, 'to allow to decide': 2313, 'to find look': 1662, 'to find empty': 1629, 'to find wait': 1621, 'to find face': 1605, 'to make decide': 1517, 'to help decide': 1416, 'to find walk': 1210, 'to find stare': 1140, 'to find close': 1113, 'to have decide': 552, 'to force to decide': 285, 'to compel to decide': 179, 'to call upon to decide': 155, 'to suppose decide': 141, 'to suppose to decide': 106}
        elif data == [] and trial == "":
            trial = "empty_basis_pos"
            (datas, api_request_sent, api_request_list, trial) = API_request_controller(basis_word, relation, basis_pos, trial, api_request_list)

        elif data == [] and trial == "empty_basis_pos":
            trial = "empty_relation"
            (datas, api_request_sent, api_request_list, trial) = API_request_controller(basis_word, relation, basis_pos, trial, api_request_list)
        # elif data == [] and trial == "empty_relation":
        #     trial = "empty_params"
        #     (datas, api_request_sent, api_request_list, trial) = API_request_controller(basis_word, relation, basis_pos, trial, api_request_list)
    else:
        # datas = {} #[{'Not found'}]
        flash(f"API connection error. No response from API provider.")
        return redirect(url_for("services"))
    
    
    return datas, api_request_sent, api_request_list, trial

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/services', methods=['POST', 'GET'])
def services():
    if request.method == "POST":
        passage = request.form["passage"]
        stem = request.form["stem"]
        # pmi_rank = int(request.form["pmi_rank"])
        correct_answer = request.form["correct_answer"]
        correct_mark = request.form.get('correct_mark')
        ans_mark = request.form.get('ans_mark')
        pmi_rank = 1

        if correct_mark:
            mark = "(correct)"
            ans = " "
        elif ans_mark:
            mark = " "
            ans = "ANS: A"
        else:
            mark = " "
            ans = " "
            
        # sentence = request.args.get('sentence')
        # stem = request.args.get('stem')
        # correct_answer = request.args.get('correct_answer')
        # sp = spacy.load('en_core_web_sm')

        correct_ans = get_one_word(correct_answer)

        (sentence, target_word, collocations, scores, target_pos, spacy_tag) = get_collocations(passage, stem, pmi_rank)
        
        (collocated_word_list, word_pos_list, rels_list, relation_list, candidates_list, basis_word_list, basis_pos_list, api_request_list, sorted_datas_dict, filtered_datas_dict, related_word_dict, distractors) = generate_distractor(collocations, target_word, target_pos, correct_ans, spacy_tag)

        return render_template('services.html', 
        mark = mark,
        ans = ans,
        passage = passage,
        sentence = sentence,
        stem = stem, 
        correct_answer = correct_answer, 
        target_word = target_word,
        target_pos = target_pos,
        spacy_tag = spacy_tag,
        collocations = collocations,
        pmi_rank = pmi_rank,
        collocation = scores[0],
        scores = scores,
        collocated_word_list = collocated_word_list,
        word_pos = word_pos_list,
        tag = rels_list,
        relation = relation_list,
        candidates = candidates_list,
        basis_word = basis_word_list,
        basis_pos = basis_pos_list,
        api_request_list = api_request_list,
        sorted_datas_dict = sorted_datas_dict,
        filtered_datas_dict = filtered_datas_dict,
        related_word_dict = related_word_dict,
        distractors = distractors
        )

    else:
        mark = " "
        distractors = []
        for i in range(3):
            distractors.append(" ")
        return render_template('services.html', distractors = distractors, mark = mark)
    
@app.route('/one', methods=['POST', 'GET'])
def process_text():
    if request.method == "POST":
        text = request.form['text']
        preprocessed_text = preprocess_text(text)
        # flash(f"Succeed!", "info")
        return render_template('preprocessed_text.html', text=text, preprocessed_text=preprocessed_text)
        # return render_template('result.html')
    else:
        return render_template('preprocess_text.html')

@app.route("/four")
def four():
    return render_template("four.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["unm"]
        session["user"] = user
        flash(f"Login successful, {user}!", "info")
        return redirect(url_for("services"))
    else:
        # cek session
        if "user" in session:
            user = session["user"]
            flash(f"You've already login, {user}!", "info")
            return redirect(url_for("services"))
        return render_template("login.html")

@app.route("/logout")
def logout():
    if "user" in session:
        user = session["user"]
        flash(f"Thank you, {user}! You have been logged out.", "info")
    session.pop("user", None)
    # session.pop("email", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    # app.app_context().push()
    # app.run()
    app.run(debug=True)