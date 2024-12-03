import nltk
from pywsd.lesk import simple_lesk
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer


class SentenceSimilarity:
    def __init__(self, mesure):
        self.word_order = False
        self.mesure = mesure

    def identifyWordsForComparison(self, sentence):
        #Taking out Noun and Verb for comparison word based
        tokens = nltk.word_tokenize(sentence)
        pos = nltk.pos_tag(tokens)
        pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]
        return pos

    def wordSenseDisambiguation(self, sentence):
        # removing the disambiguity by getting the context

        pos = self.identifyWordsForComparison(sentence)
        sense = []
        for p in pos:
            sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
        return set(sense)

    def getSimilarity(self, arr1, arr2, vector_len):
        #cross multilping all domains
        vector = [0.0] * vector_len
        count = 0
        for i,a1 in enumerate(arr1):
            all_similarityIndex=[]
            for a2 in arr2:
                if a1 and a1.name() and a2 and a2.name():
                    similarity = self.mesure(wn.synset(a1.name()), wn.synset(a2.name()))
                else:
                    similarity = None

                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)


            if not all_similarityIndex:
                vector[i]= 0.0
                continue
            all_similarityIndex = sorted(all_similarityIndex, reverse = True)
            vector[i]=all_similarityIndex[0]
            if vector[i] >= 0.804:
                count +=1
        return vector, count

    def shortestPathDistance(self, sense1, sense2):
        #getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(sense1, sense2, grt_Sense)
            v2, c2 = self.getSimilarity(sense2, sense1, grt_Sense)
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(sense2, sense1, grt_Sense)
            v2, c2 = self.getSimilarity(sense1, sense2, grt_Sense)
        return np.array(v1),np.array(v2),c1,c2

    def main(self, sentence1, sentence2):
        sense1 = self.wordSenseDisambiguation(sentence1)
        sense2 = self.wordSenseDisambiguation(sentence2)
        v1,v2,c1,c2 = self.shortestPathDistance(sense1,sense2)
        dot = np.dot(v1,v2)
        tow = (c1+c2)
        if tow == 0:
            tow = len(sense1) / 2
        if tow != 0:
            final_similarity = (dot/tow)
            return final_similarity
        else:
            return 0


# boucle sur tout une liste de différent texte (devrai etre les éléments de chaque cluster)
def calculate_cluster_similarity(sentences, mesure="wup"):
    n = len(sentences)
    total_similarity = 0
    pair_count = 0
    fmesure = None
    if mesure == "wup":
        fmesure = wn.wup_similarity
    elif mesure == "path":
        fmesure = wn.path_similarity
    else:
        fmesure = wn.wup_similarity

    obj = SentenceSimilarity(fmesure)
    for i in range(0, n, 2):
        j = i-1
        if i+1 == n:
            j = 0
        similarity = obj.main(sentences[i], sentences[j])
        total_similarity += similarity
        pair_count += 1
        if pair_count % 20 == 0:
            print(total_similarity)


    if pair_count == 0:
        return 0

    overall_similarity = total_similarity / pair_count
    return overall_similarity

# retoune les n mots plus important de la phrase (utilise tf-idf)
def importantwords(sentence, n):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    filtered_words = [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]

    filtered_sentence = " ".join(filtered_words)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered_sentence])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    word_tfidf_dict = dict(zip(feature_names, tfidf_scores))

    sorted_words = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)

    top_words = [word for word, score in sorted_words[:n]]

    return " ".join(top_words)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import re

def clean_text(text):
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Mettre en minuscules
    text = text.lower()

    # Supprimer les caractères numériques
    text = re.sub(r'\d+', '', text)

    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Supprimer les balises HTML (si applicable)
    text = re.sub(r'<.*?>', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Supprimer les mots très courts (par exemple, moins de 3 caractères)
    words = [word for word in words if len(word) > 2]

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Supprimer les mots vides (pour l'anglais dans cet exemple)
    stop_words = set(stopwords.words('english'))

    #custom_stop_words = ['got', 'know', 'yeah', 'one', 'get', 'dont', 'come', 'aint', 'day', 'life','night', 'away', 'like', 'time', 'wa', 'say', 'need', 'love', 'youre', 'make','baby', 'want', 'little', 'let', 'cause', 'thats', 'long', 'eye', 'right','shes', 'way', 'world','tell', 'look', 'gon', 'hey', 'tell', 'thing', 'think']
    # Ajouter des mots personnalisés à la liste des mots vides
    #stop_words.update(custom_stop_words)

    words = [word for word in words if word.lower() not in stop_words]

    # Rejoindre les mots en une chaîne de texte
    cleaned_text = ' '.join(words)

    return cleaned_text

# basique loop sur un cluster appliquant le calcule (voir notebook pour utilsation dataframe)
def loopover(cluster):
    used_metric = "wup"
    cluster = [clean_text(sentence) for sentence in cluster]
    cluster = [importantwords(sentence, 20) for sentence in cluster]
    overall_similarity = calculate_cluster_similarity(cluster, used_metric)
    print(overall_similarity)
