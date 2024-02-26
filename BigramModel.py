import os
import nltk
from nltk.tokenize import word_tokenize
import string
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

class BigramModel:
    def __init__(self, directory_path, all_data):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.genre_tokens = {}
        self.genre_pair_tokens = {}
        self.read_files_in_directory(directory_path, all_data)
        self.genre_probs = self.calculate_genre_probs()

    def preprocess_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text

    def read_files_in_directory(self, directory_path, all_data):
        for genre in os.listdir(directory_path):
            dic_term_frequency = {}
            dic_pairs_frequency = {}
            for file in all_data[genre]:
                with open(os.path.join(directory_path, genre, file), 'r') as rfile:
                    previous_token = None
                    for line in rfile:
                        current_line = line.strip().lower()
                        preprocess_line = self.preprocess_text(current_line)
                        tokens = word_tokenize(preprocess_line)
                        for token in tokens:
                            dic_term_frequency[token] = dic_term_frequency.get(token, 0) + 1
                        for i in range(1, len(tokens)):
                            current_token = tokens[i]
                            bigram = (previous_token, current_token)
                            dic_pairs_frequency[bigram] = dic_pairs_frequency.get(bigram, 0) + 1
                            previous_token = current_token
            self.genre_tokens[genre] = dic_term_frequency
            self.genre_pair_tokens[genre] = dic_pairs_frequency

    def freq_to_prob(self, dic_term_frequency, dic_pairs_frequency):
        dic_term_prob = {}
        total_pair_count = sum(dic_pairs_frequency.values())
        total_term_count = sum(dic_term_frequency.values())
        vocabulary_size = len(dic_term_frequency)
        for term in dic_term_frequency:
            pair_freq_with_term = sum([pair_freq for pair, pair_freq in dic_pairs_frequency.items() if term in pair])
            prob = (pair_freq_with_term + 1) / (total_pair_count + total_term_count + vocabulary_size)
            dic_term_prob[term] = prob
        return dic_term_prob
    
    def calculate_genre_probs(self):
        genre_probs = {}
        for genre_name in self.genre_tokens:
            genre_probs[genre_name] = self.freq_to_prob(self.genre_tokens[genre_name], self.genre_pair_tokens[genre_name])
        return genre_probs

    def calculate_probability(self, input_text):
        prob_result = {}
        for genre_name, genre_prob in self.genre_probs.items():
            prob = 0.0
            for token in input_text.split():
                token_prob = genre_prob.get(token.lower(), 0.001)
                prob += math.log(token_prob)
            prob_result[genre_name] = prob
        return prob_result
    
    def predict_genre(self, input_text):
        genre_probs_result = self.calculate_probability(input_text)
        genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
        predicted_genre = list(genre_probs_result.keys())[0]
        return predicted_genre
    

def main():

    nltk.download('punkt')
    nltk.download('stopwords')

    directory_path = 'TM_CA1_Lyrics/'

    all_data = {}
    for genre in os.listdir(directory_path):
        genre_files = os.listdir(os.path.join(directory_path, genre))
        all_data[genre] = genre_files


    bigram_model = BigramModel('TM_CA1_Lyrics/', all_data)
    example_text = """
    A singer in a smokey roomA smell of wine and cheap perfumeFor a smile they can share the nightIt goes on and on and on and on
        """
    genre_probs_result = bigram_model.calculate_probability(example_text)
    genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
    
    for genre_name, result in genre_probs_result.items():
        print(f'{genre_name}: {result}')

if __name__ == '__main__':
    main()
