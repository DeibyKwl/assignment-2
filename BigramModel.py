import os
import nltk
from nltk.tokenize import word_tokenize
import string
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class BigramModel:
    def __init__(self, directory_path, all_data):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.genre_tokens = {}
        self.genre_pair_tokens = {}
        self.read_files_in_directory(directory_path, all_data)
        self.genre_probs = self.calculate_genre_probs()

    # Preprocess the text
    def preprocess_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text

    # Read the files in the directory and store them in a dictionary with its frequencies, and store pairs and its frequencies in another dictionary
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

    # Turn frequencies for each word into probabilities
    def freq_to_prob(self, dic_term_frequency, dic_pairs_frequency):
        dic_pair_prob = {}
        #vocabulary_size = len(dic_term_frequency)
        for pair, pair_freq in dic_pairs_frequency.items():
            preceding_term = pair[0]
            total_term_count = sum([freq for (term1, term2), freq in dic_pairs_frequency.items() if term1 == preceding_term])
            total_unique_following_terms = len(set([term2 for (term1, term2) in dic_pairs_frequency.keys() if term1 == preceding_term]))
            
            prob = pair_freq / (total_term_count + total_unique_following_terms)
            dic_pair_prob[pair] = prob
        return dic_pair_prob
    
    # Method to call freq_to_prob for each genre and store it in a dictionary
    def calculate_genre_probs(self):
        genre_probs = {}
        for genre_name in self.genre_tokens:
            genre_probs[genre_name] = self.freq_to_prob(self.genre_tokens[genre_name], self.genre_pair_tokens[genre_name])
        return genre_probs

    # Calculate the probability of a given text to belong to a genre
    def calculate_probability(self, input_text):
        prob_result = {}
        input_text = self.preprocess_text(input_text.lower())
        input_text = input_text.split()
        
        pair_input_texts = [(input_text[i], input_text[i+1]) for i in range(len(input_text)-1)]
        
        for genre_name, genre_prob in self.genre_probs.items():
            prob = 0.0  
            for pair in pair_input_texts:
                if pair in genre_prob:
                    pair_prob = genre_prob.get(pair,0.1)
                    prob += math.log(pair_prob)

            prob_result[genre_name] = prob
        return prob_result
    
    # Predict the genre of a given text
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
    If you like to gambleI tell you, I'm your manYou win some, lose someIt's all the same to me

     """
    genre_probs_result = bigram_model.calculate_probability(example_text)
    genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
    
    for genre_name, result in genre_probs_result.items():
        print(f'{genre_name}: {result}')

if __name__ == '__main__':
    main()
