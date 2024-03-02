import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import math
from collections import Counter

class UnigramModel:
    def __init__(self, directory_path, all_data):
        self.stop_words = set(stopwords.words('english'))
        self.genre_tokens = self.read_files_in_directory(directory_path, all_data)
        self.genre_probs = self.calculate_genre_probs()

    # Preprocess the text
    def preprocess_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    # Read the files in the directory and store them in a dictionary with its frequencies
    def read_files_in_directory(self, directory_path, all_data):
        # key: tokens value: their frequency in all songs belonging to a genre
        genre_tokens = {}
        for genre in os.listdir(directory_path):
            dic_term_frequency = {}
            for file in all_data[genre]:
                with open(directory_path + '/' + genre + '/' + file, 'r') as rfile:
                    for line in rfile:
                        preprocess_line = self.preprocess_text(line.strip().lower())
                        tokens = word_tokenize(preprocess_line)
                        token_counts = Counter(tokens)
                        for token, count in token_counts.items():
                            dic_term_frequency[token] = dic_term_frequency.get(token, 0) + count
            genre_tokens[genre] = dic_term_frequency
        return genre_tokens

    # Turn frequencies for each word into probabilities
    def freq_to_prob(self, dic_term_frequency):
        dic_term_prob = {}
        total_frequency = sum(dic_term_frequency.values())

        for term, frequency in dic_term_frequency.items():
            probability = frequency / total_frequency
            dic_term_prob[term] = probability
        return dic_term_prob
    
    # Method to call freq_to_prob for each genre and store it in a dictionary
    def calculate_genre_probs(self):
        genre_probs = {}
        for genre_name in self.genre_tokens:
            genre_probs[genre_name] = self.freq_to_prob(self.genre_tokens[genre_name])
        return genre_probs

    # Calculate the probability of a given text to belong to a genre
    def calculate_probability(self, input_text):
        prob_result = {}
        input_text = self.preprocess_text(input_text.lower())
        for genre_name, genre_prob in self.genre_probs.items():
            prob = 0.0
            for token in input_text.split():
                token_prob = genre_prob.get(token, 0.001)
                prob += math.log(token_prob)
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

    
    unigram_model = UnigramModel(directory_path, all_data)
    example_text = """
    You used to call me on my cell phone
    Late night when you need my love
    Call me on my cell phone
    """
    genre_probs_result = unigram_model.calculate_probability(example_text)
    genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
    
    for genre_name, result in genre_probs_result.items():
        print(f'{genre_name}: {result}')
    

if __name__ == '__main__':
    main()