from UnigramModel import UnigramModel
from BigramModel import BigramModel
import os
import random
import math
from sklearn.metrics import f1_score

class MixedModel:
    def __init__(self, directory_path, all_data, validation_split=0.1):

        #random.seed(10)

        self.directory_path = directory_path
        self.validation_split = validation_split
        self.train_data, self.validation_data = self.split_data(directory_path)

        # Training session
        self.unigram_model = UnigramModel(directory_path, self.train_data)
        self.bigram_model = BigramModel(directory_path, self.train_data)

        self.optimal_lambda = 0
        self.find_optimal_lambda()

        # Train it with the whole data 
        self.unigram_model = UnigramModel(directory_path, all_data)
        self.bigram_model = BigramModel(directory_path, all_data)

    def split_data(self, directory_path, validation_split=0.1):
        train_data = {}
        validation_data = {}

        for genre in os.listdir(directory_path):
            genre_files = os.listdir(os.path.join(directory_path, genre))
            #random.shuffle(genre_files)  # Shuffle files to randomize

            # Determine the number of files for validation set
            split_index = int(len(genre_files) * validation_split)
            validation_files = genre_files[:split_index]
            train_files = genre_files[split_index:]

            # Save the training files for the genre
            train_data[genre] = train_files

            # Save the validation files for the genre
            validation_data[genre] = validation_files

        return train_data, validation_data

    def find_optimal_lambda(self): 
        best_lambda = 0.0
        best_f1_score = 0.0
        predicted_labels = []
        gold_labels = []
        for lambda_value in [(i+1) * 0.1 for i in range(9)]:
            self.optimal_lambda = lambda_value
            for genre in self.validation_data:
                for file in self.validation_data[genre]:
                    with open(self.directory_path + '/' + genre + '/' + file, 'r') as input_text:
                        predicted_genre = self.predict_genre(input_text.read())
                        predicted_labels.append(predicted_genre)
                        gold_labels.append(genre)

            f1 = f1_score(gold_labels, predicted_labels, average='weighted')
            if f1 > best_f1_score:
                best_f1_score = f1
                best_lambda = lambda_value
        self.optimal_lambda = best_lambda


    def calculate_combined_probability(self, input_text):
        combined_probabilities = {}
        
        genre_probs_unigram = self.unigram_model.calculate_probability(input_text)
        genre_probs_bigram = self.bigram_model.calculate_probability(input_text)
        
        for genre_name in genre_probs_unigram:
            combined_prob = self.optimal_lambda * genre_probs_unigram[genre_name] - (1 - self.optimal_lambda) * genre_probs_bigram[genre_name]
            combined_probabilities[genre_name] = combined_prob
        
        return combined_probabilities
    

    def predict_genre(self, input_text):
        genre_probs_result = self.calculate_combined_probability(input_text)
        genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
        predicted_genre = list(genre_probs_result.keys())[0]
        return predicted_genre


def main():
    directory_path = 'TM_CA1_Lyrics/'

    all_data = {}
    for genre in os.listdir(directory_path):
        genre_files = os.listdir(os.path.join(directory_path, genre))
        all_data[genre] = genre_files

    mixed_model = MixedModel(directory_path, all_data)

    input_text = """
    You used to call me on my cell phone
    Late night when you need my love
    Call me on my cell phone
    """

    genre_probs_result = mixed_model.calculate_combined_probability(input_text)

    genre_probs_result = dict(sorted(genre_probs_result.items(), key=lambda item: item[1], reverse=False))
    for genre_name, result in genre_probs_result.items():
        print(f'{genre_name}: {result}')

if __name__ == '__main__':
    main()