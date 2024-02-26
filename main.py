from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from UnigramModel import UnigramModel
from BigramModel import BigramModel
from MixedModel import MixedModel
import os
 

def evaluate_model(model, test_file):

    test_data = pd.read_csv(test_file, sep='\t')
    gold_labels = test_data['Genre'].tolist()

    # Predict labels
    predicted_labels = []
    for text in test_data['Text']:
        predicted_label = model.predict_genre(text) 
        predicted_labels.append(predicted_label)
    

    # Calculate F1-score
    f1 = f1_score(gold_labels, predicted_labels, average='weighted')

    return f1

def permutation_test(model1_scores, other_model_scores, num_permutations=1000):
    # Calculate observed difference in scores
    obs_diff = np.mean(model1_scores) - np.mean(other_model_scores)

    # Combine scores
    combined_scores = np.concatenate([model1_scores, other_model_scores])

    # Initialize array to store permutation differences
    perm_diffs = np.zeros(num_permutations)

    # Permutation test
    for i in range(num_permutations):
        np.random.shuffle(combined_scores)
        perm_model1_scores = combined_scores[:len(model1_scores)]
        perm_other_model_scores = combined_scores[len(model1_scores):]
        perm_diffs[i] = np.mean(perm_model1_scores) - np.mean(perm_other_model_scores)

    # Calculate p-value
    p_value = np.sum(perm_diffs >= obs_diff) / num_permutations

    return obs_diff, p_value

def print_significance_test(model1_name, model2_name, obs_diff, p_value):
    print(f"{model1_name} vs. {model2_name}: Observed difference = {obs_diff:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"Significantly different: {model1_name} model is better than {model2_name} model")
    else:
        print("No significant difference detected")


def main():

    directory_path = 'TM_CA1_Lyrics/'

    all_data = {}
    for genre in os.listdir(directory_path):
        genre_files = os.listdir(os.path.join(directory_path, genre))
        all_data[genre] = genre_files


    test_file = 'test.tsv'
    unigram_model = UnigramModel(directory_path, all_data)
    unigram_f1_score = evaluate_model(unigram_model, test_file)

    bigram_model = BigramModel(directory_path, all_data)
    bigram_f1_score = evaluate_model(bigram_model, test_file)

    mixed_model = MixedModel(directory_path, all_data)
    mixed_f1_score = evaluate_model(mixed_model, test_file)

    # Print f1 results in a table
    results_df = pd.DataFrame({
        'Model': ['Unigram', 'Bigram', 'Mixed'],
        'F1-score': [unigram_f1_score, bigram_f1_score, mixed_f1_score]
    })
    print(results_df)

    # Perform significance tests
    print("\nSignificance Tests:")
    obs_diff_unigram_vs_bigram, p_value_unigram_vs_bigram = permutation_test([unigram_f1_score], [bigram_f1_score])
    obs_diff_unigram_vs_mixed, p_value_unigram_vs_mixed = permutation_test([unigram_f1_score], [mixed_f1_score])
    obs_diff_bigram_vs_mixed, p_value_bigram_vs_mixed = permutation_test([bigram_f1_score], [mixed_f1_score])

    print_significance_test("Unigram", "Bigram", obs_diff_unigram_vs_bigram, p_value_unigram_vs_bigram)
    print_significance_test("Unigram", "Mixed", obs_diff_unigram_vs_mixed, p_value_unigram_vs_mixed)
    print_significance_test("Bigram", "Mixed", obs_diff_bigram_vs_mixed, p_value_bigram_vs_mixed)


if __name__ == '__main__':
    main()