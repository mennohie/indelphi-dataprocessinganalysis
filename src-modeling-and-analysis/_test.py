import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def use_fraction(master_data):
    fractions = master_data['counts']['fraction']
    result = {}
    for i in range(len(fractions)):
        sequence, indel = fractions.index[i]
        pos, indel = indel.split("+")
        if indel.isdigit():
            if int(indel) % 3 != 0:
                # frameshift
                fraction = fractions[i]
                if fraction != 0.0:
                    if sequence in result:
                        result[sequence] += fraction
                    else:
                        result[sequence] = fraction
        else:
            fraction = fractions[i]
            if fraction != 0.0:
                if sequence in result:
                    result[sequence] += fraction
                else:
                    result[sequence] = fraction

        # only plot some datapoints for debugging
        # if i == 80000:
        #     break

    print(result)
    return result

if __name__ == "__main__":
    master_data = pkl.load(open('../pickle-data/inDelphi_counts_and_deletion_features.pkl', 'rb'))
    fraction = use_fraction(master_data)

    truth = np.array(list(fraction.values()))
    predicted = np.random.randint(0, 80, len(truth))  # randomize predicted for now

    plt.plot([0, 80], linestyle='dashed', c='black')
    plt.scatter(truth, predicted)

    # find line of best fit
    a, b = np.polyfit(truth, predicted, 1)

    # add line of best fit to plot
    plt.plot(truth, a * truth + b, c='black')

    plt.show()

    corr = np.corrcoef(truth, predicted)
    print(corr)
