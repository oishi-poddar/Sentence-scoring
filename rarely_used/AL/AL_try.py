from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from alipy import ToolBox
from sklearn.metrics import average_precision_score
from alipy.data_manipulate import StandardScale
import pandas as pd
import copy
from sklearn import metrics
import numpy as np
import csv
import sklearn
from numpy import genfromtxt
df = pd.read_csv("features.csv")
# df=pd.DataFrame(StandardScale(X=df_not))
# scaler=MinMaxScaler()
# df=pd.DataFrame(scaler.fit_transform(df_not.astype(float)))
# df.columns=df_not.columns
# df_index=df_not.index
# Create an empty list
X = []

# Iterate over each row
for rows in df.itertuples():
    # Create list for the current row
    my_list = [rows.position, rows.special_char, rows.upper_case, rows.numeral, rows.aggregate, rows.length, rows.freq_score,
               rows.proper_noun, rows.noun, rows.verb, rows.named_entity, rows.cue_phrases, rows.centrality]
    # append the list to the final list
    X.append(my_list)

# Print the list

all=np.asarray(X, dtype=float)
X=all[0:1020]
df = pd.read_csv("sentences_data.csv")[0:1020]
NANindex=df['Oracle label'].index[df['Oracle label'].apply(np.isnan)]
X=np.delete(X, NANindex,0)
df=df.dropna(subset=['Oracle label'])
print(df)
print(len(X))
y=df[['Oracle label']].to_numpy().ravel().astype(int)
print(y)
print("siz",y.size)
# y = np.asarray([1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.1, split_count=3)

# Use the default Logistic Regression classifier
# model = sklearn.svm.SVC(kernel='sigmoid', probability=True)
# model=MLPClassifier(hidden_layer_sizes=(80,80),activation='logistic',solver='adam',max_iter=3000, alpha=0.01)
model=alibox.get_default_model()

# model = RandomForestClassifier()
# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', '100')


def main_loop(alibox, strategy, round):
    # print("str", strategy)
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    # X[train_idx,:] = StandardScaler.fit_transform(X[train_idx,:])
    # Set initial performance point
    model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                              y_pred=pred,
                                              performance_metric='accuracy_score')

    saver.set_initial_point(accuracy)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing any sklearn models with proba_predict method are ok
        select_ind = strategy.select(label_ind, unlab_ind, model=model, batch_size=10)
        print(select_ind)
        # print("selec", select_ind)
        # or pass your proba predict result
        # prob_pred = model.predict_proba(x[unlab_ind])
        # select_ind = strategy.select_by_prediction_mat(unlabel_index=unlab_ind, predict=prob_pred, batch_size=1)

        label_ind.update(select_ind)  #Update self with the union of itself and others.
        unlab_ind.difference_update(select_ind) # removes the index(s)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    return saver
    # unc_result.append(copy.deepcopy(saver))



unc_result = []
qbc_result = []
eer_result = []
quire_result = []
density_result = []
bmdr_result = []
spal_result = []
lal_result = []
rnd_result = []

_I_have_installed_the_cvxpy = False

for round in range(3):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)

    # Use pre-defined strategy
    unc = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty")
    qbc = alibox.get_query_strategy(strategy_name="QueryInstanceQBC")
    # eer = alibox.get_query_strategy(strategy_name="QueryExpectedErrorReduction")
    rnd = alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
    # quire = alibox.get_query_strategy(strategy_name="QueryInstanceQUIRE", train_idx=train_idx)
    # density = alibox.get_query_strategy(strategy_name="QueryInstanceGraphDensity", train_idx=train_idx)
    # lal = alibox.get_query_strategy(strategy_name="QueryInstanceLAL", cls_est=10, train_slt=False)
    # lal.download_data()
    # lal.train_selector_from_file(reg_est=30, reg_depth=5)
    unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
    qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))
    # eer_result.append(copy.deepcopy(main_loop(alibox, eer, round)))
    rnd_result.append(copy.deepcopy(main_loop(alibox, rnd, round)))
    # quire_result.append(copy.deepcopy(main_loop(alibox, quire, round)))
    # density_result.append(copy.deepcopy(main_loop(alibox, density, round)))
    # lal_result.append(copy.deepcopy(main_loop(alibox, lal, round)))
    # # #
    # if _I_have_installed_the_cvxpy:
    #     bmdr = alibox.get_query_strategy(strategy_name="QueryInstanceBMDR", kernel='rbf')
    #     spal = alibox.get_query_strategy(strategy_name="QueryInstanceSPAL", kernel='rbf')
    #
    #     bmdr_result.append(copy.deepcopy(main_loop(alibox, bmdr, round)))
    #     spal_result.append(copy.deepcopy(main_loop(alibox, spal, round)))

analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=qbc_result)
analyser.add_method(method_name='Unc', method_results=unc_result)
# analyser.add_method(method_name='EER', method_results=eer_result)
analyser.add_method(method_name='Random', method_results=rnd_result)
# analyser.add_method(method_name='QUIRE', method_results=quire_result)
# analyser.add_method(method_name='Density', method_results=density_result)
# analyser.add_method(method_name='LAL', method_results=lal_result)
# if _I_have_installed_the_cvxpy:
#     analyser.add_method(method_name='BMDR', method_results=bmdr_result)
#     analyser.add_method(method_name='SPAL', method_results=spal_result)
print(analyser)
analyser.plot_learning_curves(title='Example of alipy', std_area=False)


li=np.arange(0,1090)
# print(li)

# find metrics
predicted = model.predict(X[test_idx, :])
y_test=y[test_idx]
print('MAE:', metrics.mean_absolute_error(y_test, predicted))
print('MSE:', metrics.mean_squared_error(y_test, predicted))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R2 score:', np.sqrt(metrics.r2_score(y_test, predicted)))
print("avg precision recall", average_precision_score(y_test, predicted))
print("precision",sklearn.metrics.precision_score(y_test, predicted, average='binary'))
print("recall",sklearn.metrics.recall_score(y_test, predicted, average='binary'))
print("f1",sklearn.metrics.f1_score(y_test, predicted, average='binary', pos_label=1))
test_list=model.predict(all[li,:])
print(test_list)
csv_input = pd.read_csv('sentences_data.csv')
csv_input['Predicted'] = pd.Series(test_list)
csv_input.to_csv('output.csv', index=False)
