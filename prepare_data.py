import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print "Working..."

# Read in data files
people_data_df = pd.read_csv("data/people.csv", parse_dates=['date'], dtype={'people_id': np.str, 'char_38': np.int32})
act_train_data_df = pd.read_csv("data/act_train.csv", parse_dates=['date'], dtype={'people_id': np.str, 'activity_id': np.str})
act_test_data_df = pd.read_csv("data/act_test.csv", parse_dates=['date'], dtype={'people_id': np.str, 'activity_id': np.str})

# Extract the y_train values
y_train = pd.DataFrame(act_train_data_df['outcome'])
y_train.columns = ['outcome']
act_train_data_df.drop(['outcome'], axis=1, inplace=True)

# Extract the y_test activity_id values
y_test_activity_ids = pd.DataFrame(act_test_data_df['activity_id'])
y_test_activity_ids.columns = ['activity_id']

# Fill in missing values with new missing category
act_train_data_df.fillna(value='missing', inplace=True)
act_test_data_df.fillna(value='missing', inplace=True)

# Combine train & test data for uniform process, save train length to index apart later
train_length = act_train_data_df.shape[0]
all_data_df = pd.concat((act_train_data_df, act_test_data_df), axis=0, ignore_index=True)

# Merge people data into the activity dataframe
all_data_with_ppl_char_df = all_data_df.merge(people_data_df, how='left', on='people_id')

# Create the days_since_start feature - the number of days between person account creation and activity date
all_data_with_ppl_char_df['days_since_start'] = (all_data_with_ppl_char_df['date_x'] - all_data_with_ppl_char_df['date_y']).apply(lambda x: int(x.days))

# Create this_ppl_act_count feature -  The total number of activities of this category for each person
all_data_with_ppl_char_df['this_ppl_act_count'] = all_data_with_ppl_char_df.groupby(['people_id', 'activity_category'])['activity_category'].transform('count')

# Label encode all categorical features
all_data_with_ppl_char_df.drop(['people_id', 'activity_id', 'date_x', 'date_y'], axis=1, inplace=True)
for a_feature in all_data_with_ppl_char_df.columns.values:
    if a_feature not in ['char_38', 'days_since_start', 'this_ppl_act_count']:
        all_data_with_ppl_char_df[a_feature] = LabelEncoder().fit_transform(all_data_with_ppl_char_df[a_feature])


# Split the train and test data back out
X_train = all_data_with_ppl_char_df.iloc[:train_length,:]
X_test = all_data_with_ppl_char_df.iloc[train_length:,:].reset_index(drop=True)

# Print verification shapes/samples
print all_data_with_ppl_char_df.shape
print X_train.shape
print X_test.shape
print X_train.head(5)
print X_test.head(5)

# Save data to disk
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test_activity_ids.to_csv("y_test_activity_ids.csv", index=False)

print "Done."