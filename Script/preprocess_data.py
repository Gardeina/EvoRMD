import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your data
data = pd.read_csv('../RNAdata/11_NEG_data.csv')

# Replace T -> U
data['Sequence_41'] = data['Sequence_41'].apply(lambda x: x.replace('T', 'U'))

# Encode labels
label_encoder = LabelEncoder()
data['modType'] = label_encoder.fit_transform(data['modType'])

# Show label mapping
print("Label mapping:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{index}: {label}")

# Save processed data
data.to_pickle('../RNAdata/11_NEG_preprocessed_data.pkl')
with open('../RNAdata/11_NEG_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Data processed and saved to pickle.")
