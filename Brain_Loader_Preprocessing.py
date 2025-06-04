import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class BrainMRIDataLoader:
    def __init__(self, base_path, categories):
        """
        Initializes the data loader with the base path and category names.

        Args:
            base_path (str): Root directory containing category folders.
            categories (list): List of category folder names (e.g., ['Healthy', 'Tumor']).
        """
        self.base_path = base_path
        self.categories = categories
        self.df = None

    def load_data(self):
        """
        Loads image paths and labels from the directory structure into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns ['image_path', 'label'].
        """
        image_paths = []
        labels = []
        for category in self.categories:
            category_path = os.path.join(self.base_path, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    image_paths.append(image_path)
                    labels.append(category)
                
            else:
                print(f"⚠️ Directory not found: {category_path}")
        self.df = pd.DataFrame({'image_path': image_paths, 'label': labels})
        print('------------ Status of the data load ------------')
        print(f'* data loaded successfully')
        print(50*'-','\n')
        return self.df

    def show_summary(self):
        """
        Prints a summary of the loaded data, including a preview and class distribution.
        """
        if self.df is not None:
            print("* Initial DataFrame with image paths and labels:")
            print(self.df.head())
            print(50*'-','\n')
            print("* Initial class distribution:")
            print(self.df['label'].value_counts())
        else:
            print("No data loaded yet. Run load_data() first.")
            

class Brain_preprocessing:
    def __init__(self,df):
        """
        Initializes the preprocessing class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
        """
        self.df = df
        self.preprocessed_df = None
        self.label_encoder = LabelEncoder()
    
    def category_encoder(self):
        """
        Encodes the categorical labels into numerical values.

        Returns:
            pd.DataFrame: DataFrame with encoded labels.
        """
        self.df['category_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        print(50*'-','\n')
        print("* Categorical labels encoded successfully.\n")
        print(self.df.head(3))
        print(50*'-','\n')
        print("* Encoded class distribution:")
        print(self.df['category_encoded'].value_counts())
        print(f"Clases codificadas: {self.label_encoder.classes_} -> {self.label_encoder.transform(self.label_encoder.classes_)}")
        print(50*'-','\n')
        return self.df
    
    def train_val_test_split(self, test_size, random_state):
        """
        Splits the DataFrame into training, validation, and test sets.

        This method first splits the data into a training set and a temporary set (for validation and test)
        according to the `test_size` proportion. Then, it splits the temporary set equally into validation
        and test sets (each will be half of the temporary set, i.e., test_size/2 of the total).

        For example, with test_size=0.2:
            - 80% training
            - 10% validation
            - 10% test

        Args:
            test_size (float): Proportion of the dataset to allocate to validation+test (default 0.2).
            random_state (int): Random seed for reproducibility (default 42).

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
                                             self.df,
                                             test_size=test_size,
                                             random_state=random_state,
                                             stratify=self.df['category_encoded'])
        val_df, test_df = train_test_split(
                                            temp_df, 
                                            test_size=0.5,
                                            random_state=random_state,
                                            stratify=temp_df['category_encoded'])
        print("* Train/Validation/Test split completed successfully.\n")
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        print(50*'-','\n')
        
        return train_df, val_df, test_df
    
    def prepare_for_model(self, df):
        """
        Returns a DataFrame with only the columns needed for model training.
        """
        return df[['image_path', 'category_encoded']]