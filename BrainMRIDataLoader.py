import os
import pandas as pd

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